use std::sync::Arc;

use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, CommandBufferUsage, PrimaryAutoCommandBuffer,
    },
    device::Queue,
    pipeline::{Pipeline, PipelineBindPoint},
    sampler::Filter,
};
use winit::{dpi::PhysicalSize, window::Window};

use crate::{
    allocators::Allocators,
    descriptor_sets::DescriptorSets,
    images::Images,
    pipelines::Pipelines,
    shaders::{LM_SIZE, RADIANCE_SIZE},
    LM_LAYERS,
};

#[derive(Clone)]
pub(crate) struct CommandBuffers {
    pub(crate) pathtraces: PathtraceCommandBuffers,
    pub(crate) swapchains: Vec<Arc<PrimaryAutoCommandBuffer>>,
}

impl CommandBuffers {
    pub(crate) fn new(
        allocators: Arc<Allocators>,
        queue: Arc<Queue>,
        pipelines: Pipelines,
        window: Arc<Window>,
        images: Images,
        descriptor_sets: DescriptorSets,
    ) -> CommandBuffers {
        let pathtraces = PathtraceCommandBuffers::new(
            allocators.clone(),
            queue.clone(),
            pipelines.clone(),
            window.clone(),
            descriptor_sets.clone(),
        );

        let swapchains = swapchain(allocators.clone(), queue.clone(), images.clone());

        CommandBuffers {
            pathtraces,
            swapchains,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum PathTraceState {
    Sdf,
    Radiance(usize),
}

impl PathTraceState {
    fn next(&mut self) -> Self {
        let old = self.clone();
        match old {
            Self::Sdf => *self = Self::Radiance(0),
            Self::Radiance(frame) => *self = Self::Radiance(frame + 1),
        }
        old
    }
}

#[derive(Clone)]
pub(crate) struct PathtraceCommandBuffers {
    pub(crate) sdf: Arc<PrimaryAutoCommandBuffer>,
    pub(crate) radiance: Vec<Arc<PrimaryAutoCommandBuffer>>,
    pub(crate) direct: Arc<PrimaryAutoCommandBuffer>,
    pub(crate) state: PathTraceState,
}

impl PathtraceCommandBuffers {
    pub fn restart(&mut self) {
        self.state = PathTraceState::Sdf;
    }

    pub fn next(&mut self) -> Arc<PrimaryAutoCommandBuffer> {
        match self.state.next() {
            PathTraceState::Sdf => self.sdf.clone(),
            PathTraceState::Radiance(frame) => self.radiance[frame % 2].clone(),
        }
    }

    pub(crate) fn calculate_direct_dispatches(window: Arc<Window>) -> [u32; 3] {
        let dimensions: PhysicalSize<f32> = window.inner_size().cast();

        [
            (dimensions.width / 8.0).ceil() as u32,
            (dimensions.height / 8.0).ceil() as u32,
            1,
        ]
    }

    pub(crate) fn direct(
        allocators: Arc<Allocators>,
        queue: Arc<Queue>,
        pipelines: Pipelines,
        descriptor_sets: DescriptorSets,
        window: Arc<Window>,
    ) -> Arc<PrimaryAutoCommandBuffer> {
        let dispatch = Self::calculate_direct_dispatches(window);

        let mut builder = AutoCommandBufferBuilder::primary(
            &allocators.command_buffer,
            queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        )
        .unwrap();

        builder
            .bind_pipeline_compute(pipelines.direct.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipelines.direct.layout().clone(),
                0,
                descriptor_sets.direct.clone(),
            )
            .dispatch(dispatch)
            .unwrap();

        Arc::new(builder.build().unwrap())
    }

    fn sdf(
        allocators: Arc<Allocators>,
        queue: Arc<Queue>,
        pipelines: Pipelines,
        descriptor_sets: DescriptorSets,
    ) -> Arc<PrimaryAutoCommandBuffer> {
        let dispatch_sdf = [LM_SIZE / 4 * LM_LAYERS, LM_SIZE / 4, LM_SIZE / 4];
        let dispatch_radiance_precalc = [
            RADIANCE_SIZE / 4 * LM_LAYERS,
            RADIANCE_SIZE / 4,
            RADIANCE_SIZE / 4,
        ];

        let mut builder = AutoCommandBufferBuilder::primary(
            &allocators.command_buffer,
            queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        )
        .unwrap();

        builder
            .bind_pipeline_compute(pipelines.sdf.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipelines.sdf.layout().clone(),
                0,
                descriptor_sets.sdf.clone(),
            )
            .dispatch(dispatch_sdf)
            .unwrap();

        // radiance precalc
        builder
            .bind_pipeline_compute(pipelines.radiance_precalc.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipelines.radiance_precalc.layout().clone(),
                0,
                descriptor_sets.radiance_precalc.clone(),
            )
            .dispatch(dispatch_radiance_precalc)
            .unwrap();

        Arc::new(builder.build().unwrap())
    }

    pub(crate) fn radiance(
        allocators: Arc<Allocators>,
        queue: Arc<Queue>,
        pipelines: Pipelines,
        descriptor_sets: DescriptorSets,
    ) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
        let dispatch_radiance = [
            RADIANCE_SIZE / 4 * LM_LAYERS,
            RADIANCE_SIZE / 4 / 2, // halved for checkerboard rendering
            RADIANCE_SIZE / 4,
        ];

        let mut cmbs = Vec::new();
        for i in 0..2 {
            let mut builder = AutoCommandBufferBuilder::primary(
                &allocators.command_buffer,
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .unwrap();

            // radiance
            builder
                .bind_pipeline_compute(pipelines.radiance[i].clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    pipelines.radiance[i].layout().clone(),
                    0,
                    descriptor_sets.radiance.clone(),
                )
                .dispatch(dispatch_radiance)
                .unwrap();

            cmbs.push(Arc::new(builder.build().unwrap()));
        }
        cmbs
    }

    pub(crate) fn new(
        allocators: Arc<Allocators>,
        queue: Arc<Queue>,
        pipelines: Pipelines,
        window: Arc<Window>,
        descriptor_sets: DescriptorSets,
    ) -> PathtraceCommandBuffers {
        let sdf = Self::sdf(
            allocators.clone(),
            queue.clone(),
            pipelines.clone(),
            descriptor_sets.clone(),
        );

        let radiance = Self::radiance(
            allocators.clone(),
            queue.clone(),
            pipelines.clone(),
            descriptor_sets.clone(),
        );

        let direct = Self::direct(allocators, queue, pipelines, descriptor_sets, window);

        PathtraceCommandBuffers {
            sdf,
            radiance,
            direct,
            state: PathTraceState::Sdf,
        }
    }
}

pub(crate) fn swapchain(
    allocators: Arc<Allocators>,
    queue: Arc<Queue>,
    images: Images,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    images
        .swapchain
        .clone()
        .into_iter()
        .map(|swapchain_image| {
            let mut builder = AutoCommandBufferBuilder::primary(
                &allocators.command_buffer,
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .unwrap();

            builder
                .blit_image(BlitImageInfo {
                    filter: Filter::Linear,
                    ..BlitImageInfo::images(images.color.clone(), swapchain_image.clone())
                })
                .unwrap();

            Arc::new(builder.build().unwrap())
        })
        .collect()
}
