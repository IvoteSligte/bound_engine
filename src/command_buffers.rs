use std::sync::Arc;

use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, CommandBufferUsage, PrimaryAutoCommandBuffer,
    },
    descriptor_set::DescriptorSetsCollection,
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

        let swapchains =
            create_swapchain_command_buffers(allocators.clone(), queue.clone(), images.clone());

        CommandBuffers {
            pathtraces,
            swapchains,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum LmPathtraceState {
    Sdf,
    Render,
}

#[derive(Clone)]
pub(crate) struct PathtraceCommandBuffers {
    pub(crate) sdf: Arc<PrimaryAutoCommandBuffer>,
    pub(crate) state: LmPathtraceState,
}

impl PathtraceCommandBuffers {
    pub(crate) fn calculate_direct_dispatches(window: Arc<Window>) -> [u32; 3] {
        let dimensions: PhysicalSize<f32> = window.inner_size().cast();

        [
            (dimensions.width / 8.0).ceil() as u32,
            (dimensions.height / 8.0).ceil() as u32,
            1,
        ]
    }

    pub(crate) fn restart(&mut self) {
        self.state = LmPathtraceState::Sdf;
    }

    pub(crate) fn extend_with_direct<S>(
        cmb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        pipelines: Pipelines,
        descriptor_set: S,
        dispatch: [u32; 3],
    ) where
        S: DescriptorSetsCollection,
    {
        cmb_builder
            .bind_pipeline_compute(pipelines.direct.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipelines.direct.layout().clone(),
                0,
                descriptor_set,
            )
            .dispatch(dispatch)
            .unwrap();
    }

    fn create_sdf_command_buffer(
        allocators: Arc<Allocators>,
        queue: Arc<Queue>,
        pipelines: Pipelines,
        descriptor_sets: DescriptorSets,
        dispatch_direct: [u32; 3],
    ) -> Arc<PrimaryAutoCommandBuffer> {
        let dispatch_sdf = [LM_SIZE / 4 * LM_LAYERS, LM_SIZE / 4, LM_SIZE / 4];

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

        // direct
        PathtraceCommandBuffers::extend_with_direct(
            &mut builder,
            pipelines.clone(),
            descriptor_sets.direct.clone(),
            dispatch_direct,
        );

        Arc::new(builder.build().unwrap())
    }

    pub(crate) fn create_radiance_command_buffer(
        allocators: Arc<Allocators>,
        queue: Arc<Queue>,
        pipelines: Pipelines,
        descriptor_sets: DescriptorSets,
        dispatch_direct: [u32; 3],
    ) -> Arc<PrimaryAutoCommandBuffer> {
        let dispatch_radiance_precalc = [RADIANCE_SIZE / 4 * LM_LAYERS, RADIANCE_SIZE / 4, RADIANCE_SIZE / 4];
        let dispatch_radiance = [RADIANCE_SIZE * LM_LAYERS, RADIANCE_SIZE, RADIANCE_SIZE];

        let mut builder = AutoCommandBufferBuilder::primary(
            &allocators.command_buffer,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
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

        // radiance
        builder
            .bind_pipeline_compute(pipelines.radiance.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipelines.radiance.layout().clone(),
                0,
                descriptor_sets.radiance.clone(),
            )
            .dispatch(dispatch_radiance)
            .unwrap();

        // direct
        PathtraceCommandBuffers::extend_with_direct(
            &mut builder,
            pipelines,
            descriptor_sets.direct.clone(),
            dispatch_direct,
        );

        Arc::new(builder.build().unwrap())
    }

    pub(crate) fn new(
        allocators: Arc<Allocators>,
        queue: Arc<Queue>,
        pipelines: Pipelines,
        window: Arc<Window>,
        descriptor_sets: DescriptorSets,
    ) -> PathtraceCommandBuffers {
        let dispatch_direct = PathtraceCommandBuffers::calculate_direct_dispatches(window);

        let sdf = Self::create_sdf_command_buffer(
            allocators.clone(),
            queue.clone(),
            pipelines.clone(),
            descriptor_sets.clone(),
            dispatch_direct,
        );

        PathtraceCommandBuffers {
            sdf,
            state: LmPathtraceState::Sdf,
        }
    }
}

pub(crate) fn create_swapchain_command_buffers(
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
