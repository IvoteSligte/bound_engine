use std::sync::Arc;

use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, CommandBufferUsage, PrimaryAutoCommandBuffer,
        RenderPassBeginInfo, SubpassContents,
    },
    device::Queue,
    pipeline::{Pipeline, PipelineBindPoint},
    render_pass::Framebuffer,
    sampler::Filter,
};

use crate::{
    allocator::Allocators,
    descriptor_sets::DescriptorSets,
    image::Images,
    pipeline::Pipelines,
    shaders::{LM_SIZE, RADIANCE_SIZE},
    LM_LAYERS,
};

#[derive(Clone)]
pub struct CommandBuffers {
    pub pathtraces: PathtraceCommandBuffers,
    pub swapchains: Vec<Arc<PrimaryAutoCommandBuffer>>,
}

impl CommandBuffers {
    pub fn new(
        allocators: Arc<Allocators>,
        queue: Arc<Queue>,
        frame_buffer: Arc<Framebuffer>,
        pipelines: Pipelines,
        images: Images,
        descriptor_sets: DescriptorSets,
    ) -> CommandBuffers {
        let pathtraces = PathtraceCommandBuffers::new(
            allocators.clone(),
            queue.clone(),
            frame_buffer,
            pipelines,
            descriptor_sets,
        );

        let swapchains = swapchain(allocators.clone(), queue.clone(), images.clone());

        CommandBuffers {
            pathtraces,
            swapchains,
        }
    }
}

#[derive(Clone, Debug)]
pub enum PathTraceState {
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
pub struct PathtraceCommandBuffers {
    pub sdf: Arc<PrimaryAutoCommandBuffer>,
    pub radiance: Vec<Arc<PrimaryAutoCommandBuffer>>,
    pub direct: Arc<PrimaryAutoCommandBuffer>,
    state: PathTraceState,
}

impl PathtraceCommandBuffers {
    pub fn new(
        allocators: Arc<Allocators>,
        queue: Arc<Queue>,
        frame_buffer: Arc<Framebuffer>,
        pipelines: Pipelines,
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

        let direct = Self::direct(
            allocators,
            queue,
            frame_buffer.clone(),
            pipelines,
            descriptor_sets,
        );

        PathtraceCommandBuffers {
            sdf,
            radiance,
            direct,
            state: PathTraceState::Sdf,
        }
    }

    pub fn restart(&mut self) {
        self.state = PathTraceState::Sdf;
    }

    pub fn next(&mut self) -> Arc<PrimaryAutoCommandBuffer> {
        match self.state.next() {
            PathTraceState::Sdf => self.sdf.clone(),
            PathTraceState::Radiance(frame) => self.radiance[frame % 2].clone(),
        }
    }

    pub fn direct(
        allocators: Arc<Allocators>,
        queue: Arc<Queue>,
        frame_buffer: Arc<Framebuffer>,
        pipelines: Pipelines,
        descriptor_sets: DescriptorSets,
    ) -> Arc<PrimaryAutoCommandBuffer> {
        let mut builder = AutoCommandBufferBuilder::primary(
            &allocators.command_buffer,
            queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        )
        .unwrap();

        builder
            .begin_render_pass(
                RenderPassBeginInfo { clear_values: vec![None], ..RenderPassBeginInfo::framebuffer(frame_buffer) },
                SubpassContents::Inline,
            )
            .unwrap()
            .bind_pipeline_graphics(pipelines.direct.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipelines.direct.layout().clone(),
                0,
                descriptor_sets.direct.clone(),
            )
            .draw(3, 1, 0, 0)
            .unwrap()
            .end_render_pass()
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

    pub fn radiance(
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
}

pub fn swapchain(
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
                    ..BlitImageInfo::images(images.render.clone(), swapchain_image.clone())
                })
                .unwrap();

            Arc::new(builder.build().unwrap())
        })
        .collect()
}
