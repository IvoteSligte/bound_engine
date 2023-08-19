use std::sync::Arc;

use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, CommandBufferUsage, PrimaryAutoCommandBuffer,
        RenderPassBeginInfo, SubpassContents,
    },
    device::Queue,
    format::ClearValue,
    pipeline::{Pipeline, PipelineBindPoint},
    render_pass::Framebuffer,
    sampler::Filter,
};

use crate::{
    allocator::Allocators, buffer::Buffers, descriptor_sets::DescriptorSets, image::Images,
    pipeline::Pipelines, shaders::RADIANCE_SIZE, LM_LAYERS,
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
        buffers: Buffers,
    ) -> CommandBuffers {
        let pathtraces = PathtraceCommandBuffers::new(
            allocators.clone(),
            queue.clone(),
            frame_buffer,
            pipelines,
            descriptor_sets,
            buffers,
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
    Precalc,
    Radiance(usize),
}

impl PathTraceState {
    fn next(&mut self) -> Self {
        let old = self.clone();
        match old {
            Self::Precalc => *self = Self::Radiance(0),
            Self::Radiance(frame) => *self = Self::Radiance(frame + 1),
        }
        old
    }
}

#[derive(Clone)]
pub struct PathtraceCommandBuffers {
    pub precalc: Arc<PrimaryAutoCommandBuffer>,
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
        buffers: Buffers,
    ) -> PathtraceCommandBuffers {
        let precalc = Self::radiance_precalc(
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
            buffers,
        );

        PathtraceCommandBuffers {
            precalc,
            radiance,
            direct,
            state: PathTraceState::Precalc,
        }
    }

    pub fn next(&mut self) -> Arc<PrimaryAutoCommandBuffer> {
        match self.state.next() {
            PathTraceState::Precalc => self.precalc.clone(),
            PathTraceState::Radiance(frame) => self.radiance[frame % 2].clone(),
        }
    }

    pub fn direct(
        allocators: Arc<Allocators>,
        queue: Arc<Queue>,
        frame_buffer: Arc<Framebuffer>,
        pipelines: Pipelines,
        descriptor_sets: DescriptorSets,
        buffers: Buffers,
    ) -> Arc<PrimaryAutoCommandBuffer> {
        let mut builder = AutoCommandBufferBuilder::primary(
            &allocators.command_buffer,
            queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        )
        .unwrap();

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![
                        Some(ClearValue::Float([0.0; 4])),
                        Some(ClearValue::Depth(1e20)),
                    ],
                    ..RenderPassBeginInfo::framebuffer(frame_buffer)
                },
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
            .bind_vertex_buffers(0, buffers.vertex)
            .bind_index_buffer(buffers.vertex_idxs.clone())
            .draw_indexed(
                buffers.vertex_idxs.len() as u32, // INFO: this will break if the index/vertex count changes
                1,
                0,
                0,
                0,
            )
            .unwrap()
            .end_render_pass()
            .unwrap();

        Arc::new(builder.build().unwrap())
    }

    fn radiance_precalc(
        allocators: Arc<Allocators>,
        queue: Arc<Queue>,
        pipelines: Pipelines,
        descriptor_sets: DescriptorSets,
    ) -> Arc<PrimaryAutoCommandBuffer> {
        let dispatch = [
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

        // radiance precalc
        builder
            .bind_pipeline_compute(pipelines.radiance_precalc.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipelines.radiance_precalc.layout().clone(),
                0,
                descriptor_sets.radiance_precalc.clone(),
            )
            .dispatch(dispatch)
            .unwrap();

        Arc::new(builder.build().unwrap())
    }

    pub fn radiance(
        allocators: Arc<Allocators>,
        queue: Arc<Queue>,
        pipelines: Pipelines,
        descriptor_sets: DescriptorSets,
    ) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
        let dispatch = [
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
                .dispatch(dispatch)
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
