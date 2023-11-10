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
    pipeline::Pipelines, shaders,
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

#[derive(Clone)]
pub struct PathtraceCommandBuffers {
    pub dynamic_particles: Vec<Arc<PrimaryAutoCommandBuffer>>,
    pub dynamic_particles2: Vec<Arc<PrimaryAutoCommandBuffer>>,
    pub static_particles: Vec<Arc<PrimaryAutoCommandBuffer>>,
    pub static_particles2: Vec<Arc<PrimaryAutoCommandBuffer>>,
    pub clear_grid: Vec<Arc<PrimaryAutoCommandBuffer>>,
    pub direct: Arc<PrimaryAutoCommandBuffer>,
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
        let dynamic_particles = Self::dynamic_particles(
            allocators.clone(),
            queue.clone(),
            pipelines.clone(),
            descriptor_sets.clone(),
        );
        let dynamic_particles2 = Self::dynamic_particles2(
            allocators.clone(),
            queue.clone(),
            pipelines.clone(),
            descriptor_sets.clone(),
        );
        let static_particles = Self::static_particles(
            allocators.clone(),
            queue.clone(),
            pipelines.clone(),
            descriptor_sets.clone(),
            buffers.clone(),
        );
        let static_particles2 = Self::static_particles2(
            allocators.clone(),
            queue.clone(),
            pipelines.clone(),
            descriptor_sets.clone(),
            buffers.clone(),
        );
        let clear_grid = Self::clear_grid(
            allocators.clone(),
            queue.clone(),
            pipelines.clone(),
            descriptor_sets.clone(),
        );
        let direct = Self::direct(
            allocators,
            queue,
            frame_buffer,
            pipelines,
            descriptor_sets,
            buffers,
        );
        PathtraceCommandBuffers {
            dynamic_particles,
            dynamic_particles2,
            static_particles,
            static_particles2,
            clear_grid,
            direct,
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
            .draw(
                // FIXME: this will break if the index/vertex count changes
                buffers.vertex_idxs.len() as u32,
                1,
                0,
                0,
            )
            .unwrap()
            .end_render_pass()
            .unwrap();

        Arc::new(builder.build().unwrap())
    }

    pub fn clear_grid(
        allocators: Arc<Allocators>,
        queue: Arc<Queue>,
        pipelines: Pipelines,
        descriptor_sets: DescriptorSets,
    ) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
        const DISPATCH: [u32; 3] = [shaders::CELLS / 4; 3];

        let mut builders = vec![];

        for _ in 0..3 {
            let mut builder = AutoCommandBufferBuilder::primary(
                &allocators.command_buffer,
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .unwrap();

            builder
                .bind_pipeline_compute(pipelines.clear_grid.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    pipelines.clear_grid.layout().clone(),
                    1,
                    descriptor_sets.dynamic_particles.clone(),
                )
                .dispatch(DISPATCH)
                .unwrap();

            builders.push(Arc::new(builder.build().unwrap()));
        }
        builders
    }

    pub fn dynamic_particles(
        allocators: Arc<Allocators>,
        queue: Arc<Queue>,
        pipelines: Pipelines,
        descriptor_sets: DescriptorSets,
    ) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
        const DISPATCH: [u32; 3] = [shaders::DYN_PARTICLES / 64, 1, 1];

        let mut builders = vec![];

        for _ in 0..3 {
            let mut builder = AutoCommandBufferBuilder::primary(
                &allocators.command_buffer,
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .unwrap();

            builder
                .bind_pipeline_compute(pipelines.dynamic_particles.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    pipelines.dynamic_particles.layout().clone(),
                    0,
                    descriptor_sets.dynamic_particles.clone(),
                )
                .dispatch(DISPATCH)
                .unwrap();

            builders.push(Arc::new(builder.build().unwrap()));
        }
        builders
    }

    pub fn dynamic_particles2(
        allocators: Arc<Allocators>,
        queue: Arc<Queue>,
        pipelines: Pipelines,
        descriptor_sets: DescriptorSets,
    ) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
        const DISPATCH: [u32; 3] = [shaders::DYN_PARTICLES / 64, 1, 1];

        let mut builders = vec![];

        for _ in 0..3 {
            let mut builder = AutoCommandBufferBuilder::primary(
                &allocators.command_buffer,
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .unwrap();

            builder
                .bind_pipeline_compute(pipelines.dynamic_particles2.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    pipelines.dynamic_particles2.layout().clone(),
                    0,
                    descriptor_sets.dynamic_particles.clone(),
                )
                .dispatch(DISPATCH)
                .unwrap();

            builders.push(Arc::new(builder.build().unwrap()));
        }
        builders
    }

    pub fn static_particles(
        allocators: Arc<Allocators>,
        queue: Arc<Queue>,
        pipelines: Pipelines,
        descriptor_sets: DescriptorSets,
        buffers: Buffers,
    ) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
        let dispatch = [
            // FIXME: make sure the number of static particles is always a multiple of 64
            (buffers.static_particles.len() as u32) / 64,
            1,
            1,
        ];
        let mut builders = vec![];

        for _ in 0..3 {
            let mut builder = AutoCommandBufferBuilder::primary(
                &allocators.command_buffer,
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .unwrap();

            builder
                .bind_pipeline_compute(pipelines.static_particles.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    pipelines.static_particles.layout().clone(),
                    0,
                    descriptor_sets.static_particles.clone(),
                )
                .dispatch(dispatch)
                .unwrap();

            builders.push(Arc::new(builder.build().unwrap()));
        }
        builders
    }

    pub fn static_particles2(
        allocators: Arc<Allocators>,
        queue: Arc<Queue>,
        pipelines: Pipelines,
        descriptor_sets: DescriptorSets,
        buffers: Buffers,
    ) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
        let dispatch = [
            // FIXME: make sure the number of static particles is always a multiple of 64
            (buffers.static_particles.len() as u32) / 64,
            1,
            1,
        ];
        let mut builders = vec![];

        for i in 0..3 {
            let mut builder = AutoCommandBufferBuilder::primary(
                &allocators.command_buffer,
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .unwrap();

            builder
                .bind_pipeline_compute(pipelines.static_particles2.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    pipelines.static_particles2.layout().clone(),
                    0,
                    descriptor_sets.static_particles[i].clone(),
                )
                .dispatch(dispatch)
                .unwrap();

            builders.push(Arc::new(builder.build().unwrap()));
        }
        builders
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
