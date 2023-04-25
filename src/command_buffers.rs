use std::sync::Arc;

use glam::{IVec3, UVec3};
use vec_once::VecOnce;
use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, CommandBufferUsage, CopyImageInfo,
        DispatchIndirectCommand, ImageCopy, PrimaryAutoCommandBuffer,
    },
    device::Queue,
    image::{ImageAccess, ImageSubresourceLayers},
    pipeline::{Pipeline, PipelineBindPoint},
    sampler::Filter,
};
use winit::{dpi::PhysicalSize, window::Window};

use crate::{
    allocators::Allocators,
    buffers::Buffers,
    descriptor_sets::{create_compute_descriptor_sets, DescriptorSets},
    images::Images,
    pipelines::Pipelines,
    shaders::{self, LM_SIZE, LM_BUFFER_SLICES},
    LM_COUNT,
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
        buffers: Buffers,
        images: Images,
    ) -> CommandBuffers {
        let descriptor_sets = create_compute_descriptor_sets(
            allocators.clone(),
            pipelines.clone(),
            buffers.clone(),
            images.clone(),
        );

        let pathtraces = create_pathtrace_command_buffers(
            allocators.clone(),
            queue.clone(),
            pipelines.clone(),
            window.clone(),
            descriptor_sets.clone(),
            buffers.clone(),
        );

        let swapchains =
            create_swapchain_command_buffers(allocators.clone(), queue.clone(), images.clone());

        CommandBuffers {
            pathtraces,
            swapchains,
        }
    }
}

#[derive(Clone)]
pub(crate) struct PathtraceCommandBuffers {
    // TODO: implement fn next(), which returns `direct` when `acc.next()` returns `None`
    pub(crate) lightmap: VecOnce<Arc<PrimaryAutoCommandBuffer>>,
    pub(crate) direct: Arc<PrimaryAutoCommandBuffer>,
}

impl PathtraceCommandBuffers {
    pub(crate) fn next(&mut self) -> Arc<PrimaryAutoCommandBuffer> {
        self.lightmap.next().unwrap_or(self.direct.clone())
    }
}

// TODO: impl PathtraceCommandBuffers
pub(crate) fn create_pathtrace_command_buffers(
    allocators: Arc<Allocators>,
    queue: Arc<Queue>,
    pipelines: Pipelines,
    window: Arc<Window>,
    descriptor_sets: DescriptorSets,
    buffers: Buffers,
) -> PathtraceCommandBuffers {
    let dimensions: PhysicalSize<f32> = window.inner_size().cast();

    let dispatch_direct = [
        (dimensions.width / 8.0).ceil() as u32,
        (dimensions.height / 8.0).ceil() as u32,
        1,
    ];

    let dispatch_lm_init = [LM_SIZE / 4 * LM_COUNT, LM_SIZE / 4, LM_SIZE / 4];

    let create_builder = || {
        AutoCommandBufferBuilder::primary(
            &allocators.command_buffer,
            queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        )
        .unwrap()
    };

    let mut command_buffers = vec![];

    let mut builder = create_builder();
    builder
        .update_buffer(
            buffers.lm_dispatch.clone(),
            &[DispatchIndirectCommand { x: 0, y: 1, z: 1 }; LM_BUFFER_SLICES as usize][..],
        )
        .unwrap()
        .bind_pipeline_compute(pipelines.lm_init.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipelines.lm_init.layout().clone(),
            0,
            descriptor_sets.lm_init.clone(),
        )
        .dispatch(dispatch_lm_init)
        .unwrap();

    let lm_init = Arc::new(builder.build().unwrap());
    command_buffers.push(lm_init);

    for (i, lm_primary_pipeline) in pipelines.lm_primary.iter().enumerate() {
        let mut builder = create_builder();
        builder
            .bind_pipeline_compute(lm_primary_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                lm_primary_pipeline.layout().clone(),
                0,
                descriptor_sets.lm_primary.clone(),
            )
            .dispatch_indirect(buffers.lm_dispatch.clone().index(i as u64).into_slice())
            .unwrap();

        let lm_primary = Arc::new(builder.build().unwrap());
        command_buffers.push(lm_primary);
    }

    for lm_secondary_descriptor_set in descriptor_sets.lm_secondary {
        for (i, lm_secondary_pipeline) in pipelines.lm_secondary.iter().enumerate() {
            let mut builder = create_builder();
            builder
                .bind_pipeline_compute(lm_secondary_pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    lm_secondary_pipeline.layout().clone(),
                    0,
                    lm_secondary_descriptor_set.clone(),
                )
                .dispatch_indirect(buffers.lm_dispatch.clone().index(i as u64).into_slice())
                .unwrap();

            let lm_accumulate = Arc::new(builder.build().unwrap());
            command_buffers.push(lm_accumulate);
        }
    }

    let mut builder = create_builder();
    builder
        .bind_pipeline_compute(pipelines.direct.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipelines.direct.layout().clone(),
            0,
            descriptor_sets.direct.clone(),
        )
        .dispatch(dispatch_direct)
        .unwrap();

    let direct = Arc::new(builder.build().unwrap());

    PathtraceCommandBuffers {
        lightmap: VecOnce::new(command_buffers),
        direct,
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

pub(crate) fn create_dynamic_move_lightmaps_command_buffer(
    allocators: Arc<Allocators>,
    queue: Arc<Queue>,
    images: Images,
    movement: IVec3,
) -> Arc<PrimaryAutoCommandBuffer> {
    let mut builder = AutoCommandBufferBuilder::primary(
        &allocators.command_buffer,
        queue.queue_family_index(),
        CommandBufferUsage::SimultaneousUse,
    )
    .unwrap();

    // TODO: check validity
    const LIGHTMAP_SIZE_I: i32 = LM_SIZE as i32;

    const SMALLEST_UNIT: f32 = 0.5;
    // TODO: check if units_moved is less than LIGHTMAP_SIZE cause otherwise this is useless
    let units_moved_per_layer = (0..LM_COUNT)
        .map(|i| SMALLEST_UNIT * 2.0f32.powi(i as i32))
        .map(|unit_size| movement.as_vec3() / unit_size)
        .map(|units_moved| units_moved.as_ivec3())
        .collect::<Vec<_>>();

    let (src_offset_per_layer, dst_offset_per_layer): (Vec<UVec3>, Vec<UVec3>) =
        units_moved_per_layer
            .iter()
            .map(|&units_moved| {
                let (src_offset, dst_offset): (Vec<_>, Vec<_>) = units_moved
                    .to_array()
                    .into_iter()
                    .map(|n| {
                        if n.is_positive() {
                            (n.abs(), 0)
                        } else {
                            (0, n.abs())
                        }
                    })
                    .unzip();
                (
                    IVec3::from_slice(&src_offset),
                    IVec3::from_slice(&dst_offset),
                )
            })
            .map(|(a, b)| (a.as_uvec3(), b.as_uvec3()))
            .unzip();

    // TODO: check if all(extent > 0) cause otherwise this is useless as well
    let extent_per_layer = units_moved_per_layer
        .iter()
        .map(|&units_moved| LIGHTMAP_SIZE_I - units_moved.abs())
        .map(|v| v.as_uvec3())
        .collect::<Vec<_>>();

    images
        .lightmap
        .colors
        .clone()
        .into_iter()
        .for_each(|lightmaps| {
            for i in 0..(LM_COUNT as usize) {
                builder
                    .copy_image(CopyImageInfo {
                        regions: [ImageCopy {
                            src_subresource: ImageSubresourceLayers::from_parameters(
                                images.lightmap.staging_color.format(),
                                1,
                            ),
                            dst_subresource: ImageSubresourceLayers::from_parameters(
                                lightmaps[i].format(),
                                1,
                            ),
                            src_offset: src_offset_per_layer[i].to_array(),
                            dst_offset: dst_offset_per_layer[i].to_array(),
                            extent: extent_per_layer[i].to_array(),
                            ..ImageCopy::default()
                        }]
                        .into(),
                        ..CopyImageInfo::images(
                            lightmaps[i].clone(),
                            images.lightmap.staging_color.clone(),
                        )
                    })
                    .unwrap()
                    .copy_image(CopyImageInfo::images(
                        images.lightmap.staging_color.clone(),
                        lightmaps[i].clone(),
                    ))
                    .unwrap();
            }
        });

    // FIXME: moving the `lm_buffer` !!!

    Arc::new(builder.build().unwrap())
}

pub(crate) fn create_real_time_command_buffer(
    allocators: Arc<Allocators>,
    queue: Arc<Queue>,
    real_time_data: shaders::RealTimeBuffer,
    buffers: Buffers,
) -> PrimaryAutoCommandBuffer {
    let mut real_time_command_buffer_builder = AutoCommandBufferBuilder::primary(
        &allocators.command_buffer,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    real_time_command_buffer_builder
        .update_buffer(buffers.real_time.clone(), Arc::new(real_time_data))
        .unwrap();

    real_time_command_buffer_builder.build().unwrap()
}
