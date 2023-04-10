use std::sync::Arc;

use glam::{IVec3, UVec3};
use vec_cycle::VecCycle;
use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, ClearColorImageInfo, CommandBufferUsage,
        CopyImageInfo, FillBufferInfo, ImageCopy, PrimaryAutoCommandBuffer,
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
    descriptor_sets::{DescriptorSetCollection, create_compute_descriptor_sets},
    images::Images,
    pipelines::Pipelines,
    shaders::{self, ITEM_COUNT, LIGHTMAP_SIZE},
    LIGHTMAP_COUNT,
};

#[derive(Clone)]
pub(crate) struct CommandBuffers {
    pub(crate) pathtraces: VecCycle<Arc<PrimaryAutoCommandBuffer>>,
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

pub(crate) fn create_pathtrace_command_buffers(
    allocators: Arc<Allocators>,
    queue: Arc<Queue>,
    pipelines: Pipelines,
    window: Arc<Window>,
    mut descriptor_sets: DescriptorSetCollection,
    mut buffers: Buffers,
) -> VecCycle<Arc<PrimaryAutoCommandBuffer>> {
    let dimensions: PhysicalSize<f32> = window.inner_size().cast();

    let dispatch_direct = [
        (dimensions.width / 8.0).ceil() as u32,
        (dimensions.height / 8.0).ceil() as u32,
        1,
    ];

    let dispatch_buffer_rays = [ITEM_COUNT, 1, 1];

    buffers.lightmap.restart();

    VecCycle::new(
        (0..2)
            .map(|_| {
                let lm_unit = buffers.lightmap.next().unwrap();
                let desc_unit = descriptor_sets.ray_units.next().unwrap();

                let mut builder = AutoCommandBufferBuilder::primary(
                    &allocators.command_buffer,
                    queue.queue_family_index(),
                    CommandBufferUsage::SimultaneousUse, // TODO: multiplesubmit
                )
                .unwrap();

                builder
                    .fill_buffer(FillBufferInfo::dst_buffer(lm_unit[1].counters.clone())) // clear buffer
                    .unwrap()
                    .bind_pipeline_compute(pipelines.direct.clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        pipelines.direct.layout().clone(),
                        0,
                        desc_unit.direct.clone(),
                    )
                    .dispatch(dispatch_direct)
                    .unwrap()
                    .bind_pipeline_compute(pipelines.buffer_rays.clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        pipelines.buffer_rays.layout().clone(),
                        0,
                        desc_unit.buffer_rays.clone(),
                    )
                    .dispatch(dispatch_buffer_rays)
                    .unwrap();

                Arc::new(builder.build().unwrap())
            })
            .collect(),
    )
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
    const LIGHTMAP_SIZE_I: i32 = LIGHTMAP_SIZE as i32;

    const SMALLEST_UNIT: f32 = 0.5;
    // TODO: check if units_moved is less than LIGHTMAP_SIZE cause otherwise this is useless
    let units_moved_per_layer = (0..LIGHTMAP_COUNT)
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
            for i in 0..LIGHTMAP_COUNT {
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

    // TODO: clear regions that are not copied to
    for i in 0..LIGHTMAP_COUNT {
        builder
            .clear_color_image(ClearColorImageInfo::image(
                images.lightmap.staging_sync.clone(),
            ))
            .unwrap()
            .copy_image(CopyImageInfo {
                regions: [ImageCopy {
                    src_subresource: ImageSubresourceLayers::from_parameters(
                        images.lightmap.staging_sync.format(),
                        1,
                    ),
                    dst_subresource: ImageSubresourceLayers::from_parameters(
                        images.lightmap.syncs[i].format(),
                        1,
                    ),
                    src_offset: src_offset_per_layer[i].to_array(),
                    dst_offset: dst_offset_per_layer[i].to_array(),
                    extent: extent_per_layer[i].to_array(),
                    ..ImageCopy::default()
                }]
                .into(),
                ..CopyImageInfo::images(
                    images.lightmap.syncs[i].clone(),
                    images.lightmap.staging_sync.clone(),
                )
            })
            .unwrap()
            .copy_image(CopyImageInfo::images(
                images.lightmap.staging_sync.clone(),
                images.lightmap.syncs[i].clone(),
            ))
            .unwrap();
    }

    Arc::new(builder.build().unwrap())
}

pub(crate) fn create_real_time_command_buffer(
    allocators: Arc<Allocators>,
    queue: Arc<Queue>,
    real_time_data: shaders::ty::RealTimeBuffer,
    buffers: Buffers,
) -> PrimaryAutoCommandBuffer {
    let mut real_time_command_buffer_builder = AutoCommandBufferBuilder::primary(
        &allocators.command_buffer,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    real_time_command_buffer_builder
        .update_buffer(Arc::new(real_time_data), buffers.real_time.clone(), 0) // TODO: replace with copy_buffer using staging buffer
        .unwrap();

    real_time_command_buffer_builder.build().unwrap()
}
