use std::sync::Arc;

use glam::{IVec3, UVec3};
use vec_cycle::VecCycle;
use vulkano::{
    buffer::DeviceLocalBuffer,
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, BlitImageInfo,
        ClearColorImageInfo, CommandBufferUsage, CopyImageInfo, FillBufferInfo, ImageCopy,
        PrimaryAutoCommandBuffer,
    },
    device::Queue,
    image::{ImageAccess, ImageSubresourceLayers, StorageImage, SwapchainImage},
    pipeline::{Pipeline, PipelineBindPoint},
    sampler::Filter,
};
use winit::{dpi::PhysicalSize, window::Window};

use crate::{
    descriptor_sets::DescriptorSetCollection,
    lightmap::{LightmapBufferSet, LightmapImages},
    pipelines::Pipelines,
    shaders::{self, ITEM_COUNT, LIGHTMAP_SIZE},
    LIGHTMAP_COUNT,
};

#[derive(Clone)]
pub(crate) struct CommandBufferCollection {
    pub(crate) pathtraces: VecCycle<Arc<PrimaryAutoCommandBuffer>>,
    pub(crate) swapchains: Vec<Arc<PrimaryAutoCommandBuffer>>,
}

impl CommandBufferCollection {
    pub(crate) fn new(
        command_buffer_allocator: &StandardCommandBufferAllocator,
        queue: Arc<Queue>,
        pipelines: Pipelines,
        window: Arc<Window>,
        descriptor_sets: DescriptorSetCollection,
        lightmap_buffers: &LightmapBufferSet,
        color_image: Arc<StorageImage>,
        swapchain_images: Vec<Arc<SwapchainImage>>,
    ) -> CommandBufferCollection {
        let pathtraces = get_pathtrace_command_buffers(
            command_buffer_allocator,
            queue.clone(),
            pipelines.clone(),
            window.clone(),
            descriptor_sets.clone(),
            lightmap_buffers.clone(),
        );

        let swapchains = get_swapchain_command_buffers(
            command_buffer_allocator,
            queue.clone(),
            color_image.clone(),
            swapchain_images.clone(),
        );

        CommandBufferCollection {
            pathtraces,
            swapchains,
        }
    }
}

pub(crate) fn get_pathtrace_command_buffers(
    allocator: &StandardCommandBufferAllocator,
    queue: Arc<Queue>,
    pipelines: Pipelines,
    window: Arc<Window>,
    mut descriptor_sets: DescriptorSetCollection,
    mut lightmap_buffers: LightmapBufferSet,
) -> VecCycle<Arc<PrimaryAutoCommandBuffer>> {
    let dimensions: PhysicalSize<f32> = window.inner_size().cast();

    let dispatch_direct = [
        (dimensions.width / 8.0).ceil() as u32,
        (dimensions.height / 8.0).ceil() as u32,
        1,
    ];

    let dispatch_buffer_rays = [ITEM_COUNT / 64, 1, 1];

    descriptor_sets.ray_units.restart();
    lightmap_buffers.restart();

    VecCycle::new(
        (0..2)
            .map(|_| {
                let lm_unit = lightmap_buffers.next().unwrap();
                let desc_unit = descriptor_sets.ray_units.next().unwrap();

                let mut builder = AutoCommandBufferBuilder::primary(
                    allocator,
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

pub(crate) fn get_swapchain_command_buffers(
    allocator: &StandardCommandBufferAllocator,
    queue: Arc<Queue>,
    color_image: Arc<StorageImage>,
    swapchain_images: Vec<Arc<SwapchainImage>>,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    swapchain_images // TODO: remove the double/triple buffering and just have two command buffers
        .clone()
        .into_iter()
        .map(|swapchain_image| {
            let mut builder = AutoCommandBufferBuilder::primary(
                allocator,
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .unwrap();

            builder
                .blit_image(BlitImageInfo {
                    filter: Filter::Linear,
                    ..BlitImageInfo::images(color_image.clone(), swapchain_image.clone())
                })
                .unwrap();

            Arc::new(builder.build().unwrap())
        })
        .collect()
}

pub(crate) fn get_dynamic_move_lightmaps_command_buffer(
    allocator: &StandardCommandBufferAllocator,
    queue: Arc<Queue>,
    lightmap_images: LightmapImages,
    movement: IVec3,
) -> Arc<PrimaryAutoCommandBuffer> {
    let mut builder = AutoCommandBufferBuilder::primary(
        allocator,
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

    lightmap_images
        .colors
        .clone()
        .into_iter()
        .for_each(|lightmaps| {
            for i in 0..LIGHTMAP_COUNT {
                builder
                    .copy_image(CopyImageInfo {
                        regions: [ImageCopy {
                            src_subresource: ImageSubresourceLayers::from_parameters(
                                lightmap_images.staging_color.format(),
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
                            lightmap_images.staging_color.clone(),
                        )
                    })
                    .unwrap()
                    .copy_image(CopyImageInfo::images(
                        lightmap_images.staging_color.clone(),
                        lightmaps[i].clone(),
                    ))
                    .unwrap();
            }
        });

    // TODO: clear regions that are not copied to
    for i in 0..LIGHTMAP_COUNT {
        builder
            .clear_color_image(ClearColorImageInfo::image(
                lightmap_images.staging_sync.clone(),
            ))
            .unwrap()
            .copy_image(CopyImageInfo {
                regions: [ImageCopy {
                    src_subresource: ImageSubresourceLayers::from_parameters(
                        lightmap_images.staging_sync.format(),
                        1,
                    ),
                    dst_subresource: ImageSubresourceLayers::from_parameters(
                        lightmap_images.syncs[i].format(),
                        1,
                    ),
                    src_offset: src_offset_per_layer[i].to_array(),
                    dst_offset: dst_offset_per_layer[i].to_array(),
                    extent: extent_per_layer[i].to_array(),
                    ..ImageCopy::default()
                }]
                .into(),
                ..CopyImageInfo::images(
                    lightmap_images.syncs[i].clone(),
                    lightmap_images.staging_sync.clone(),
                )
            })
            .unwrap()
            .copy_image(CopyImageInfo::images(
                lightmap_images.staging_sync.clone(),
                lightmap_images.syncs[i].clone(),
            ))
            .unwrap();
    }

    Arc::new(builder.build().unwrap())
}

pub(crate) fn get_real_time_command_buffer(
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue_family_index: u32,
    real_time_data: shaders::ty::RealTimeBuffer,
    real_time_buffer: Arc<DeviceLocalBuffer<shaders::ty::RealTimeBuffer>>,
) -> PrimaryAutoCommandBuffer {
    let mut real_time_command_buffer_builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue_family_index,
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    real_time_command_buffer_builder
        .update_buffer(Box::new(real_time_data), real_time_buffer.clone(), 0) // TODO: replace with copy_buffer using staging buffer
        .unwrap();

    real_time_command_buffer_builder.build().unwrap()
}
