use std::sync::Arc;

use glam::UVec3;
use vec_cycle::VecCycle;
use vulkano::{
    buffer::DeviceLocalBuffer,
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, BlitImageInfo,
        ClearColorImageInfo, CommandBufferUsage, CopyImageInfo, FillBufferInfo,
        PrimaryAutoCommandBuffer,
    },
    device::Queue,
    image::{StorageImage, SwapchainImage},
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
    pub(crate) move_lightmap: Arc<PrimaryAutoCommandBuffer>,
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
        lightmap_images: &LightmapImages,
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

        let move_lightmap = get_move_lightmaps_command_buffer(
            command_buffer_allocator,
            queue.clone(),
            pipelines.clone(),
            descriptor_sets.clone(),
            lightmap_images.clone(),
        );

        CommandBufferCollection {
            pathtraces,
            swapchains,
            move_lightmap,
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

// FIXME: number of lightmaps moved is not correct, etc, etc
pub(crate) fn get_move_lightmaps_command_buffer(
    allocator: &StandardCommandBufferAllocator,
    queue: Arc<Queue>,
    pipelines: Pipelines,
    descriptor_sets: DescriptorSetCollection,
    lightmap_images: LightmapImages,
) -> Arc<PrimaryAutoCommandBuffer> {
    let dispatch_lightmap = UVec3::splat(LIGHTMAP_SIZE / 4).to_array();

    let mut builder = AutoCommandBufferBuilder::primary(
        allocator,
        queue.queue_family_index(),
        CommandBufferUsage::SimultaneousUse,
    )
    .unwrap();

    lightmap_images
        .colors
        .clone()
        .into_iter()
        .zip(descriptor_sets.move_colors.clone())
        .for_each(|(lightmaps, descriptors)| {
            for i in 0..LIGHTMAP_COUNT {
                builder
                    .bind_pipeline_compute(pipelines.move_lightmap_colors[i].clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        pipelines.move_lightmap_colors[i].layout().clone(),
                        0,
                        descriptors[i].clone(),
                    )
                    .dispatch(dispatch_lightmap)
                    .unwrap()
                    .copy_image(CopyImageInfo::images(
                        lightmap_images.staging_color.clone(),
                        lightmaps[i].clone(),
                    ))
                    .unwrap();
            }
        });

    for i in 0..LIGHTMAP_COUNT {
        builder
            .bind_pipeline_compute(pipelines.move_lightmap_syncs[i].clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipelines.move_lightmap_syncs[i].layout().clone(),
                0,
                descriptor_sets.move_syncs[i].clone(),
            )
            .dispatch(dispatch_lightmap)
            .unwrap()
            .clear_color_image(ClearColorImageInfo::image(lightmap_images.syncs[i].clone()))
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
