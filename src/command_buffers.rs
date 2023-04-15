use std::sync::Arc;

use glam::{IVec3, UVec3};
use vec_cycle::VecCycle;
use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, ClearColorImageInfo, CommandBufferUsage,
        CopyImageInfo, ImageCopy, PrimaryAutoCommandBuffer,
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
    shaders::{self, LM_RAYS, LM_SIZE},
    LM_COUNT,
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
            images.clone(),
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
    descriptor_sets: DescriptorSets,
    images: Images,
) -> VecCycle<Arc<PrimaryAutoCommandBuffer>> {
    let dimensions: PhysicalSize<f32> = window.inner_size().cast();

    let dispatch_direct = [
        (dimensions.width / 8.0).ceil() as u32,
        (dimensions.height / 8.0).ceil() as u32,
        1,
    ];

    let dispatch_lm = [LM_COUNT * LM_SIZE / 32, LM_SIZE, LM_SIZE];

    let builder_with_direct = |offset: usize| {
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
                descriptor_sets.direct[offset].clone(),
            )
            .dispatch(dispatch_direct)
            .unwrap();

        builder
    };

    let mut command_buffers = vec![];

    for offset in 0..LM_RAYS {
        let mut builder = builder_with_direct(offset);

        let used_images = images.lightmap.useds[(offset + LM_RAYS - 1) % LM_RAYS].clone();
        for image in used_images {
            builder
                .clear_color_image(ClearColorImageInfo::image(image))
                .unwrap();
        }

        let clear = Arc::new(builder.build().unwrap());
        command_buffers.push(clear);

        for i in 0..32 {
            let mut builder = builder_with_direct(offset);

            builder
                .bind_pipeline_compute(pipelines.lm_primary[i].clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    pipelines.lm_primary[0].layout().clone(),
                    0,
                    descriptor_sets.lm_primary[offset].clone(),
                )
                .dispatch(dispatch_lm)
                .unwrap();

            let lm_primary = Arc::new(builder.build().unwrap());
            command_buffers.push(lm_primary);
        }

        for r in 1..LM_RAYS {
            for i in 0..32 {
                let mut builder = builder_with_direct(offset);

                builder
                    .bind_pipeline_compute(pipelines.lm_secondary[i].clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        pipelines.lm_secondary[0].layout().clone(),
                        0,
                        descriptor_sets.lm_secondary[(offset + r) % LM_RAYS].clone(),
                    )
                    .dispatch(dispatch_lm)
                    .unwrap();

                let lm_secondary = Arc::new(builder.build().unwrap());
                command_buffers.push(lm_secondary);
            }
        }
    }

    VecCycle::new(command_buffers)
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

    // TODO: remove duplication
    for r in 0..LM_RAYS {
        for i in 0..(LM_COUNT as usize) {
            builder
                .clear_color_image(ClearColorImageInfo::image(
                    images.lightmap.staging_useds.clone(),
                ))
                .unwrap()
                .copy_image(CopyImageInfo {
                    regions: [ImageCopy {
                        src_subresource: ImageSubresourceLayers::from_parameters(
                            images.lightmap.staging_useds.format(),
                            1,
                        ),
                        dst_subresource: ImageSubresourceLayers::from_parameters(
                            images.lightmap.useds[r][i].format(),
                            1,
                        ),
                        src_offset: src_offset_per_layer[i].to_array(), // FIXME: incorrect
                        dst_offset: dst_offset_per_layer[i].to_array(), // FIXME: incorrect
                        extent: extent_per_layer[i].to_array(), // FIXME: extent is incorrect here
                        ..ImageCopy::default()
                    }]
                    .into(),
                    ..CopyImageInfo::images(
                        images.lightmap.useds[r][i].clone(),
                        images.lightmap.staging_useds.clone(),
                    )
                })
                .unwrap()
                .copy_image(CopyImageInfo::images(
                    images.lightmap.staging_useds.clone(),
                    images.lightmap.useds[r][i].clone(),
                ))
                .unwrap();
        }
    }

    for i in 0..(LM_COUNT as usize) {
        builder
            .clear_color_image(ClearColorImageInfo::image(
                images.lightmap.staging_integers.clone(),
            ))
            .unwrap()
            .copy_image(CopyImageInfo {
                regions: [ImageCopy {
                    src_subresource: ImageSubresourceLayers::from_parameters(
                        images.lightmap.staging_integers.format(),
                        1,
                    ),
                    dst_subresource: ImageSubresourceLayers::from_parameters(
                        images.lightmap.object_hits[i].format(),
                        1,
                    ),
                    src_offset: src_offset_per_layer[i].to_array(),
                    dst_offset: dst_offset_per_layer[i].to_array(),
                    extent: extent_per_layer[i].to_array(),
                    ..ImageCopy::default()
                }]
                .into(),
                ..CopyImageInfo::images(
                    images.lightmap.object_hits[i].clone(),
                    images.lightmap.staging_integers.clone(),
                )
            })
            .unwrap()
            .copy_image(CopyImageInfo::images(
                images.lightmap.staging_integers.clone(),
                images.lightmap.object_hits[i].clone(),
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
