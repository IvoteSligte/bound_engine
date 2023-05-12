use std::sync::Arc;

use glam::{IVec3, UVec3};
use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, CommandBufferUsage, CopyImageInfo, ImageCopy,
        PrimaryAutoCommandBuffer,
    },
    descriptor_set::DescriptorSetsCollection,
    device::Queue,
    image::{ImageAccess, ImageSubresourceLayers},
    pipeline::{Pipeline, PipelineBindPoint},
    sampler::Filter,
};
use winit::{dpi::PhysicalSize, window::Window};

use crate::{
    allocators::Allocators,
    descriptor_sets::{DescriptorSets, DescriptorUnit},
    images::Images,
    pipelines::Pipelines,
    shaders::LM_SIZE,
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
    Init,
    InitToRender,
    Render { point_count: u32 },
}

#[derive(Clone)]
pub(crate) struct PathtraceCommandBuffers {
    pub(crate) lm_init: Arc<PrimaryAutoCommandBuffer>,
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
        self.state = LmPathtraceState::Init;
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

    fn create_lm_init_command_buffer(
        allocators: Arc<Allocators>,
        queue: Arc<Queue>,
        pipelines: Pipelines,
        descriptor_sets: DescriptorSets,
        dispatch_direct: [u32; 3],
    ) -> Arc<PrimaryAutoCommandBuffer> {
        let dispatch_lm_init = [LM_SIZE / 4 * LM_COUNT, LM_SIZE / 4, LM_SIZE / 4];

        let mut builder = AutoCommandBufferBuilder::primary(
            &allocators.command_buffer,
            queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        )
        .unwrap();

        builder
            .bind_pipeline_compute(pipelines.lm_init.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipelines.lm_init.layout().clone(),
                0,
                descriptor_sets.lm_init.clone(),
            )
            .dispatch(dispatch_lm_init)
            .unwrap();

        // direct
        PathtraceCommandBuffers::extend_with_direct(
            &mut builder,
            pipelines.clone(),
            descriptor_sets.units[0].direct.clone(),
            dispatch_direct,
        );

        Arc::new(builder.build().unwrap())
    }

    pub(crate) fn create_lm_primary_command_buffer(
        allocators: Arc<Allocators>,
        queue: Arc<Queue>,
        pipelines: Pipelines,
        descriptor_unit: DescriptorUnit,
        dispatch_direct: [u32; 3],
        dispatch_lm_render: [u32; 3],
    ) -> Arc<PrimaryAutoCommandBuffer> {
        let mut builder = AutoCommandBufferBuilder::primary(
            &allocators.command_buffer,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // lm_primary
        builder
            .bind_pipeline_compute(pipelines.lm_primary.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipelines.lm_primary.layout().clone(),
                0,
                descriptor_unit.lm_primary.clone(),
            )
            .dispatch(dispatch_lm_render)
            .unwrap();

        // direct
        PathtraceCommandBuffers::extend_with_direct(
            &mut builder,
            pipelines,
            descriptor_unit.direct.clone(),
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

        let lm_init = Self::create_lm_init_command_buffer(
            allocators.clone(),
            queue.clone(),
            pipelines.clone(),
            descriptor_sets.clone(),
            dispatch_direct,
        );

        PathtraceCommandBuffers {
            lm_init,
            state: LmPathtraceState::Init,
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
