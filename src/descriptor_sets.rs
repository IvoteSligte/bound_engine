use crate::lightmap::LightmapBufferSet;
use crate::lightmap::LightmapImages;
use crate::pipelines::Pipelines;

use vec_cycle::VecCycle;
use vulkano::image::ImageViewAbstract;
use vulkano::pipeline::ComputePipeline;
use vulkano::pipeline::Pipeline;

use vulkano::descriptor_set::WriteDescriptorSet;

use vulkano::descriptor_set::PersistentDescriptorSet;

use vulkano::image::view::ImageView;

use vulkano::image::StorageImage;

use vulkano::buffer::BufferAccess;

use std::sync::Arc;

use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;

#[derive(Clone)]
pub(crate) struct DescriptorSetUnit {
    pub(crate) direct: Arc<PersistentDescriptorSet>,
    pub(crate) buffer_rays: Arc<PersistentDescriptorSet>,
}

#[derive(Clone)]
pub(crate) struct DescriptorSetCollection {
    pub(crate) ray_units: VecCycle<DescriptorSetUnit>,
    pub(crate) move_colors: Vec<Vec<Arc<PersistentDescriptorSet>>>,
    pub(crate) move_syncs: Vec<Arc<PersistentDescriptorSet>>,
}

pub(crate) fn get_compute_descriptor_sets(
    allocator: &StandardDescriptorSetAllocator,
    pipelines: Pipelines,
    real_time_buffer: Arc<dyn BufferAccess>,
    bvh_buffer: Arc<dyn BufferAccess>,
    mutable_buffer: Arc<dyn BufferAccess>,
    color_image: Arc<StorageImage>,
    lightmap_images: LightmapImages,
    mut lightmap_buffers: LightmapBufferSet,
    blue_noise_buffer: Arc<dyn BufferAccess>,
) -> DescriptorSetCollection {
    let color_image_view = ImageView::new_default(color_image.clone()).unwrap();
    let lightmap_image_views = lightmap_images.image_views();

    lightmap_buffers.restart();

    let ray_units = (0..2)
        .map(|_| {
            let lm_buffer_units = lightmap_buffers.next().unwrap();

            let flattened_colors = lightmap_image_views
                .colors
                .clone()
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();

            let direct = PersistentDescriptorSet::new(
                allocator,
                pipelines.direct.layout().set_layouts()[0].clone(),
                [
                    WriteDescriptorSet::buffer(0, real_time_buffer.clone()),
                    WriteDescriptorSet::buffer(1, bvh_buffer.clone()),
                    WriteDescriptorSet::image_view(2, color_image_view.clone()),
                    WriteDescriptorSet::image_view_array(3, 0, flattened_colors.clone()),
                    WriteDescriptorSet::image_view_array(4, 0, lightmap_image_views.syncs.clone()),
                    WriteDescriptorSet::buffer(5, lm_buffer_units[0].buffer.clone()),
                    WriteDescriptorSet::buffer(6, lm_buffer_units[0].counters.clone()),
                ],
            )
            .unwrap();

            let buffer_rays = PersistentDescriptorSet::new(
                allocator,
                pipelines.buffer_rays.layout().set_layouts()[0].clone(),
                [
                    WriteDescriptorSet::buffer(0, real_time_buffer.clone()),
                    WriteDescriptorSet::buffer(1, bvh_buffer.clone()),
                    WriteDescriptorSet::buffer(2, mutable_buffer.clone()),
                    WriteDescriptorSet::image_view_array(3, 0, flattened_colors.clone()),
                    WriteDescriptorSet::image_view_array(4, 0, lightmap_image_views.syncs.clone()),
                    WriteDescriptorSet::buffer(5, lm_buffer_units[0].buffer.clone()),
                    WriteDescriptorSet::buffer(6, lm_buffer_units[0].counters.clone()),
                    WriteDescriptorSet::buffer(7, lm_buffer_units[1].buffer.clone()),
                    WriteDescriptorSet::buffer(8, lm_buffer_units[1].counters.clone()),
                    WriteDescriptorSet::buffer(9, blue_noise_buffer.clone()),
                ],
            )
            .unwrap();

            DescriptorSetUnit {
                direct,
                buffer_rays,
            }
        })
        .collect::<Vec<_>>();

    let move_descriptor = |image: Arc<dyn ImageViewAbstract>, staging: Arc<dyn ImageViewAbstract>, pipeline: Arc<ComputePipeline>| {
        PersistentDescriptorSet::new(
            allocator,
            pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, real_time_buffer.clone()),
                WriteDescriptorSet::image_view(1, image.clone()),
                WriteDescriptorSet::image_view(2, staging.clone()),
            ],
        )
        .unwrap()
    };

    let move_colors = lightmap_image_views
        .colors
        .clone()
        .into_iter()
        .map(|vec| {
            vec.into_iter()
                .zip(pipelines.move_lightmap_colors.clone().into_iter())
                .map(|(image, pipeline)| move_descriptor(image, lightmap_image_views.staging_color.clone(), pipeline))
                .collect()
        })
        .collect();

    let move_syncs = lightmap_image_views
        .syncs
        .clone()
        .into_iter()
        .zip(pipelines.move_lightmap_syncs.clone().into_iter())
        .map(|(image, pipeline)| move_descriptor(image, lightmap_image_views.staging_sync.clone(), pipeline))
        .collect();

    DescriptorSetCollection {
        ray_units: VecCycle::new(ray_units),
        move_colors,
        move_syncs,
    }
}
