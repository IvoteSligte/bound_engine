use crate::lightmap::LightmapBufferSet;
use crate::lightmap::LightmapImages;
use crate::pipelines::PathtracePipelines;

use vec_cycle::VecCycle;
use vulkano::pipeline::Pipeline;

use vulkano::descriptor_set::WriteDescriptorSet;

use vulkano::descriptor_set::PersistentDescriptorSet;

use vulkano::image::view::ImageView;

use vulkano::image::StorageImage;

use vulkano::buffer::BufferAccess;

use vulkano::pipeline::ComputePipeline;

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
    pub(crate) move_colors: Vec<Arc<PersistentDescriptorSet>>,
}

pub(crate) fn get_compute_descriptor_sets(
    allocator: &StandardDescriptorSetAllocator,
    pathtrace_pipelines: PathtracePipelines,
    move_lightmap_pipelines: Vec<Arc<ComputePipeline>>,
    real_time_buffer: Arc<dyn BufferAccess>,
    bvh_buffer: Arc<dyn BufferAccess>,
    mutable_buffer: Arc<dyn BufferAccess>,
    constant_buffer: Arc<dyn BufferAccess>,
    color_image: Arc<StorageImage>,
    lightmap_images: LightmapImages,
    mut lightmap_buffers: LightmapBufferSet,
) -> DescriptorSetCollection {
    let color_image_view = ImageView::new_default(color_image.clone()).unwrap();

    let lightmap_image_views = lightmap_images.image_views();

    lightmap_buffers.restart();

    let ray_units = (0..2)
        .map(|_| {
            let lm_buffer_units = lightmap_buffers.next().unwrap();

            let direct = PersistentDescriptorSet::new(
                allocator,
                pathtrace_pipelines.direct.layout().set_layouts()[0].clone(),
                [
                    WriteDescriptorSet::buffer(0, real_time_buffer.clone()),
                    WriteDescriptorSet::buffer(1, bvh_buffer.clone()),
                    WriteDescriptorSet::buffer(2, constant_buffer.clone()),
                    WriteDescriptorSet::image_view(3, color_image_view.clone()),
                    WriteDescriptorSet::image_view_array(4, 0, lightmap_image_views.colors.clone()),
                    WriteDescriptorSet::image_view_array(5, 0, lightmap_image_views.syncs.clone()),
                    WriteDescriptorSet::buffer(6, lm_buffer_units[0].buffer.clone()),
                    WriteDescriptorSet::buffer(7, lm_buffer_units[0].counters.clone()),
                ],
            )
            .unwrap();

            let buffer_rays = PersistentDescriptorSet::new(
                allocator,
                pathtrace_pipelines.buffer_rays.layout().set_layouts()[0].clone(),
                [
                    WriteDescriptorSet::buffer(0, real_time_buffer.clone()),
                    WriteDescriptorSet::buffer(1, bvh_buffer.clone()),
                    WriteDescriptorSet::buffer(2, mutable_buffer.clone()),
                    WriteDescriptorSet::image_view_array(3, 0, lightmap_image_views.colors.clone()),
                    WriteDescriptorSet::image_view_array(4, 0, lightmap_image_views.syncs.clone()),
                    WriteDescriptorSet::buffer(5, lm_buffer_units[0].buffer.clone()),
                    WriteDescriptorSet::buffer(6, lm_buffer_units[0].counters.clone()),
                    WriteDescriptorSet::buffer(7, lm_buffer_units[1].buffer.clone()),
                    WriteDescriptorSet::buffer(8, lm_buffer_units[1].counters.clone()),
                ],
            )
            .unwrap();

            DescriptorSetUnit {
                direct,
                buffer_rays,
            }
        })
        .collect::<Vec<_>>();

    // FIXME: (general fix, move lightmap_image_views.syncs as well)
    let move_colors = lightmap_image_views
        .colors
        .iter()
        .zip(move_lightmap_pipelines.iter())
        .map(|(lm_view, lm_pipeline)| {
            PersistentDescriptorSet::new(
                allocator,
                lm_pipeline.layout().set_layouts()[0].clone(),
                [
                    WriteDescriptorSet::buffer(0, real_time_buffer.clone()),
                    WriteDescriptorSet::image_view(1, lm_view.clone()),
                    WriteDescriptorSet::image_view(2, lightmap_image_views.staging.clone()),
                ],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    DescriptorSetCollection {
        ray_units: VecCycle::new(ray_units),
        move_colors,
    }
}
