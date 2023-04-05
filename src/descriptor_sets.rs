use crate::lightmap::LightmapImages;
use crate::pipelines::Pipelines;

use vulkano::pipeline::Pipeline;

use vulkano::descriptor_set::WriteDescriptorSet;

use vulkano::descriptor_set::PersistentDescriptorSet;

use vulkano::image::view::ImageView;

use vulkano::image::StorageImage;

use vulkano::buffer::BufferAccess;

use std::sync::Arc;

use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;

#[derive(Clone)]
pub(crate) struct DescriptorSets {
    pub(crate) direct: Arc<PersistentDescriptorSet>,
    pub(crate) buffer_rays: Arc<PersistentDescriptorSet>,
}

pub(crate) fn get_compute_descriptor_sets(
    allocator: &StandardDescriptorSetAllocator,
    pipelines: Pipelines,
    real_time_buffer: Arc<dyn BufferAccess>,
    bvh_buffer: Arc<dyn BufferAccess>,
    mutable_buffer: Arc<dyn BufferAccess>,
    color_image: Arc<StorageImage>,
    lightmap_images: LightmapImages,
    blue_noise_buffer: Arc<dyn BufferAccess>,
) -> DescriptorSets {
    let color_image_view = ImageView::new_default(color_image.clone()).unwrap();
    let lightmap_image_views = lightmap_images.image_views();

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
        ],
    )
    .unwrap();

    let buffer_rays = PersistentDescriptorSet::new(
        allocator,
        pipelines.buffer_rays[0].layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::buffer(0, real_time_buffer.clone()),
            WriteDescriptorSet::buffer(1, bvh_buffer.clone()),
            WriteDescriptorSet::buffer(2, mutable_buffer.clone()),
            WriteDescriptorSet::image_view_array(3, 0, flattened_colors.clone()),
            WriteDescriptorSet::image_view_array(4, 0, lightmap_image_views.syncs.clone()),
            WriteDescriptorSet::buffer(5, blue_noise_buffer.clone()),
        ],
    )
    .unwrap();

    DescriptorSets {
        direct,
        buffer_rays,
    }
}
