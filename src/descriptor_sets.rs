use crate::lightmap::LightmapImages;
use crate::pipelines::Pipelines;
use crate::shaders;

use vulkano::buffer::Subbuffer;
use vulkano::pipeline::Pipeline;

use vulkano::descriptor_set::WriteDescriptorSet;

use vulkano::descriptor_set::PersistentDescriptorSet;

use vulkano::image::view::ImageView;

use vulkano::image::StorageImage;

use std::sync::Arc;

use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;

#[derive(Clone)]
pub(crate) struct DescriptorSets {
    pub(crate) direct: Arc<PersistentDescriptorSet>,
    pub(crate) lightmap_rays: Arc<PersistentDescriptorSet>,
}

pub(crate) fn get_compute_descriptor_sets(
    allocator: &StandardDescriptorSetAllocator,
    pipelines: Pipelines,
    bvh_buffer: Subbuffer<shaders::GpuBVH>,
    mutable_buffer: Subbuffer<shaders::MutableData>,
    color_image: Arc<StorageImage>,
    lightmap_images: LightmapImages,
    blue_noise_buffer: Subbuffer<shaders::BlueNoise>,
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
            WriteDescriptorSet::buffer(0, bvh_buffer.clone()),
            WriteDescriptorSet::image_view(1, color_image_view.clone()),
            WriteDescriptorSet::image_view_array(2, 0, flattened_colors.clone()),
            WriteDescriptorSet::image_view_array(3, 0, lightmap_image_views.syncs.clone()),
        ],
    )
    .unwrap();

    let lightmap_rays = PersistentDescriptorSet::new(
        allocator,
        pipelines.lightmap_rays[0].layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::buffer(0, bvh_buffer.clone()),
            WriteDescriptorSet::buffer(1, mutable_buffer.clone()),
            WriteDescriptorSet::image_view_array(2, 0, flattened_colors.clone()),
            WriteDescriptorSet::image_view_array(3, 0, lightmap_image_views.syncs.clone()),
            WriteDescriptorSet::buffer(4, blue_noise_buffer.clone()),
        ],
    )
    .unwrap();

    DescriptorSets {
        direct,
        lightmap_rays,
    }
}
