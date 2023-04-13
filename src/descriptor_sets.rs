use crate::allocators::Allocators;
use crate::buffers::Buffers;
use crate::images::Images;

use crate::pipelines::Pipelines;

use vulkano::pipeline::Pipeline;

use vulkano::descriptor_set::WriteDescriptorSet;

use vulkano::descriptor_set::PersistentDescriptorSet;

use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct DescriptorSets {
    pub(crate) direct: Arc<PersistentDescriptorSet>,
    pub(crate) accumulation: Arc<PersistentDescriptorSet>,
}

pub(crate) fn create_compute_descriptor_sets(
    allocators: Arc<Allocators>,
    pipelines: Pipelines,
    buffers: Buffers,
    images: Images,
) -> DescriptorSets {
    let image_views = images.image_views();

    let flattened_colors = image_views
        .lightmap
        .colors
        .clone()
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    let direct = PersistentDescriptorSet::new(
        &allocators.descriptor_set,
        pipelines.direct.layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::buffer(0, buffers.real_time.clone()),
            WriteDescriptorSet::buffer(1, buffers.bvh.clone()),
            WriteDescriptorSet::image_view(2, image_views.color.clone()),
            WriteDescriptorSet::image_view_array(3, 0, flattened_colors.clone()),
            WriteDescriptorSet::image_view_array(4, 0, image_views.lightmap.useds.clone()),
            WriteDescriptorSet::image_view_array(5, 0, image_views.lightmap.object_hits.clone()),
            WriteDescriptorSet::image_view_array(6, 0, image_views.lightmap.levels.clone()),
        ],
    )
    .unwrap();

    let accumulation = PersistentDescriptorSet::new(
        &allocators.descriptor_set,
        pipelines.accumulation[0].layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::buffer(0, buffers.real_time.clone()),
            WriteDescriptorSet::buffer(1, buffers.bvh.clone()),
            WriteDescriptorSet::buffer(2, buffers.mutable.clone()),
            WriteDescriptorSet::image_view_array(3, 0, flattened_colors.clone()),
            WriteDescriptorSet::image_view_array(4, 0, image_views.lightmap.useds.clone()),
            WriteDescriptorSet::image_view_array(5, 0, image_views.lightmap.object_hits.clone()),
            WriteDescriptorSet::image_view_array(6, 0, image_views.lightmap.levels.clone()),
            WriteDescriptorSet::buffer(7, buffers.blue_noise.clone()),
        ],
    )
    .unwrap();

    DescriptorSets {
        direct,
        accumulation,
    }
}
