use crate::allocators::Allocators;
use crate::buffers::Buffers;
use crate::images::Images;

use crate::pipelines::Pipelines;
use crate::shaders::LM_RAYS;

use vulkano::pipeline::Pipeline;

use vulkano::descriptor_set::WriteDescriptorSet;

use vulkano::descriptor_set::PersistentDescriptorSet;

use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct DescriptorSets {
    pub(crate) direct: Arc<PersistentDescriptorSet>,
    pub(crate) lm_init: Arc<PersistentDescriptorSet>,
    pub(crate) lm_primary: Arc<PersistentDescriptorSet>,
    pub(crate) lm_secondary: Vec<Arc<PersistentDescriptorSet>>,
}

pub(crate) fn create_compute_descriptor_sets(
    allocators: Arc<Allocators>,
    pipelines: Pipelines,
    buffers: Buffers,
    images: Images,
) -> DescriptorSets {
    let image_views = images.image_views();

    let direct = PersistentDescriptorSet::new(
        &allocators.descriptor_set,
        pipelines.direct.layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::buffer(0, buffers.real_time.clone()),
            WriteDescriptorSet::buffer(1, buffers.bvh.clone()),
            WriteDescriptorSet::image_view(2, image_views.color.clone()),
            WriteDescriptorSet::image_view_array(3, 0, image_views.lightmap.colors.last().unwrap().clone()),
        ],
    )
    .unwrap();

    let lm_init = PersistentDescriptorSet::new(
        &allocators.descriptor_set,
        pipelines.lm_init.layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::buffer(0, buffers.real_time.clone()),
            WriteDescriptorSet::buffer(1, buffers.bvh.clone()),
            WriteDescriptorSet::buffer(2, buffers.lm_buffer.clone()), // writes to
            WriteDescriptorSet::buffer(3, buffers.lm_dispatch.clone()), // writes to and reads from
        ],
    )
    .unwrap();

    let lm_primary = PersistentDescriptorSet::new(
        &allocators.descriptor_set,
        pipelines.lm_primary.layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::buffer(0, buffers.real_time.clone()),
            WriteDescriptorSet::buffer(1, buffers.bvh.clone()),
            WriteDescriptorSet::buffer(2, buffers.mutable.clone()),
            WriteDescriptorSet::image_view_array(3, 0, image_views.lightmap.colors[0].clone()), // writes to
            WriteDescriptorSet::buffer(4, buffers.lm_buffer.clone()), // reads from
            WriteDescriptorSet::buffer(5, buffers.blue_noise.clone()),
        ],
    )
    .unwrap();

    let lm_secondary = (0..(LM_RAYS - 1))
        .map(|r| {
            PersistentDescriptorSet::new(
                &allocators.descriptor_set,
                pipelines.lm_secondary.layout().set_layouts()[0].clone(),
                [
                    WriteDescriptorSet::buffer(0, buffers.real_time.clone()),
                    WriteDescriptorSet::buffer(1, buffers.bvh.clone()),
                    WriteDescriptorSet::buffer(2, buffers.mutable.clone()),
                    WriteDescriptorSet::image_view_array(
                        3,
                        0,
                        image_views.lightmap.colors[r].clone(),
                    ), // reads from
                    WriteDescriptorSet::image_view_array(
                        4,
                        0,
                        image_views.lightmap.colors[(r + 1) % LM_RAYS].clone(),
                    ), // writes to
                    WriteDescriptorSet::buffer(5, buffers.lm_buffer.clone()), // reads from
                    WriteDescriptorSet::buffer(6, buffers.blue_noise.clone()),
                ],
            )
            .unwrap()
        })
        .collect();

    DescriptorSets {
        direct,
        lm_init,
        lm_primary,
        lm_secondary,
    }
}
