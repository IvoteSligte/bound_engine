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
    pub(crate) direct: Vec<Arc<PersistentDescriptorSet>>,
    pub(crate) lm_primary: Vec<Arc<PersistentDescriptorSet>>,
    pub(crate) lm_secondary: Vec<Arc<PersistentDescriptorSet>>,
}

pub(crate) fn create_compute_descriptor_sets(
    allocators: Arc<Allocators>,
    pipelines: Pipelines,
    buffers: Buffers,
    images: Images,
) -> DescriptorSets {
    let image_views = images.image_views();

    let mut direct = vec![];
    let mut lm_primary = vec![];
    let mut lm_secondary = vec![];

    for r in 0..LM_RAYS {
        direct.push(
            PersistentDescriptorSet::new(
                &allocators.descriptor_set,
                pipelines.direct.layout().set_layouts()[0].clone(),
                [
                    WriteDescriptorSet::buffer(0, buffers.real_time.clone()),
                    WriteDescriptorSet::buffer(1, buffers.bvh.clone()),
                    WriteDescriptorSet::image_view(2, image_views.color.clone()),
                    WriteDescriptorSet::image_view_array(
                        3,
                        0,
                        image_views.lightmap.colors[LM_RAYS - 1].clone(),
                    ),
                    WriteDescriptorSet::image_view_array(
                        4,
                        0,
                        image_views.lightmap.useds[r].clone(), // writes to lowest bounce image
                    ),
                    WriteDescriptorSet::image_view_array(
                        5,
                        0,
                        image_views.lightmap.object_hits.clone(),
                    ),
                ],
            )
            .unwrap(),
        );

        const LM_RAYS_I: i32 = LM_RAYS as i32;

        lm_primary.push(
            PersistentDescriptorSet::new(
                &allocators.descriptor_set,
                pipelines.lm_primary[0].layout().set_layouts()[0].clone(),
                [
                    WriteDescriptorSet::buffer(0, buffers.real_time.clone()),
                    WriteDescriptorSet::buffer(1, buffers.bvh.clone()),
                    WriteDescriptorSet::buffer(2, buffers.mutable.clone()),
                    WriteDescriptorSet::image_view_array(
                        3,
                        0,
                        image_views.lightmap.colors[r].clone(), // writes to
                    ),
                    WriteDescriptorSet::image_view_array(
                        4,
                        0,
                        image_views.lightmap.useds[r].clone(), // reads from
                    ),
                    WriteDescriptorSet::image_view_array(
                        5,
                        0,
                        image_views.lightmap.useds[((((r as i32 - 1) % LM_RAYS_I) + LM_RAYS_I) % LM_RAYS_I) as usize].clone(), // writes to
                    ),
                    WriteDescriptorSet::image_view_array(
                        6,
                        0,
                        image_views.lightmap.object_hits.clone(), // reads from and writes to
                    ),
                    WriteDescriptorSet::buffer(7, buffers.blue_noise.clone()),
                ],
            )
            .unwrap(),
        );

        lm_secondary.push(
            PersistentDescriptorSet::new(
                &allocators.descriptor_set,
                pipelines.lm_secondary[0].layout().set_layouts()[0].clone(),
                [
                    WriteDescriptorSet::buffer(0, buffers.real_time.clone()),
                    WriteDescriptorSet::buffer(1, buffers.bvh.clone()),
                    WriteDescriptorSet::buffer(2, buffers.mutable.clone()),
                    WriteDescriptorSet::image_view_array(
                        3,
                        0,
                        image_views.lightmap.colors[r].clone(), // reads from
                    ),
                    WriteDescriptorSet::image_view_array(
                        4,
                        0,
                        image_views.lightmap.colors[(r + 1) % LM_RAYS].clone(), // writes to
                    ),
                    WriteDescriptorSet::image_view_array(
                        5,
                        0,
                        image_views.lightmap.useds[r].clone(), // reads from
                    ),
                    WriteDescriptorSet::image_view_array(
                        6,
                        0,
                        image_views.lightmap.object_hits.clone(), // reads from
                    ),
                    WriteDescriptorSet::buffer(7, buffers.blue_noise.clone()),
                ],
            )
            .unwrap(),
        );
    }

    DescriptorSets {
        direct,
        lm_primary,
        lm_secondary,
    }
}
