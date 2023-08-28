use crate::allocator::Allocators;
use crate::buffer::Buffers;
use crate::image::Images;

use crate::pipeline::Pipelines;

use vulkano::pipeline::Pipeline;

use vulkano::descriptor_set::WriteDescriptorSet;

use vulkano::descriptor_set::PersistentDescriptorSet;

use std::sync::Arc;

#[derive(Clone)]
pub struct DescriptorSets {
    pub direct: Arc<PersistentDescriptorSet>,
    pub radiance: Arc<PersistentDescriptorSet>,
    pub radiance_precalc: Arc<PersistentDescriptorSet>,
}

impl DescriptorSets {
    pub fn new(
        allocators: Arc<Allocators>,
        pipelines: Pipelines,
        buffers: Buffers,
        images: Images,
    ) -> DescriptorSets {
        let image_views = images.views(); // TODO: change image usage here to optimize

        let radiance_precalc = PersistentDescriptorSet::new(
            &allocators.descriptor_set,
            pipelines.radiance_precalc.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, buffers.vertex.clone()),
                WriteDescriptorSet::buffer(1, buffers.vertex_idxs.clone()),
                WriteDescriptorSet::buffer(2, buffers.material_idxs.clone()),
                WriteDescriptorSet::buffer(3, buffers.material.clone()),
                WriteDescriptorSet::buffer(4, buffers.radiance.clone()),
            ],
        )
        .unwrap();

        let direct = PersistentDescriptorSet::new(
            &allocators.descriptor_set,
            pipelines.direct.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, buffers.real_time.clone()),
                WriteDescriptorSet::buffer(1, buffers.vertex.clone()),
                WriteDescriptorSet::buffer(2, buffers.vertex_idxs.clone()),
                WriteDescriptorSet::image_view_sampler_array(
                    3,
                    0,
                    images.radiance.combined_image_samplers(),
                ),
            ],
        )
        .unwrap();

        let radiance = PersistentDescriptorSet::new(
            &allocators.descriptor_set,
            pipelines.radiance[0].layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, buffers.radiance.clone()),
                WriteDescriptorSet::image_view_array(1, 0, image_views.radiance.storage.clone()),
            ],
        )
        .unwrap();

        DescriptorSets {
            direct,
            radiance,
            radiance_precalc,
        }
    }
}
