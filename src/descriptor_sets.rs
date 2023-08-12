use crate::allocator::Allocators;
use crate::buffer::Buffers;
use crate::image::Images;

use crate::pipeline::Pipelines;

use vulkano::pipeline::Pipeline;

use vulkano::descriptor_set::WriteDescriptorSet;

use vulkano::descriptor_set::PersistentDescriptorSet;

use std::iter::repeat;
use std::sync::Arc;

#[derive(Clone)]
pub struct DescriptorSets {
    pub direct: Arc<PersistentDescriptorSet>,
    pub sdf: Arc<PersistentDescriptorSet>,
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

        let combined_image_sampler_sdfs = image_views
            .sdf
            .sampled
            .iter()
            .cloned()
            .zip(repeat(images.sdf.sampler()))
            .collect::<Vec<_>>();

        let combined_image_sampler_radiances = image_views
            .radiance
            .sampled
            .iter()
            .cloned()
            .zip(repeat(images.radiance.sampler()))
            .collect::<Vec<_>>();

        let sdf = PersistentDescriptorSet::new(
            &allocators.descriptor_set,
            pipelines.sdf.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, buffers.real_time.clone()),
                WriteDescriptorSet::buffer(1, buffers.objects.clone()),
                WriteDescriptorSet::image_view_array(2, 0, image_views.sdf.storage.clone()),
            ],
        )
        .unwrap();

        let radiance_precalc = PersistentDescriptorSet::new(
            &allocators.descriptor_set,
            pipelines.radiance_precalc.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, buffers.objects.clone()),
                WriteDescriptorSet::buffer(1, buffers.materials.clone()),
                WriteDescriptorSet::buffer(2, buffers.radiance.clone()),
            ],
        )
        .unwrap();

        let direct = PersistentDescriptorSet::new(
            &allocators.descriptor_set,
            pipelines.direct.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, buffers.real_time.clone()),
                WriteDescriptorSet::image_view_sampler_array(
                    1,
                    0,
                    combined_image_sampler_sdfs.clone(),
                ),
                WriteDescriptorSet::image_view_sampler_array(
                    2,
                    0,
                    combined_image_sampler_radiances,
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
            sdf,
            radiance,
            radiance_precalc,
        }
    }
}
