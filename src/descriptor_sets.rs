use crate::allocators::Allocators;
use crate::buffers::Buffers;
use crate::images::Images;

use crate::pipelines::Pipelines;
use crate::shaders::LM_RAYS;

use vulkano::pipeline::Pipeline;

use vulkano::descriptor_set::WriteDescriptorSet;

use vulkano::descriptor_set::PersistentDescriptorSet;

use std::iter::repeat;
use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct DescriptorSets {
    pub(crate) direct: Arc<PersistentDescriptorSet>,
    pub(crate) lm_init: Arc<PersistentDescriptorSet>,
    pub(crate) lm_primary: Arc<PersistentDescriptorSet>,
    pub(crate) lm_secondary: Vec<Arc<PersistentDescriptorSet>>,
}

impl DescriptorSets {
    pub(crate) fn new(
        allocators: Arc<Allocators>,
        pipelines: Pipelines,
        buffers: Buffers,
        images: Images,
    ) -> DescriptorSets {
        let image_views = images.image_views(); // TODO: change image usage here to optimize

        let combined_image_sampler_sdfs = image_views
            .lightmap
            .sdfs_sampled
            .iter()
            .cloned()
            .zip(repeat(images.sampler()))
            .collect::<Vec<_>>();

        let combined_image_sampler_final_colors = image_views
            .lightmap
            .colors_final_sampled
            .iter()
            .cloned()
            .zip(repeat(images.sampler()))
            .collect::<Vec<_>>();

        let direct = PersistentDescriptorSet::new(
            &allocators.descriptor_set,
            pipelines.direct.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, buffers.real_time.clone()),
                WriteDescriptorSet::image_view(1, image_views.color.clone()),
                WriteDescriptorSet::image_view_sampler_array(
                    2,
                    0,
                    combined_image_sampler_final_colors,
                ),
                WriteDescriptorSet::image_view_sampler_array(
                    3,
                    0,
                    combined_image_sampler_sdfs.clone(),
                ),
            ],
        )
        .unwrap();

        let lm_init = PersistentDescriptorSet::new(
            &allocators.descriptor_set,
            pipelines.lm_init.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, buffers.real_time.clone()),
                WriteDescriptorSet::buffer(1, buffers.objects.clone()),
                WriteDescriptorSet::buffer(2, buffers.lm_buffers.gpu.clone()), // writes to
                WriteDescriptorSet::buffer(3, buffers.lm_buffers.counter.clone()), // writes to and reads from
                WriteDescriptorSet::image_view_array(
                    4,
                    0,
                    image_views.lightmap.sdfs_storage.clone(),
                ),
                WriteDescriptorSet::image_view_array(
                    5,
                    0,
                    image_views.lightmap.materials_storage.clone(),
                ),
            ],
        )
        .unwrap();

        let lm_primary = PersistentDescriptorSet::new(
            &allocators.descriptor_set,
            pipelines.lm_primary.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, buffers.real_time.clone()),
                WriteDescriptorSet::buffer(1, buffers.mutable.clone()),
                WriteDescriptorSet::image_view_array(
                    2,
                    0,
                    image_views.lightmap.colors_storage[0].clone(),
                ), // writes to
                WriteDescriptorSet::buffer(3, buffers.lm_buffers.gpu.clone()), // reads from
                WriteDescriptorSet::buffer(4, buffers.noise.clone()),
                WriteDescriptorSet::image_view_sampler_array(
                    5,
                    0,
                    combined_image_sampler_sdfs.clone(),
                ),
                WriteDescriptorSet::image_view_array(
                    6,
                    0,
                    image_views.lightmap.materials_storage.clone(),
                ),
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
                        WriteDescriptorSet::buffer(1, buffers.mutable.clone()),
                        WriteDescriptorSet::image_view_array(
                            2,
                            0,
                            image_views.lightmap.colors_storage[r].clone(),
                        ), // reads from
                        WriteDescriptorSet::image_view_array(
                            3,
                            0,
                            image_views.lightmap.colors_storage[r + 1].clone(),
                        ), // writes to
                        WriteDescriptorSet::buffer(4, buffers.lm_buffers.gpu.clone()), // reads from
                        WriteDescriptorSet::buffer(5, buffers.noise.clone()),
                        WriteDescriptorSet::image_view_sampler_array(
                            6,
                            0,
                            combined_image_sampler_sdfs.clone(),
                        ),
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
}
