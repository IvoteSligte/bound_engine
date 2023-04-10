use crate::allocators::Allocators;
use crate::buffers::Buffers;
use crate::images::Images;

use crate::pipelines::Pipelines;

use vec_cycle::VecCycle;
use vulkano::pipeline::Pipeline;

use vulkano::descriptor_set::WriteDescriptorSet;

use vulkano::descriptor_set::PersistentDescriptorSet;

use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct DescriptorSetUnit {
    pub(crate) direct: Arc<PersistentDescriptorSet>,
    pub(crate) buffer_rays: Arc<PersistentDescriptorSet>,
}

#[derive(Clone)]
pub(crate) struct DescriptorSetCollection {
    pub(crate) ray_units: VecCycle<DescriptorSetUnit>,
}

pub(crate) fn create_compute_descriptor_sets(
    allocators: Arc<Allocators>,
    pipelines: Pipelines,
    mut buffers: Buffers,
    images: Images,
) -> DescriptorSetCollection {
    let image_views = images.image_views();

    buffers.lightmap.restart();

    let ray_units = (0..2)
        .map(|_| {
            let lm_buffer_units = buffers.lightmap.next().unwrap();

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
                    WriteDescriptorSet::image_view_array(4, 0, image_views.lightmap.syncs.clone()),
                    WriteDescriptorSet::buffer(5, lm_buffer_units[0].buffer.clone()),
                    WriteDescriptorSet::buffer(6, lm_buffer_units[0].counters.clone()),
                ],
            )
            .unwrap();

            let buffer_rays = PersistentDescriptorSet::new(
                &allocators.descriptor_set,
                pipelines.buffer_rays.layout().set_layouts()[0].clone(),
                [
                    WriteDescriptorSet::buffer(0, buffers.real_time.clone()),
                    WriteDescriptorSet::buffer(1, buffers.bvh.clone()),
                    WriteDescriptorSet::buffer(2, buffers.mutable.clone()),
                    WriteDescriptorSet::image_view_array(3, 0, flattened_colors.clone()),
                    WriteDescriptorSet::image_view_array(4, 0, image_views.lightmap.syncs.clone()),
                    WriteDescriptorSet::buffer(5, lm_buffer_units[0].buffer.clone()),
                    WriteDescriptorSet::buffer(6, lm_buffer_units[0].counters.clone()),
                    WriteDescriptorSet::buffer(7, lm_buffer_units[1].buffer.clone()),
                    WriteDescriptorSet::buffer(8, lm_buffer_units[1].counters.clone()),
                    WriteDescriptorSet::buffer(9, buffers.blue_noise.clone()),
                ],
            )
            .unwrap();

            DescriptorSetUnit {
                direct,
                buffer_rays,
            }
        })
        .collect::<Vec<_>>();

    DescriptorSetCollection {
        ray_units: VecCycle::new(ray_units),
    }
}
