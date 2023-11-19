use crate::allocator::Allocators;
use crate::buffer::Buffers;

use crate::image::Images;
use crate::pipeline::Pipelines;

use vulkano::image::ImageViewAbstract;
use vulkano::image::view::ImageView;
use vulkano::pipeline::Pipeline;

use vulkano::descriptor_set::WriteDescriptorSet;

use vulkano::descriptor_set::PersistentDescriptorSet;

use std::sync::Arc;

#[derive(Clone)]
pub struct DescriptorSets {
    pub direct: Arc<PersistentDescriptorSet>,
    pub init_dynamic_particles: Vec<Arc<PersistentDescriptorSet>>,
    pub clear_grid: Vec<Arc<PersistentDescriptorSet>>,
    pub dynamic_particles: Vec<Arc<PersistentDescriptorSet>>,
    pub static_particles: Vec<Arc<PersistentDescriptorSet>>,
}

impl DescriptorSets {
    pub fn new(
        allocators: Arc<Allocators>,
        pipelines: Pipelines,
        buffers: Buffers,
        images: Images,
    ) -> DescriptorSets {
        let energy_grid_views: Vec<_> = images
            .energy
            .iter()
            .map(|img| ImageView::new_default(img.clone()).unwrap() as Arc<dyn ImageViewAbstract>)
            .collect();

        let direct = PersistentDescriptorSet::new(
            &allocators.descriptor_set,
            pipelines.direct.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, buffers.real_time.clone()),
                WriteDescriptorSet::buffer(1, buffers.vertex.clone()),
                WriteDescriptorSet::buffer(2, buffers.vertex_idxs.clone()),
                WriteDescriptorSet::image_view_array(3, 0, energy_grid_views.clone()),
            ],
        )
        .unwrap();

        let mut init_dynamic_particles = vec![];

        for i in 0..3 {
            let set = PersistentDescriptorSet::new(
                &allocators.descriptor_set,
                pipelines.init_dynamic_particles.layout().set_layouts()[0].clone(),
                [WriteDescriptorSet::buffer(
                    0,
                    buffers.dynamic_particles[i].clone(),
                )],
            )
            .unwrap();

            init_dynamic_particles.push(set);
        }

        let mut clear_grid = vec![];

        for i in 0..3 {
            let set = PersistentDescriptorSet::new(
                &allocators.descriptor_set,
                pipelines.clear_grid.layout().set_layouts()[0].clone(),
                [
                    WriteDescriptorSet::buffer(0, buffers.grid[i].clone()),
                    WriteDescriptorSet::image_view(1, energy_grid_views[i].clone()),
                ],
            )
            .unwrap();

            clear_grid.push(set);
        }

        let mut dynamic_particles = vec![];

        for i in 0..3 {
            let set = PersistentDescriptorSet::new(
                &allocators.descriptor_set,
                pipelines.dynamic_particles.layout().set_layouts()[0].clone(),
                [
                    WriteDescriptorSet::buffer(0, buffers.grid[i].clone()),
                    WriteDescriptorSet::buffer(1, buffers.dynamic_particles[i].clone()),
                    WriteDescriptorSet::image_view(2, energy_grid_views[i].clone())
                ],
            )
            .unwrap();

            dynamic_particles.push(set);
        }

        let mut static_particles = vec![];

        for i in 0..3 {
            let set = PersistentDescriptorSet::new(
                &allocators.descriptor_set,
                pipelines.static_particles.layout().set_layouts()[0].clone(),
                [
                    WriteDescriptorSet::buffer(0, buffers.grid[i].clone()),
                    WriteDescriptorSet::buffer(1, buffers.static_particles[i].clone()),
                    WriteDescriptorSet::image_view(2, energy_grid_views[i].clone())
                ],
            )
            .unwrap();

            static_particles.push(set);
        }

        DescriptorSets {
            direct,
            init_dynamic_particles,
            clear_grid,
            dynamic_particles,
            static_particles,
        }
    }
}
