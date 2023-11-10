use crate::allocator::Allocators;
use crate::buffer::Buffers;

use crate::pipeline::Pipelines;

use vulkano::pipeline::Pipeline;

use vulkano::descriptor_set::WriteDescriptorSet;

use vulkano::descriptor_set::PersistentDescriptorSet;

use std::sync::Arc;

#[derive(Clone)]
pub struct DescriptorSets {
    pub direct: Arc<PersistentDescriptorSet>,
    pub dynamic_particles: Vec<Arc<PersistentDescriptorSet>>,
    pub static_particles: Vec<Arc<PersistentDescriptorSet>>,
}

impl DescriptorSets {
    pub fn new(
        allocators: Arc<Allocators>,
        pipelines: Pipelines,
        buffers: Buffers,
    ) -> DescriptorSets {
        let direct = PersistentDescriptorSet::new(
            &allocators.descriptor_set,
            pipelines.direct.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, buffers.real_time.clone()),
                WriteDescriptorSet::buffer(1, buffers.vertex.clone()),
                WriteDescriptorSet::buffer(2, buffers.vertex_idxs.clone()),
            ],
        )
        .unwrap();

        let mut dynamic_particles = vec![];

        for i in 0..3 {
            let set = PersistentDescriptorSet::new(
                &allocators.descriptor_set,
                pipelines.dynamic_particles.layout().set_layouts()[0].clone(),
                [
                    WriteDescriptorSet::buffer(0, buffers.grid[i].clone()),
                    WriteDescriptorSet::buffer(1, buffers.dynamic_particles[i].clone()),
                ],
            )
            .unwrap();
            dynamic_particles.push(set);
        }

        let mut static_particles = vec![];

        for i in 0..3 {
            let set = PersistentDescriptorSet::new(
                &allocators.descriptor_set,
                pipelines.dynamic_particles.layout().set_layouts()[0].clone(),
                [
                    WriteDescriptorSet::buffer(0, buffers.grid[i].clone()),
                    WriteDescriptorSet::buffer(1, buffers.static_particles[i].clone()),
                ],
            )
            .unwrap();
            static_particles.push(set);
        }

        DescriptorSets {
            direct,
            dynamic_particles,
            static_particles,
        }
    }
}
