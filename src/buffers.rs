use std::sync::Arc;

use rand_distr::{Distribution, UnitSphere};
use vulkano::{
    buffer::{BufferAccess, BufferUsage, DeviceLocalBuffer},
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    memory::allocator::{FreeListAllocator, GenericMemoryAllocator},
};

use crate::{
    bvh::CpuBVH,
    scene,
    shaders::{self, SAMPLES},
};

pub(crate) fn get_mutable_buffer(
    memory_allocator: &GenericMemoryAllocator<Arc<FreeListAllocator>>,
    alloc_command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
) -> Arc<DeviceLocalBuffer<shaders::ty::MutableData>> {
    let mut mutable_data = shaders::ty::MutableData {
        ..Default::default()
    };
    mutable_data.mats[..scene::MATERIALS.len()].copy_from_slice(&scene::MATERIALS);

    DeviceLocalBuffer::from_data(
        memory_allocator,
        mutable_data,
        BufferUsage {
            uniform_buffer: true,
            ..BufferUsage::empty()
        },
        alloc_command_buffer_builder,
    )
    .unwrap()
}

pub(crate) fn get_bvh_buffer(
    memory_allocator: &GenericMemoryAllocator<Arc<FreeListAllocator>>,
    alloc_command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
) -> Arc<DeviceLocalBuffer<shaders::ty::GpuBVH>> {
    let mut bvh: CpuBVH = scene::BVH_OBJECTS[0].clone().into();

    for n in scene::BVH_OBJECTS[1..].iter() {
        bvh.merge_in_place(n.clone().into());
    }

    // DEBUG
    //bvh.graphify();

    DeviceLocalBuffer::<shaders::ty::GpuBVH>::from_data(
        memory_allocator,
        bvh.into(),
        BufferUsage {
            uniform_buffer: true,
            ..BufferUsage::empty()
        },
        alloc_command_buffer_builder,
    )
    .unwrap()
}

pub(crate) fn get_blue_noise_buffer(
    memory_allocator: &GenericMemoryAllocator<Arc<FreeListAllocator>>,
    alloc_command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
) -> Arc<dyn BufferAccess> {
    // TODO: make this path relative or something
    let blue_noise_data = UnitSphere
        .sample_iter(rand::thread_rng())
        .take(SAMPLES as usize)
        .collect::<Vec<[f32; 3]>>()
        .into_iter()
        .map(|x| [x[0], x[1], x[2], 0.0])
        .collect::<Vec<[f32; 4]>>();

    DeviceLocalBuffer::from_iter(
        memory_allocator,
        blue_noise_data,
        BufferUsage {
            uniform_buffer: true,
            ..BufferUsage::empty()
        },
        alloc_command_buffer_builder,
    )
    .unwrap()
}
