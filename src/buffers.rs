use std::{sync::Arc, f32::consts::PI};

use glam::{Vec3, Quat};
use vulkano::{memory::allocator::{GenericMemoryAllocator, FreeListAllocator}, command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer}, buffer::{DeviceLocalBuffer, BufferUsage, BufferAccess}};

use crate::{shaders::{self, SAMPLES}, scene, bvh::CpuBVH};

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
    let blue_noise_data = image::open("/home/ivo/Code/bound_engine/images/blue_noise_rgba.png")
        .unwrap()
        .to_rgba32f()
        .chunks_exact(2)
        .into_iter()
        .take(SAMPLES as usize)
        .map(|chunk| {
            let r1 = chunk[0] * 2.0 * PI;
            let r2 = chunk[1] * 0.5 * PI;

            let rand = Vec3::new(r1.cos() * r2.cos(), r1.sin() * r2.cos(), r2.sin());

            Quat::from_rotation_arc(Vec3::new(0.0, 0.0, 1.0), rand).to_array()
        })
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