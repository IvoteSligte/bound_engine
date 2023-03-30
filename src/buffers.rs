use std::sync::Arc;

use vulkano::{pipeline::graphics::viewport::Viewport, memory::allocator::{GenericMemoryAllocator, FreeListAllocator}, command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer}, buffer::{DeviceLocalBuffer, BufferUsage}};

use crate::{FOV, shaders, scene, bvh::CpuBVH};

pub(crate) fn get_constant_buffer(
    viewport: &Viewport,
    memory_allocator: &GenericMemoryAllocator<Arc<FreeListAllocator>>,
    alloc_command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
) -> Arc<DeviceLocalBuffer<shaders::ty::ConstantBuffer>> {
    let constant_data = shaders::ty::ConstantBuffer {
        ratio: [FOV, -FOV * viewport.dimensions[1] / viewport.dimensions[0]],
    };

    DeviceLocalBuffer::from_data(
        memory_allocator,
        constant_data,
        BufferUsage {
            uniform_buffer: true,
            ..BufferUsage::empty()
        },
        alloc_command_buffer_builder,
    )
    .unwrap()
}

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