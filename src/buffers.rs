use std::sync::Arc;

use rand_distr::{Distribution, UnitSphere};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{AutoCommandBufferBuilder, CopyBufferInfo, PrimaryAutoCommandBuffer},
    memory::allocator::{
        AllocationCreateInfo, FreeListAllocator, GenericMemoryAllocator, MemoryUsage,
    },
};

use crate::{
    scene::{get_materials, get_objects},
    shaders::{self, SAMPLES},
};

fn subbuffer_from_data<T: BufferContents>(
    memory_allocator: &GenericMemoryAllocator<Arc<FreeListAllocator>>,
    command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    usage: BufferUsage,
    data: T,
) -> Subbuffer<T> {
    let staging = Buffer::from_data(
        memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
        data,
    )
    .unwrap();

    let buffer = Buffer::new_sized::<T>(
        memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST | usage,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::DeviceOnly,
            ..Default::default()
        },
    )
    .unwrap();

    command_buffer_builder
        .copy_buffer(CopyBufferInfo::buffers(staging, buffer.clone()))
        .unwrap();

    buffer
}

// TODO: rename to material_buffer
pub(crate) fn get_mutable_buffer(
    memory_allocator: &GenericMemoryAllocator<Arc<FreeListAllocator>>,
    command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
) -> Subbuffer<shaders::MutableData> {
    let mut mutable_data = shaders::MutableData {
        mats: [shaders::Material {
            reflectance: [0.0; 3].into(),
            emittance: [0.0; 3],
        }
        .into(); 256],
    };
    let materials = get_materials()
        .into_iter()
        .map(|x| x.into())
        .collect::<Vec<_>>();
    mutable_data.mats[..materials.len()].copy_from_slice(&materials);

    subbuffer_from_data(
        memory_allocator,
        command_buffer_builder,
        BufferUsage::UNIFORM_BUFFER,
        mutable_data,
    )
}

pub(crate) fn get_bvh_buffer(
    memory_allocator: &GenericMemoryAllocator<Arc<FreeListAllocator>>,
    command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
) -> Subbuffer<shaders::GpuBVH> {
    let bvh = get_objects();

    subbuffer_from_data(
        memory_allocator,
        command_buffer_builder,
        BufferUsage::UNIFORM_BUFFER,
        bvh,
    )
}

pub(crate) fn get_blue_noise_buffer(
    memory_allocator: &GenericMemoryAllocator<Arc<FreeListAllocator>>,
    command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
) -> Subbuffer<shaders::BlueNoise> {
    let mut blue_noise_data = shaders::BlueNoise {
        items: [[0.0; 4]; 1024],
    };

    let blue_noise = UnitSphere
        .sample_iter(rand::thread_rng())
        .take(SAMPLES as usize)
        .collect::<Vec<[f32; 3]>>()
        .into_iter()
        .map(|x| [x[0], x[1], x[2], 0.0])
        .collect::<Vec<[f32; 4]>>();

    blue_noise_data.items.copy_from_slice(&blue_noise);

    subbuffer_from_data(
        memory_allocator,
        command_buffer_builder,
        BufferUsage::UNIFORM_BUFFER,
        blue_noise_data,
    )
}
