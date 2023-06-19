use std::{mem::size_of, sync::Arc};

use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryAutoCommandBuffer,
        PrimaryCommandBufferAbstract,
    },
    device::Queue,
    memory::allocator::{AllocationCreateInfo, MemoryUsage},
    sync::GpuFuture,
};

use crate::{
    allocators::Allocators,
    scene::{get_materials, get_objects, RawObject},
    shaders,
};

#[derive(Clone)]
pub(crate) struct Buffers {
    pub(crate) real_time: Subbuffer<shaders::RealTimeBuffer>,
    pub(crate) materials: Subbuffer<shaders::MaterialBuffer>, // TODO: rename to MaterialBuffer
    pub(crate) objects: Subbuffer<[RawObject]>,
    pub(crate) radiance: Subbuffer<[u8]>,
}

impl Buffers {
    pub(crate) fn new(allocators: Arc<Allocators>, queue: Arc<Queue>) -> Self {
        let mut builder = AutoCommandBufferBuilder::primary(
            &allocators.command_buffer,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let buffers = Self {
            real_time: get_real_time_buffer(allocators.clone()),
            materials: get_material_buffer(allocators.clone(), &mut builder),
            objects: get_object_buffer(allocators.clone(), &mut builder),
            radiance: create_zeroed_buffer(
                allocators.clone(),
                &mut builder,
                size_of::<shaders::RadianceBuffer>() as u64,
                BufferUsage::STORAGE_BUFFER,
            ),
        };

        builder
            .build()
            .unwrap()
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        buffers
    }
}

// TODO: move these functions to impl Buffers
fn stage_buffer_with_data<T: BufferContents>(
    allocators: Arc<Allocators>,
    cmb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    buffer: Subbuffer<T>,
    data: T,
) {
    let staging = Buffer::from_data(
        &allocators.memory,
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

    cmb_builder
        .copy_buffer(CopyBufferInfo::buffers(staging, buffer))
        .unwrap();
}

fn stage_buffer_with_iter<T, I>(
    allocators: Arc<Allocators>,
    cmb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    buffer: Subbuffer<[T]>,
    iter: I,
) where
    T: BufferContents,
    I: IntoIterator<Item = T>,
    I::IntoIter: ExactSizeIterator,
{
    let staging = Buffer::from_iter(
        &allocators.memory,
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
        iter,
    )
    .unwrap();

    cmb_builder
        .copy_buffer(CopyBufferInfo::buffers(staging, buffer))
        .unwrap();
}

fn create_zeroed_buffer(
    allocators: Arc<Allocators>,
    cmb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    byte_size: u64,
    usage: BufferUsage,
) -> Subbuffer<[u8]> {
    let iter = (0..byte_size).map(|_| 0u8).collect::<Vec<u8>>();

    let buffer = Buffer::new_slice(
        &allocators.memory,
        BufferCreateInfo {
            usage: usage | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::DeviceOnly,
            ..Default::default()
        },
        byte_size,
    )
    .unwrap();

    stage_buffer_with_iter(allocators, cmb_builder, buffer.clone(), iter);

    buffer
}

fn get_real_time_buffer(allocators: Arc<Allocators>) -> Subbuffer<shaders::RealTimeBuffer> {
    Buffer::from_data(
        &allocators.memory,
        BufferCreateInfo {
            usage: BufferUsage::UNIFORM_BUFFER,
            ..BufferCreateInfo::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..AllocationCreateInfo::default()
        },
        shaders::RealTimeBuffer {
            rotation: Default::default(),
            position: Default::default(),
            lightmapOrigin: Default::default(),
            deltaLightmapOrigins: Default::default(),
        },
    )
    .unwrap()
}

fn get_material_buffer(
    allocators: Arc<Allocators>,
    cmb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
) -> Subbuffer<shaders::MaterialBuffer> {
    let mut material_data = shaders::MaterialBuffer {
        materials: [shaders::Material {
            emittance: Default::default(),
            reflectance: Default::default(),
        }
        .into(); 32], // TODO: dynamic size
    };
    let materials = get_materials()
        .into_iter()
        .map(|x| x.into())
        .collect::<Vec<_>>();
    material_data.materials[..materials.len()].copy_from_slice(&materials);

    let buffer = Buffer::new_sized(
        &allocators.memory,
        BufferCreateInfo {
            usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::DeviceOnly,
            ..Default::default()
        },
    )
    .unwrap();

    stage_buffer_with_data(allocators, cmb_builder, buffer.clone(), material_data);

    buffer
}

fn get_object_buffer(
    allocators: Arc<Allocators>,
    cmb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
) -> Subbuffer<[RawObject]> {
    let object_data = get_objects();

    let buffer = Buffer::new_slice(
        &allocators.memory,
        BufferCreateInfo {
            usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::DeviceOnly,
            ..Default::default()
        },
        object_data.len() as u64,
    )
    .unwrap();

    stage_buffer_with_iter(allocators, cmb_builder, buffer.clone(), object_data);

    buffer
}
