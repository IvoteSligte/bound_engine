use std::{mem::size_of, sync::Arc};

use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryAutoCommandBuffer,
        PrimaryCommandBufferAbstract,
    },
    device::Queue,
    memory::allocator::{AllocationCreateInfo, MemoryUsage},
    padded::Padded,
    sync::GpuFuture,
};

use crate::{
    allocator::Allocators,
    scene::{self},
    shaders,
};

#[derive(Clone)]
pub struct Buffers {
    pub real_time: Subbuffer<shaders::RealTimeBuffer>,
    pub vertex: Subbuffer<[scene::Vertex]>,
    pub vertex_idxs: Subbuffer<[u32]>,
    pub material_idxs: Subbuffer<[u32]>,
    pub material: Subbuffer<shaders::MaterialBuffer>,
    pub radiance: Subbuffer<[u8]>,
}

impl Buffers {
    pub fn new(allocators: Arc<Allocators>, queue: Arc<Queue>) -> Self {
        let mut builder = AutoCommandBufferBuilder::primary(
            &allocators.command_buffer,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let (vertex, vertex_idxs, material_idxs, material) =
            scene(allocators.clone(), &mut builder);

        let buffers = Self {
            real_time: real_time_buffer(allocators.clone()),
            vertex,
            vertex_idxs,
            material_idxs,
            material,
            radiance: zeroed(
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

fn stage_with_data<T: BufferContents>(
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

fn stage_with_iter<T, I>(
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

fn zeroed(
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

    stage_with_iter(allocators, cmb_builder, buffer.clone(), iter);

    buffer
}

fn real_time_buffer(allocators: Arc<Allocators>) -> Subbuffer<shaders::RealTimeBuffer> {
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
            projection_view: Default::default(),
            position: Default::default(),
        },
    )
    .unwrap()
}

fn scene(
    allocators: Arc<Allocators>,
    cmb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
) -> (
    Subbuffer<[scene::Vertex]>,
    Subbuffer<[u32]>,
    Subbuffer<[u32]>,
    Subbuffer<shaders::MaterialBuffer>,
) {
    let (vertex_data, vertex_idx_data, material_idx_data, material_data) = scene::load();
    let vertex_buffer = vertices(allocators.clone(), cmb_builder, vertex_data);
    let vertex_index_buffer = vertex_indices(allocators.clone(), cmb_builder, vertex_idx_data);
    let material_index_buffer =
        material_indices(allocators.clone(), cmb_builder, material_idx_data);
    let material_buffer = materials(allocators.clone(), cmb_builder, material_data);

    (
        vertex_buffer,
        vertex_index_buffer,
        material_index_buffer,
        material_buffer,
    )
}

fn vertices(
    allocators: Arc<Allocators>,
    cmb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    vertices: Vec<scene::Vertex>,
) -> Subbuffer<[scene::Vertex]> {
    let buffer = Buffer::new_slice(
        &allocators.memory,
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER
                | BufferUsage::VERTEX_BUFFER
                | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::DeviceOnly,
            ..Default::default()
        },
        vertices.len() as u64,
    )
    .unwrap();

    stage_with_iter(allocators, cmb_builder, buffer.clone(), vertices);

    buffer
}

fn vertex_indices(
    allocators: Arc<Allocators>,
    cmb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    vertex_indices: Vec<u32>,
) -> Subbuffer<[u32]> {
    let buffer = Buffer::new_slice(
        &allocators.memory,
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER
                | BufferUsage::INDEX_BUFFER
                | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::DeviceOnly,
            ..Default::default()
        },
        vertex_indices.len() as u64,
    )
    .unwrap();

    stage_with_iter(allocators, cmb_builder, buffer.clone(), vertex_indices);

    buffer
}

fn material_indices(
    allocators: Arc<Allocators>,
    cmb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    material_indices: Vec<u32>,
) -> Subbuffer<[u32]> {
    let buffer = Buffer::new_slice(
        &allocators.memory,
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::DeviceOnly,
            ..Default::default()
        },
        material_indices.len() as u64,
    )
    .unwrap();

    stage_with_iter(allocators, cmb_builder, buffer.clone(), material_indices);

    buffer
}

fn materials(
    allocators: Arc<Allocators>,
    cmb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    materials: Vec<shaders::Material>,
) -> Subbuffer<shaders::MaterialBuffer> {
    let material_data = shaders::MaterialBuffer {
        materials: materials
            .into_iter()
            .map(<Padded<shaders::Material, 4> as From<shaders::Material>>::from)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap(),
    };

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

    stage_with_data(allocators, cmb_builder, buffer.clone(), material_data);

    buffer
}
