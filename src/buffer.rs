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
    allocator::Allocators,
    scene::{self},
    shaders,
};

#[derive(Clone)]
pub struct Buffers {
    pub real_time: Subbuffer<shaders::RealTimeBuffer>,
    pub vertex: Subbuffer<[[f32; 4]]>,
    pub vertex_idxs: Subbuffer<[u32]>,
    pub grid: Vec<Subbuffer<[shaders::GridCell]>>,
    pub particles: Vec<Subbuffer<[shaders::ParticleUnit]>>,
}

impl Buffers {
    pub fn new(allocators: Arc<Allocators>, queue: Arc<Queue>) -> Self {
        let mut builder = AutoCommandBufferBuilder::primary(
            &allocators.command_buffer,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let (vertex, vertex_idxs, grid, particles) = scene(allocators.clone(), &mut builder);

        let buffers = Self {
            real_time: real_time_buffer(allocators.clone()),
            vertex,
            vertex_idxs,
            grid,
            particles,
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
    Subbuffer<[[f32; 4]]>,
    Subbuffer<[u32]>,
    Vec<Subbuffer<[shaders::GridCell]>>,
    Vec<Subbuffer<[shaders::ParticleUnit]>>,
) {
    let (vertexes, vertex_indexes, static_particles) = scene::load();
    let vertex_buffer = vertices(allocators.clone(), cmb_builder, vertexes);
    let vertex_index_buffer = vertex_indices(allocators.clone(), cmb_builder, vertex_indexes);
    let grid_buffer = grid(allocators.clone(), cmb_builder);
    let particle_buffers = particles(allocators.clone(), cmb_builder, static_particles);

    (
        vertex_buffer,
        vertex_index_buffer,
        grid_buffer,
        particle_buffers,
    )
}

fn vertices(
    allocators: Arc<Allocators>,
    cmb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    vertices: Vec<[f32; 4]>,
) -> Subbuffer<[[f32; 4]]> {
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

fn grid(
    allocators: Arc<Allocators>,
    cmb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
) -> Vec<Subbuffer<[shaders::GridCell]>> {
    let mut buffers = vec![];

    for i in 0..3 {
        let buffer = zeroed(
            allocators,
            cmb_builder,
            size_of::<shaders::GridCell>() as u64 * (shaders::CELLS as u64).pow(3),
            BufferUsage::STORAGE_BUFFER,
        );
        buffers.push(buffer);
    }
}

fn particles(
    allocators: Arc<Allocators>,
    cmb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    static_particles: [Vec<shaders::StaticParticle>; 3],
) -> Vec<Subbuffer<[shaders::ParticleUnit]>> {
    let mut buffers = vec![];

    for static_particles in static_particles {
        let particles = (0..shaders::DYN_PARTICLES)
            .map(|_| shaders::DynamicParticle::default().cast::<shaders::ParticleUnit>())
            .collect::<Vec<_>>();
        particles.extend(static_particles.map(|particle| particle.cast::<shaders::ParticleUnit>()));

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
            shaders::DYN_PARTICLES as u64 + static_particles.len() as u64,
        )
        .unwrap();

        stage_with_iter(allocators, cmb_builder, buffer.clone(), particles); // FIXME: combine dynamic and static particles

        buffers.push(buffer);
    }
    buffers
}
