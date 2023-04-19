use std::sync::Arc;

use glam::Vec3;
use rand_distr::{Distribution, UnitSphere};
use vulkano::{
    buffer::{BufferAccess, BufferUsage, DeviceLocalBuffer},
    command_buffer::{
        allocator::CommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        DispatchIndirectCommand, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
    },
    device::Queue,
    sync::GpuFuture,
};

use crate::{
    allocators::Allocators,
    scene::{get_materials, get_objects},
    shaders::{self, LM_SAMPLES, LM_SIZE},
};

#[derive(Clone)]
pub(crate) struct Buffers {
    pub(crate) real_time: Arc<DeviceLocalBuffer<shaders::ty::RealTimeBuffer>>,
    pub(crate) mutable: Arc<DeviceLocalBuffer<shaders::ty::MutableData>>, // TODO: rename to MaterialBuffer
    pub(crate) bvh: Arc<DeviceLocalBuffer<shaders::ty::GpuBVH>>,
    pub(crate) lm_buffer: Arc<dyn BufferAccess>,
    pub(crate) lm_dispatch: Arc<DeviceLocalBuffer<[DispatchIndirectCommand]>>,
    pub(crate) noise: Arc<dyn BufferAccess>,
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
            real_time: get_real_time_buffer(allocators.clone(), &mut builder),
            mutable: get_mutable_buffer(allocators.clone(), &mut builder),
            bvh: get_bvh_buffer(allocators.clone(), &mut builder),
            lm_buffer: get_lm_buffer(allocators.clone(), &mut builder),
            lm_dispatch: get_lm_dispatch_buffer(allocators.clone(), &mut builder),
            noise: get_noise_buffer(allocators.clone(), &mut builder),
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

pub(crate) fn get_real_time_buffer<A>(
    allocators: Arc<Allocators>,
    alloc_command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer, A>,
) -> Arc<DeviceLocalBuffer<shaders::ty::RealTimeBuffer>>
where
    A: CommandBufferAllocator,
{
    DeviceLocalBuffer::from_data(
        &allocators.memory,
        shaders::ty::RealTimeBuffer::default(),
        BufferUsage {
            uniform_buffer: true,
            transfer_dst: true,
            ..BufferUsage::empty()
        },
        alloc_command_buffer_builder,
    )
    .unwrap()
}

pub(crate) fn get_mutable_buffer<A>(
    allocators: Arc<Allocators>,
    alloc_command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer, A>,
) -> Arc<DeviceLocalBuffer<shaders::ty::MutableData>>
where
    A: CommandBufferAllocator,
{
    let mut mutable_data = shaders::ty::MutableData {
        ..Default::default()
    };
    let materials = get_materials();
    mutable_data.mats[..materials.len()].copy_from_slice(&materials);

    DeviceLocalBuffer::from_data(
        &allocators.memory,
        mutable_data,
        BufferUsage {
            uniform_buffer: true,
            ..BufferUsage::empty()
        },
        alloc_command_buffer_builder,
    )
    .unwrap()
}

pub(crate) fn get_bvh_buffer<A>(
    allocators: Arc<Allocators>,
    alloc_command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer, A>,
) -> Arc<DeviceLocalBuffer<shaders::ty::GpuBVH>>
where
    A: CommandBufferAllocator,
{
    let bvh = get_objects();

    DeviceLocalBuffer::<shaders::ty::GpuBVH>::from_data(
        &allocators.memory,
        bvh,
        BufferUsage {
            uniform_buffer: true,
            ..BufferUsage::empty()
        },
        alloc_command_buffer_builder,
    )
    .unwrap()
}

pub(crate) fn get_noise_buffer<A>(
    allocators: Arc<Allocators>,
    alloc_command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer, A>,
) -> Arc<dyn BufferAccess>
where
    A: CommandBufferAllocator,
{
    let mut points = UnitSphere // TODO: sort so similar directions are grouped together
        .sample_iter(rand::thread_rng())
        .take(LM_SAMPLES as usize)
        .collect::<Vec<[f32; 3]>>()
        .into_iter()
        .map(Vec3::from_array)
        .collect::<Vec<Vec3>>();

    let mut sorted_points = vec![];

    while !points.is_empty() {
        let p = points.swap_remove(0);

        let mut sorted = points
            .into_iter()
            .map(|p2| (p.distance(p2), p2))
            .collect::<Vec<_>>();
        sorted.sort_by(|(dist1, _), (dist2, _)| dist1.total_cmp(dist2));
        points = sorted[3..].into_iter().map(|(_, p2)| *p2).collect();

        let new_points = sorted[0..3]
            .into_iter()
            .map(|(_, p2)| *p2)
            .collect::<Vec<Vec3>>();
        sorted_points.push([p, new_points[0], new_points[1], new_points[2]]);
    }

    let noise_data = sorted_points // TODO: store in file and use include_bytes! macro
        .into_iter()
        .map(|array| array.map(|v| v.extend(0.0).to_array()))
        .collect::<Vec<_>>();

    DeviceLocalBuffer::from_iter(
        &allocators.memory,
        noise_data,
        BufferUsage {
            uniform_buffer: true,
            ..BufferUsage::empty()
        },
        alloc_command_buffer_builder,
    )
    .unwrap()
}

pub(crate) fn get_lm_buffer<A>(
    allocators: Arc<Allocators>,
    alloc_command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer, A>,
) -> Arc<dyn BufferAccess>
where
    A: CommandBufferAllocator,
{
    let iter = (0..(LM_SIZE.pow(3))).map(|_| shaders::ty::Voxel::default());

    DeviceLocalBuffer::from_iter(
        &allocators.memory,
        iter,
        BufferUsage {
            storage_buffer: true,
            ..BufferUsage::empty()
        },
        alloc_command_buffer_builder,
    )
    .unwrap()
}

pub(crate) fn get_lm_dispatch_buffer<A>(
    allocators: Arc<Allocators>,
    alloc_command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer, A>,
) -> Arc<DeviceLocalBuffer<[DispatchIndirectCommand]>>
where
    A: CommandBufferAllocator,
{
    DeviceLocalBuffer::from_iter(
        &allocators.memory,
        [DispatchIndirectCommand { x: 0, y: 1, z: 1 }],
        BufferUsage {
            storage_buffer: true,
            indirect_buffer: true,
            ..BufferUsage::empty()
        },
        alloc_command_buffer_builder,
    )
    .unwrap()
}
