use std::sync::Arc;

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
    pub(crate) blue_noise: Arc<dyn BufferAccess>,
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
            blue_noise: get_blue_noise_buffer(allocators.clone(), &mut builder),
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

pub(crate) fn get_blue_noise_buffer<A>(
    allocators: Arc<Allocators>,
    alloc_command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer, A>,
) -> Arc<dyn BufferAccess>
where
    A: CommandBufferAllocator,
{
    let blue_noise_data = UnitSphere
        .sample_iter(rand::thread_rng())
        .take(LM_SAMPLES as usize)
        .collect::<Vec<[f32; 3]>>()
        .into_iter()
        .map(|x| [x[0], x[1], x[2], 0.0])
        .collect::<Vec<[f32; 4]>>();

    DeviceLocalBuffer::from_iter(
        &allocators.memory,
        blue_noise_data,
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
    let iter = (0..(LM_SIZE.pow(3))).map(|_| 0u32).collect::<Vec<u32>>();

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
