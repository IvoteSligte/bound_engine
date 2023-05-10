use std::{mem::size_of, ops::Range, sync::Arc};

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
    ray_directions,
    scene::{get_materials, get_objects, RawObject},
    shaders::{self, LM_SAMPLES, LM_SIZE, LM_MAX_POINTS},
};

#[derive(Clone)]
pub(crate) struct Buffers {
    pub(crate) real_time: Subbuffer<shaders::RealTimeBuffer>,
    pub(crate) mutable: Subbuffer<shaders::MutableData>, // TODO: rename to MaterialBuffer
    pub(crate) objects: Subbuffer<[RawObject]>,
    pub(crate) lm_buffers: LmBuffers,
    pub(crate) noise: Subbuffer<shaders::NoiseBuffer>,
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
            mutable: get_mutable_buffer(allocators.clone(), &mut builder),
            objects: get_object_buffer(allocators.clone(), &mut builder),
            lm_buffers: LmBuffers::new(allocators.clone(), &mut builder),
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

pub(crate) fn get_real_time_buffer(
    allocators: Arc<Allocators>,
) -> Subbuffer<shaders::RealTimeBuffer> {
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
            previousRotation: Default::default(),
            position: Default::default(),
            previousPosition: Default::default(),
            lightmapOrigin: Default::default(),
            lightmapBufferOffset: Default::default(),
            deltaLightmapOrigins: Default::default(),
        },
    )
    .unwrap()
}

pub(crate) fn get_mutable_buffer(
    allocators: Arc<Allocators>,
    cmb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
) -> Subbuffer<shaders::MutableData> {
    let mut mutable_data = shaders::MutableData {
        mats: [shaders::Material {
            emittance: Default::default(),
            reflectance: Default::default(),
        }
        .into(); 32], // TODO: dynamic size
    };
    let materials = get_materials()
        .into_iter()
        .map(|x| x.into())
        .collect::<Vec<_>>();
    mutable_data.mats[..materials.len()].copy_from_slice(&materials);

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

    stage_buffer_with_data(allocators, cmb_builder, buffer.clone(), mutable_data);

    buffer
}

pub(crate) fn get_object_buffer(
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

pub(crate) fn get_noise_buffer(
    allocators: Arc<Allocators>,
    cmb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
) -> Subbuffer<shaders::NoiseBuffer> {
    debug_assert!(ray_directions::VECTORS.len() == LM_SAMPLES as usize);

    let extend_to_vec4 = |iterable: &[[f32; 3]]| {
        iterable
            .into_iter()
            .map(|v| [v[0], v[1], v[2], 0.0])
            .collect::<Vec<_>>()
    };

    let noise_data = shaders::NoiseBuffer {
        dirs: extend_to_vec4(&ray_directions::VECTORS).try_into().unwrap(),
        midDirs: extend_to_vec4(&ray_directions::MID_VECTORS)
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

    stage_buffer_with_data(allocators, cmb_builder, buffer.clone(), noise_data);

    buffer
}

#[derive(Clone)]
pub(crate) struct LmBuffers {
    pub(crate) march: Subbuffer<[u8]>,
    pub(crate) gpu: Subbuffer<[shaders::LMPoint]>,
    pub(crate) counter: Subbuffer<u32>, // TODO: merge with `gpu` buffer
    pub(crate) range_left: Range<u32>,
}

impl LmBuffers {
    pub(crate) fn read_to_range(&mut self) {
        let count = self.counter.read().unwrap().min(LM_MAX_POINTS);
        self.range_left = 0..count;
    }

    pub(crate) fn new(
        allocators: Arc<Allocators>,
        cmb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) -> LmBuffers {
        let gpu_buffer = Self::create_lm_buffer(allocators.clone(), cmb_builder);

        let march_buffer = Self::create_lm_march_buffer(allocators.clone(), cmb_builder);

        LmBuffers {
            march: march_buffer,
            gpu: gpu_buffer,
            counter: Self::create_lm_counter_buffer(allocators),
            range_left: 0..0,
        }
    }

    fn create_lm_buffer(
        allocators: Arc<Allocators>,
        cmb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) -> Subbuffer<[shaders::LMPoint]> {
        let iter = (0..(LM_SIZE.pow(3)))
            .map(|_| shaders::LMPoint {
                lmIndex: Default::default(),
                material: Default::default(),
                position: Default::default(),
                normal: Default::default(),
            })
            .collect::<Vec<_>>();

        let buffer = Buffer::new_slice(
            &allocators.memory,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::DeviceOnly,
                ..Default::default()
            },
            iter.len() as u64,
        )
        .unwrap();

        stage_buffer_with_iter(
            allocators.clone(),
            cmb_builder,
            buffer.clone(),
            iter.clone(),
        );

        buffer
    }

    fn create_lm_march_buffer(
        allocators: Arc<Allocators>,
        cmb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) -> Subbuffer<[u8]> {
        let len = size_of::<shaders::LMMarchBuffer>();
        let iter = (0..len).map(|_| 0u8);

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
            len as u64,
        )
        .unwrap();

        stage_buffer_with_iter(allocators.clone(), cmb_builder, buffer.clone(), iter);

        buffer
    }

    pub(crate) fn create_lm_counter_buffer(allocators: Arc<Allocators>) -> Subbuffer<u32> {
        let data = 0u32;

        Buffer::from_data(
            &allocators.memory,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Download,
                ..Default::default()
            },
            data,
        )
        .unwrap()
    }
}
