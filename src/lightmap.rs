use itertools::Itertools;
use vec_cycle::VecCycle;
use vulkano::buffer::BufferUsage;

use crate::shaders::LIGHTMAP_COUNT;
use crate::shaders::LIGHTMAP_SIZE;
use crate::shaders::RAYS_INDIRECT;

use super::shaders;

use vulkano::buffer::DeviceLocalBuffer;

use std::ops::DerefMut;

use std::ops::Deref;

use vulkano::memory::allocator::FreeListAllocator;

use vulkano::memory::allocator::GenericMemoryAllocator;

use vulkano::image::view::ImageView;

use vulkano::format::Format;

use vulkano::image::ImageCreateFlags;

use vulkano::image::ImageUsage;

use vulkano::image::ImageDimensions;

use vulkano::memory::allocator::MemoryAllocator;

use vulkano::image::ImageViewAbstract;

use vulkano::image::StorageImage;

use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct LightmapImages {
    pub(crate) colors: Vec<Arc<StorageImage>>,
    pub(crate) syncs: Vec<Arc<StorageImage>>,
    pub(crate) staging: Arc<StorageImage>,
}

#[derive(Clone)]
pub(crate) struct LightmapImageViews {
    pub(crate) colors: Vec<Arc<dyn ImageViewAbstract>>,
    pub(crate) syncs: Vec<Arc<dyn ImageViewAbstract>>,
    pub(crate) staging: Arc<dyn ImageViewAbstract>,
}

impl LightmapImages {
    pub(crate) fn new(
        allocator: &(impl MemoryAllocator + ?Sized),
        queue_family_index: u32,
    ) -> Self {
        let dimensions = ImageDimensions::Dim3d {
            width: LIGHTMAP_SIZE,
            height: LIGHTMAP_SIZE,
            depth: LIGHTMAP_SIZE,
        };

        let create_storage_image = |usage, format| {
            StorageImage::with_usage(
                allocator,
                dimensions,
                format,
                ImageUsage {
                    storage: true,
                    ..usage
                },
                ImageCreateFlags::empty(),
                [queue_family_index],
            )
            .unwrap()
        };

        let colors = (0..(LIGHTMAP_COUNT * RAYS_INDIRECT))
            .map(|_| {
                create_storage_image(
                    ImageUsage {
                        transfer_dst: true,
                        ..ImageUsage::default()
                    },
                    Format::R16G16B16A16_UNORM,
                )
            })
            .collect();

        let staging = create_storage_image(
            ImageUsage {
                transfer_src: true,
                ..ImageUsage::default()
            },
            Format::R16G16B16A16_UNORM,
        );

        let syncs = (0..LIGHTMAP_COUNT)
            .map(|_| create_storage_image(ImageUsage::default(), Format::R32_UINT))
            .collect();

        Self {
            colors,
            staging,
            syncs,
        }
    }

    pub(crate) fn image_views(&self) -> LightmapImageViews {
        LightmapImageViews {
            colors: self
                .colors
                .iter()
                .map(|vlm| {
                    ImageView::new_default(vlm.clone()).unwrap() as Arc<dyn ImageViewAbstract>
                })
                .collect(),
            syncs: self
                .syncs
                .iter()
                .map(|vlm| {
                    ImageView::new_default(vlm.clone()).unwrap() as Arc<dyn ImageViewAbstract>
                })
                .collect(),
            staging: ImageView::new_default(self.staging.clone()).unwrap(),
        }
    }
}

#[derive(Clone)]
pub(crate) struct LightmapBufferSet(VecCycle<Vec<LightmapBufferUnit>>);

impl LightmapBufferSet {
    pub(crate) fn new(
        memory_allocator: &GenericMemoryAllocator<Arc<FreeListAllocator>>,
        queue_family_index: u32,
        count: usize,
    ) -> Self {
        assert_ne!(count, 0, "count must be greater than zero");

        Self(VecCycle::new(
            (0..count)
                .map(|_| LightmapBufferUnit::new(memory_allocator, queue_family_index))
                .permutations(count)
                .collect(),
        ))
    }
}

impl Deref for LightmapBufferSet {
    type Target = VecCycle<Vec<LightmapBufferUnit>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for LightmapBufferSet {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Clone)]
pub(crate) struct LightmapBufferUnit {
    pub(crate) buffer: Arc<DeviceLocalBuffer<shaders::ty::CurrBuffer>>,
    pub(crate) counters: Arc<DeviceLocalBuffer<shaders::ty::CurrCounters>>,
}

impl LightmapBufferUnit {
    pub(crate) fn new(
        memory_allocator: &GenericMemoryAllocator<Arc<FreeListAllocator>>,
        queue_family_index: u32,
    ) -> Self {
        Self {
            buffer: DeviceLocalBuffer::new(
                memory_allocator,
                BufferUsage {
                    storage_buffer: true,
                    uniform_buffer: true,
                    ..BufferUsage::empty()
                },
                [queue_family_index],
            )
            .unwrap(),
            counters: DeviceLocalBuffer::new(
                memory_allocator,
                BufferUsage {
                    storage_buffer: true,
                    uniform_buffer: true,
                    transfer_dst: true,
                    ..BufferUsage::empty()
                },
                [queue_family_index],
            )
            .unwrap(),
        }
    }
}
