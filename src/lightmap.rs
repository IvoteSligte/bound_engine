use crate::shaders::LIGHTMAP_COUNT;
use crate::shaders::LIGHTMAP_SIZE;
use crate::shaders::RAYS_INDIRECT;

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
    pub(crate) colors: Vec<Vec<Arc<StorageImage>>>,
    pub(crate) syncs: Vec<Arc<StorageImage>>,
    pub(crate) staging_color: Arc<StorageImage>,
    pub(crate) staging_sync: Arc<StorageImage>,
}

#[derive(Clone)]
pub(crate) struct LightmapImageViews {
    pub(crate) colors: Vec<Vec<Arc<dyn ImageViewAbstract>>>,
    pub(crate) syncs: Vec<Arc<dyn ImageViewAbstract>>,
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

        let create_storage_transfer_image = |format| {
            StorageImage::with_usage(
                allocator,
                dimensions,
                format,
                ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST,
                ImageCreateFlags::empty(),
                [queue_family_index],
            )
            .unwrap()
        };

        let colors = (0..RAYS_INDIRECT)
            .map(|_| {
                (0..(LIGHTMAP_COUNT))
                    .map(|_| create_storage_transfer_image(Format::R16G16B16A16_UNORM))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let syncs = (0..LIGHTMAP_COUNT)
            .map(|_| create_storage_transfer_image(Format::R32G32_UINT))
            .collect();

        let staging_color = create_storage_transfer_image(Format::R16G16B16A16_UNORM);

        let staging_sync = create_storage_transfer_image(Format::R32G32_UINT);

        Self {
            colors,
            syncs,
            staging_color,
            staging_sync,
        }
    }

    pub(crate) fn image_views(&self) -> LightmapImageViews {
        LightmapImageViews {
            colors: self
                .colors
                .iter()
                .map(|vec| {
                    vec.iter()
                        .map(|vlm| {
                            ImageView::new_default(vlm.clone()).unwrap()
                                as Arc<dyn ImageViewAbstract>
                        })
                        .collect()
                })
                .collect(),
            syncs: self
                .syncs
                .iter()
                .map(|vlm| {
                    ImageView::new_default(vlm.clone()).unwrap() as Arc<dyn ImageViewAbstract>
                })
                .collect(),
        }
    }
}
