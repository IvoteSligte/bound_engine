use std::sync::Arc;

use vulkano::{
    device::Queue,
    format::Format,
    image::{
        view::{ImageView, ImageViewCreateInfo},
        ImageCreateFlags, ImageDimensions, ImageUsage, ImageViewAbstract, StorageImage,
        SwapchainImage,
    },
    sampler::Sampler,
};
use winit::window::Window;

use crate::{
    allocators::Allocators,
    shaders::{LM_COUNT, LM_RAYS, LM_SIZE},
};

#[derive(Clone)]
pub(crate) struct Images {
    pub(crate) color: Arc<StorageImage>,
    pub(crate) lightmap: LightmapImages,
    pub(crate) swapchain: Vec<Arc<SwapchainImage>>,
    pub(crate) sampler: Arc<Sampler>,
}

impl Images {
    pub(crate) fn new(
        allocators: Arc<Allocators>,
        window: Arc<Window>,
        swapchain_images: Vec<Arc<SwapchainImage>>,
        sampler: Arc<Sampler>,
    ) -> Self {
        Self {
            color: create_color_image(allocators.clone(), window),
            lightmap: LightmapImages::new(allocators.clone()),
            swapchain: swapchain_images,
            sampler,
        }
    }

    pub(crate) fn image_views(&self) -> ImageViews {
        ImageViews {
            color: ImageView::new_default(self.color.clone()).unwrap(),
            lightmap: self.lightmap.image_views(),
        }
    }

    pub(crate) fn sampler(&self) -> Arc<Sampler> {
        self.sampler.clone()
    }
}

#[derive(Clone)]
pub(crate) struct ImageViews {
    pub(crate) color: Arc<ImageView<StorageImage>>,
    pub(crate) lightmap: LightmapImageViews,
}

pub(crate) fn create_color_image(
    allocators: Arc<Allocators>,
    window: Arc<Window>,
) -> Arc<StorageImage> {
    StorageImage::with_usage(
        &allocators.memory,
        ImageDimensions::Dim2d {
            width: window.inner_size().width,
            height: window.inner_size().height,
            array_layers: 1,
        },
        Format::R16G16B16A16_UNORM, // double precision for copying to srgb
        ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
        ImageCreateFlags::empty(),
        [], // INFO: empty queue_family_indices sets SharingMode to Exclusive in vulkano 0.33.0 with no other side effects
    )
    .unwrap()
}

#[derive(Clone)]
pub(crate) struct LightmapImages {
    pub(crate) colors: Vec<Vec<Arc<StorageImage>>>,
    pub(crate) staging_color: Arc<StorageImage>,
    pub(crate) sdfs: Vec<Arc<StorageImage>>,
    pub(crate) materials: Vec<Arc<StorageImage>>,
}

#[derive(Clone)]
pub(crate) struct LightmapImageViews {
    pub(crate) colors_storage: Vec<Vec<Arc<dyn ImageViewAbstract>>>,
    pub(crate) colors_final_sampled: Vec<Arc<dyn ImageViewAbstract>>,
    pub(crate) sdfs_storage: Vec<Arc<dyn ImageViewAbstract>>,
    pub(crate) sdfs_sampled: Vec<Arc<dyn ImageViewAbstract>>,
    pub(crate) materials_storage: Vec<Arc<dyn ImageViewAbstract>>,
}

impl LightmapImages {
    pub(crate) fn new(allocators: Arc<Allocators>) -> Self {
        let dimensions = ImageDimensions::Dim3d {
            width: LM_SIZE,
            height: LM_SIZE,
            depth: LM_SIZE,
        };

        // TODO: layout optimisation for storage
        let create_storage_image = |usage, format| {
            StorageImage::with_usage(
                &allocators.memory,
                dimensions,
                format,
                ImageUsage::STORAGE | usage,
                ImageCreateFlags::empty(),
                [], // INFO: empty queue_family_indices sets SharingMode to Exclusive in vulkano 0.33.0 with no other side effects
            )
            .unwrap()
        };

        let mut colors = (0..(LM_RAYS - 1))
            .map(|_| {
                (0..LM_COUNT)
                    .map(|_| {
                        create_storage_image(
                            ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST,
                            Format::R16G16B16A16_UNORM,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        colors.push(
            (0..LM_COUNT)
                .map(|_| {
                    create_storage_image(
                        ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                        Format::R16G16B16A16_UNORM,
                    )
                })
                .collect::<Vec<_>>(),
        );

        let staging_color = create_storage_image(
            ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST,
            Format::R16G16B16A16_UNORM,
        );

        // FIXME: sdf staging + moving
        let sdfs = (0..LM_COUNT)
            .map(|_| {
                create_storage_image(
                    ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                    Format::R16_SFLOAT,
                )
            })
            .collect::<Vec<_>>();

        let materials = (0..LM_COUNT)
            .map(|_| {
                create_storage_image(
                    ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST,
                    Format::R16_UINT,
                )
            })
            .collect::<Vec<_>>();

        Self {
            colors,
            staging_color,
            sdfs,
            materials,
        }
    }

    pub(crate) fn image_views(&self) -> LightmapImageViews {
        let views = |imgs: &Vec<Arc<StorageImage>>, usage: ImageUsage| {
            imgs.iter()
                .map(|img| {
                    ImageView::new(
                        img.clone(),
                        ImageViewCreateInfo {
                            usage,
                            ..ImageViewCreateInfo::from_image(img)
                        },
                    )
                    .unwrap() as Arc<dyn ImageViewAbstract>
                })
                .collect()
        };

        let colors_storage = self.colors.iter().map(|imgs| views(imgs, ImageUsage::STORAGE)).collect();
        let sdfs_storage = views(&self.sdfs, ImageUsage::STORAGE);
        let materials_storage = views(&self.materials, ImageUsage::STORAGE);

        let colors_final_sampled = views(self.colors.last().unwrap(), ImageUsage::SAMPLED);
        let sdfs_sampled = views(&self.sdfs, ImageUsage::SAMPLED);

        LightmapImageViews {
            colors_storage,
            colors_final_sampled,
            sdfs_storage,
            sdfs_sampled,
            materials_storage,
        }
    }
}
