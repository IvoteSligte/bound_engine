use std::sync::Arc;

use vulkano::{
    device::Queue,
    format::Format,
    image::{
        view::ImageView, ImageCreateFlags, ImageDimensions, ImageUsage, ImageViewAbstract,
        StorageImage, SwapchainImage,
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
        queue: Arc<Queue>,
        swapchain_images: Vec<Arc<SwapchainImage>>,
        sampler: Arc<Sampler>,
    ) -> Self {
        Self {
            color: create_color_image(allocators.clone(), window, queue.clone()),
            lightmap: LightmapImages::new(allocators.clone(), queue.clone()),
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
    queue: Arc<Queue>,
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
        [queue.queue_family_index()],
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
    pub(crate) colors: Vec<Vec<Arc<dyn ImageViewAbstract>>>,
    pub(crate) sdfs: Vec<Arc<dyn ImageViewAbstract>>,
    pub(crate) materials: Vec<Arc<dyn ImageViewAbstract>>,
}

impl LightmapImages {
    pub(crate) fn new(allocators: Arc<Allocators>, queue: Arc<Queue>) -> Self {
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
                [queue.queue_family_index()],
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
        let view = |vlm: &Arc<StorageImage>| {
            ImageView::new_default(vlm.clone()).unwrap() as Arc<dyn ImageViewAbstract>
        };

        LightmapImageViews {
            colors: self
                .colors
                .iter()
                .map(|vec| vec.iter().map(view).collect())
                .collect(),
            sdfs: self.sdfs.iter().map(view).collect(),
            materials: self.materials.iter().map(view).collect(),
        }
    }
}
