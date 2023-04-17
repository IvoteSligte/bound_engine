use std::sync::Arc;

use vulkano::{
    device::Queue,
    format::Format,
    image::{
        view::ImageView, ImageCreateFlags, ImageDimensions, ImageUsage, ImageViewAbstract,
        StorageImage, SwapchainImage,
    },
};
use winit::window::Window;

use crate::{
    allocators::Allocators,
    shaders::{LM_COUNT, LM_RAYS, LM_SIZE},
};

#[derive(Clone)]
pub(crate) struct Images {
    pub(crate) color: Arc<StorageImage>, // TODO: eliminate `StorageImage`s
    pub(crate) lightmap: LightmapImages,
    pub(crate) swapchain: Vec<Arc<SwapchainImage>>,
}

impl Images {
    pub(crate) fn new(
        allocators: Arc<Allocators>,
        window: Arc<Window>,
        queue: Arc<Queue>,
        swapchain_images: Vec<Arc<SwapchainImage>>,
    ) -> Self {
        Self {
            color: create_color_image(allocators.clone(), window, queue.clone()),
            lightmap: LightmapImages::new(allocators.clone(), queue.clone()),
            swapchain: swapchain_images,
        }
    }

    pub(crate) fn image_views(&self) -> ImageViews {
        ImageViews {
            color: ImageView::new_default(self.color.clone()).unwrap(),
            lightmap: self.lightmap.image_views(),
        }
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
        ImageUsage {
            storage: true,
            transfer_src: true,
            ..ImageUsage::empty()
        },
        ImageCreateFlags::empty(),
        [queue.queue_family_index()],
    )
    .unwrap()
}

#[derive(Clone)]
pub(crate) struct LightmapImages {
    pub(crate) colors: Vec<Vec<Arc<StorageImage>>>,
    pub(crate) final_color: Vec<Arc<StorageImage>>,
    pub(crate) staging_color: Arc<StorageImage>,
}

#[derive(Clone)]
pub(crate) struct LightmapImageViews {
    pub(crate) colors: Vec<Vec<Arc<dyn ImageViewAbstract>>>,
    pub(crate) final_color: Vec<Arc<dyn ImageViewAbstract>>,
}

impl LightmapImages {
    pub(crate) fn new(allocators: Arc<Allocators>, queue: Arc<Queue>) -> Self {
        let dimensions = ImageDimensions::Dim3d {
            width: LM_SIZE,
            height: LM_SIZE,
            depth: LM_SIZE,
        };

        let create_storage_image = |usage, format| {
            StorageImage::with_usage(
                &allocators.memory,
                dimensions,
                format,
                ImageUsage {
                    storage: true,
                    ..usage
                },
                ImageCreateFlags::empty(),
                [queue.queue_family_index()],
            )
            .unwrap()
        };

        let colors = (0..LM_RAYS)
            .map(|_| {
                (0..(LM_COUNT))
                    .map(|_| {
                        create_storage_image(
                            ImageUsage {
                                transfer_src: true,
                                transfer_dst: true,
                                ..ImageUsage::default()
                            },
                            Format::R16G16B16A16_UNORM,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let final_color = (0..(LM_COUNT))
        .map(|_| {
            create_storage_image(
                ImageUsage {
                    transfer_src: true,
                    transfer_dst: true,
                    ..ImageUsage::default()
                },
                Format::R16G16B16A16_UNORM,
            )
        })
        .collect::<Vec<_>>();

        let staging_color = create_storage_image(
            ImageUsage {
                transfer_src: true,
                transfer_dst: true,
                ..ImageUsage::default()
            },
            Format::R16G16B16A16_UNORM,
        );

        Self {
            colors,
            final_color,
            staging_color,
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
            final_color: self.final_color.iter().map(view).collect(),
        }
    }
}
