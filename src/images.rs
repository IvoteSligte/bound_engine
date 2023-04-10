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
    shaders::{LIGHTMAP_COUNT, LIGHTMAP_SIZE, RAYS_INDIRECT},
};

#[derive(Clone)]
pub(crate) struct Images {
    // TODO: swapchain images
    pub(crate) color: Arc<StorageImage>,
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
    pub(crate) fn new(allocators: Arc<Allocators>, queue: Arc<Queue>) -> Self {
        let dimensions = ImageDimensions::Dim3d {
            width: LIGHTMAP_SIZE,
            height: LIGHTMAP_SIZE,
            depth: LIGHTMAP_SIZE,
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

        let colors = (0..RAYS_INDIRECT)
            .map(|_| {
                (0..(LIGHTMAP_COUNT))
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

        let syncs = (0..LIGHTMAP_COUNT)
            .map(|_| {
                create_storage_image(
                    ImageUsage {
                        transfer_src: true,
                        transfer_dst: true,
                        ..ImageUsage::default()
                    },
                    Format::R32_UINT,
                )
            })
            .collect();

        let staging_color = create_storage_image(
            ImageUsage {
                transfer_src: true,
                transfer_dst: true,
                ..ImageUsage::default()
            },
            Format::R16G16B16A16_UNORM,
        );

        let staging_sync = create_storage_image(
            ImageUsage {
                transfer_src: true,
                transfer_dst: true,
                ..ImageUsage::default()
            },
            Format::R32_UINT,
        );

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
