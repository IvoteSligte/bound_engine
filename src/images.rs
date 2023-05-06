use std::sync::Arc;

use vulkano::{
    format::Format,
    image::{
        view::{ImageView, ImageViewCreateInfo},
        ImageCreateFlags, ImageDimensions, ImageUsage, ImageViewAbstract,
        SwapchainImage,
    },
    sampler::Sampler,
};
use winit::window::Window;

use crate::{
    allocators::Allocators,
    shaders::{LM_COUNT, LM_RAYS, LM_SIZE},
};

use self::image::CustomImage;

#[derive(Clone)]
pub(crate) struct Images {
    pub(crate) color: Arc<CustomImage>,
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
    pub(crate) color: Arc<ImageView<CustomImage>>,
    pub(crate) lightmap: LightmapImageViews,
}

pub(crate) fn create_color_image(
    allocators: Arc<Allocators>,
    window: Arc<Window>,
) -> Arc<CustomImage> {
    CustomImage::with_usage(
        &allocators.memory,
        ImageDimensions::Dim2d {
            width: window.inner_size().width,
            height: window.inner_size().height,
            array_layers: 1,
        },
        Format::R16G16B16A16_UNORM, // double precision for copying to srgb
        ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
        ImageCreateFlags::empty(),
    )
    .unwrap()
}

#[derive(Clone)]
pub(crate) struct LightmapImages {
    pub(crate) colors: Vec<Vec<Arc<CustomImage>>>,
    pub(crate) staging_color: Arc<CustomImage>,
    pub(crate) sdfs: Vec<Arc<CustomImage>>,
    pub(crate) materials: Vec<Arc<CustomImage>>,
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
            CustomImage::with_usage(
                &allocators.memory,
                dimensions,
                format,
                ImageUsage::STORAGE | usage,
                ImageCreateFlags::empty(),
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
        let views = |imgs: &Vec<Arc<CustomImage>>, usage: ImageUsage| {
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

        let colors_storage = self
            .colors
            .iter()
            .map(|imgs| views(imgs, ImageUsage::STORAGE))
            .collect();
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

mod image {
    use vulkano::{
        device::{Device, DeviceOwned},
        format::Format,
        image::{
            sys::{Image, ImageCreateInfo, RawImage},
            traits::ImageContent,
            ImageAccess, ImageCreateFlags, ImageDescriptorLayouts, ImageDimensions, ImageError,
            ImageInner, ImageLayout, ImageUsage,
        },
        memory::{
            allocator::{
                AllocationCreateInfo, AllocationType, MemoryAllocatePreference, MemoryAllocator,
                MemoryUsage,
            },
            DedicatedAllocation,
        },
        sync::Sharing,
    };

    use std::{
        hash::{Hash, Hasher},
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        },
    };

    /// ADAPTED FROM vulkano::image::StorageImage
    #[derive(Debug)]
    pub(crate) struct CustomImage {
        inner: Arc<Image>,

        // If false, then the image is still `Undefined`.
        layout_initialized: AtomicBool,
    }

    impl CustomImage {
        /// Creates an image with a specific usage
        pub(crate) fn with_usage(
            allocator: &(impl MemoryAllocator + ?Sized),
            dimensions: ImageDimensions,
            format: Format,
            usage: ImageUsage,
            flags: ImageCreateFlags,
        ) -> Result<Arc<CustomImage>, ImageError> {
            assert!(!flags.intersects(ImageCreateFlags::DISJOINT)); // INFO: should be safe, but might not be

            let raw_image = RawImage::new(
                allocator.device().clone(),
                ImageCreateInfo {
                    flags,
                    dimensions,
                    format: Some(format),
                    usage,
                    sharing: Sharing::Exclusive, // TODO: Sharing::Concurrent implementation
                    ..Default::default()
                },
            )?;
            let requirements = raw_image.memory_requirements()[0];
            let res = unsafe {
                allocator.allocate_unchecked(
                    requirements,
                    AllocationType::NonLinear,
                    AllocationCreateInfo {
                        usage: MemoryUsage::DeviceOnly,
                        allocate_preference: MemoryAllocatePreference::Unknown,
                        ..AllocationCreateInfo::default()
                    },
                    Some(DedicatedAllocation::Image(&raw_image)),
                )
            };

            match res {
                Ok(alloc) => {
                    let inner =
                        Arc::new(raw_image.bind_memory([alloc]).map_err(|(err, _, _)| err)?);

                    Ok(Arc::new(CustomImage {
                        inner,
                        layout_initialized: AtomicBool::new(false),
                    }))
                }
                Err(err) => Err(err.into()),
            }
        }
    }

    unsafe impl DeviceOwned for CustomImage {
        #[inline]
        fn device(&self) -> &Arc<Device> {
            self.inner.device()
        }
    }

    unsafe impl ImageAccess for CustomImage {
        #[inline]
        fn inner(&self) -> ImageInner<'_> {
            ImageInner {
                image: &self.inner,
                first_layer: 0,
                num_layers: self.inner.dimensions().array_layers(),
                first_mipmap_level: 0,
                num_mipmap_levels: 1,
            }
        }

        #[inline]
        fn initial_layout_requirement(&self) -> ImageLayout {
            ImageLayout::General
        }

        #[inline]
        fn final_layout_requirement(&self) -> ImageLayout {
            ImageLayout::General
        }

        #[inline]
        unsafe fn layout_initialized(&self) {
            self.layout_initialized.store(true, Ordering::Relaxed);
        }

        #[inline]
        fn is_layout_initialized(&self) -> bool {
            self.layout_initialized.load(Ordering::Relaxed)
        }

        #[inline]
        fn descriptor_layouts(&self) -> Option<ImageDescriptorLayouts> {
            Some(ImageDescriptorLayouts {
                storage_image: ImageLayout::General,
                combined_image_sampler: ImageLayout::ShaderReadOnlyOptimal,
                sampled_image: ImageLayout::ShaderReadOnlyOptimal,
                input_attachment: ImageLayout::ColorAttachmentOptimal,
            })
        }
    }

    unsafe impl<P> ImageContent<P> for CustomImage {
        fn matches_format(&self) -> bool {
            true // FIXME: copied from vulkano::image::StorageImage
        }
    }

    impl PartialEq for CustomImage {
        #[inline]
        fn eq(&self, other: &Self) -> bool {
            self.inner() == other.inner()
        }
    }

    impl Eq for CustomImage {}

    impl Hash for CustomImage {
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.inner().hash(state);
        }
    }
}
