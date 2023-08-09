use std::sync::Arc;

use vulkano::{
    format::Format,
    image::{
        view::{ImageView, ImageViewCreateInfo},
        ImageCreateFlags, ImageDimensions, ImageUsage, ImageViewAbstract, SwapchainImage,
    },
    sampler::Sampler,
};
use winit::window::Window;

use crate::{
    allocator::Allocators,
    shaders::{LM_LAYERS, LM_SIZE},
};

use self::custom::CustomImage;

#[derive(Clone)]
pub struct Images {
    pub color: Arc<CustomImage>,
    pub lightmap: SdfImages,
    pub swapchain: Vec<Arc<SwapchainImage>>,
    pub sampler: Arc<Sampler>,
}

impl Images {
    pub fn new(
        allocators: Arc<Allocators>,
        window: Arc<Window>,
        swapchain_images: Vec<Arc<SwapchainImage>>,
        sampler: Arc<Sampler>,
    ) -> Self {
        Self {
            color: color(allocators.clone(), window),
            lightmap: SdfImages::new(allocators.clone()),
            swapchain: swapchain_images,
            sampler,
        }
    }

    pub fn views(&self) -> ImageViews {
        ImageViews {
            color: ImageView::new_default(self.color.clone()).unwrap(),
            lightmap: self.lightmap.views(),
        }
    }

    pub fn sampler(&self) -> Arc<Sampler> {
        self.sampler.clone()
    }
}

#[derive(Clone)]
pub struct ImageViews {
    pub color: Arc<ImageView<CustomImage>>,
    pub lightmap: SdfImageViews,
}

pub fn color(allocators: Arc<Allocators>, window: Arc<Window>) -> Arc<CustomImage> {
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
pub struct SdfImages(Vec<Arc<CustomImage>>);

#[derive(Clone)]
pub struct SdfImageViews {
    pub storage: Vec<Arc<dyn ImageViewAbstract>>,
    pub sampled: Vec<Arc<dyn ImageViewAbstract>>,
}

impl SdfImages {
    pub fn new(allocators: Arc<Allocators>) -> Self {
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

        // FIXME: sdf staging + moving
        let sdfs = (0..LM_LAYERS)
            .map(|_| {
                create_storage_image(
                    ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                    Format::R16_SFLOAT,
                )
            })
            .collect::<Vec<_>>();

        Self(sdfs)
    }

    pub fn views(&self) -> SdfImageViews {
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

        let storage = views(&self.0, ImageUsage::STORAGE);
        let sampled = views(&self.0, ImageUsage::SAMPLED);

        SdfImageViews { storage, sampled }
    }
}

mod custom {
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

    /// ADAPTED FROM [vulkano::image::StorageImage]
    #[derive(Debug)]
    pub struct CustomImage {
        inner: Arc<Image>,
        /// If false, the image is in the `Undefined` layout.
        layout_initialized: AtomicBool,
    }

    impl CustomImage {
        /// Creates an image with a specific usage
        pub fn with_usage(
            allocator: &(impl MemoryAllocator + ?Sized),
            dimensions: ImageDimensions,
            format: Format,
            usage: ImageUsage,
            flags: ImageCreateFlags,
        ) -> Result<Arc<CustomImage>, ImageError> {
            assert!(!flags.intersects(ImageCreateFlags::DISJOINT)); // INFO: should be safe even without this, but might not be

            let raw_image = RawImage::new(
                allocator.device().clone(),
                ImageCreateInfo {
                    flags,
                    dimensions,
                    format: Some(format),
                    usage,
                    sharing: Sharing::Exclusive, // TODO: [Sharing::Concurrent] implementation
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
            true // FIXME: copied from [vulkano::image::StorageImage]
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
