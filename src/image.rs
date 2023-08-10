use std::sync::Arc;

use vulkano::{
    device::Device,
    format::Format,
    image::{
        view::{ImageView, ImageViewCreateInfo},
        ImageCreateFlags, ImageDimensions, ImageUsage, ImageViewAbstract, SwapchainImage,
    },
    sampler::{BorderColor, Sampler, SamplerAddressMode, SamplerCreateInfo},
};
use winit::window::Window;

use crate::{
    allocator::Allocators,
    shaders::{LM_LAYERS, LM_SIZE, RADIANCE_SH_COEFS, RADIANCE_SIZE},
};

use self::custom::CustomImage;

#[derive(Clone)]
pub struct Images {
    pub render: Arc<CustomImage>,
    pub sdf: SdfImages,
    pub radiance: RadianceImages,
    pub swapchain: Vec<Arc<SwapchainImage>>,
}

impl Images {
    pub fn new(
        device: Arc<Device>,
        allocators: Arc<Allocators>,
        window: Arc<Window>,
        swapchain_images: Vec<Arc<SwapchainImage>>,
    ) -> Self {
        Self {
            render: create_render(allocators.clone(), window),
            sdf: SdfImages::new(device.clone(), allocators.clone()),
            radiance: RadianceImages::new(device, allocators),
            swapchain: swapchain_images,
        }
    }

    pub fn views(&self) -> ImageViewsCollection {
        ImageViewsCollection {
            render: ImageView::new_default(self.render.clone()).unwrap(),
            sdf: self.sdf.views(),
            radiance: self.radiance.views(),
        }
    }
}

pub fn create_render(allocators: Arc<Allocators>, window: Arc<Window>) -> Arc<CustomImage> {
    CustomImage::with_usage(
        &allocators.memory,
        ImageDimensions::Dim2d {
            width: window.inner_size().width,
            height: window.inner_size().height,
            array_layers: 1,
        },
        Format::R16G16B16A16_UNORM, // double precision for copying to srgb
        ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
        ImageCreateFlags::empty(),
    )
    .unwrap()
}

#[derive(Clone)]
pub struct ImageViewsCollection {
    pub render: Arc<ImageView<CustomImage>>,
    pub sdf: ImageViews,
    pub radiance: ImageViews,
}

#[derive(Clone)]
pub struct ImageViews {
    pub storage: Vec<Arc<dyn ImageViewAbstract>>,
    pub sampled: Vec<Arc<dyn ImageViewAbstract>>,
}

impl ImageViews {
    fn create_views(
        images: &[Arc<CustomImage>],
        usage: ImageUsage,
    ) -> Vec<Arc<dyn ImageViewAbstract>> {
        images
            .iter()
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
    }

    fn from_images(images: &[Arc<CustomImage>]) -> Self {
        let storage = Self::create_views(images, ImageUsage::STORAGE);
        let sampled = Self::create_views(images, ImageUsage::SAMPLED);
        Self { storage, sampled }
    }
}

#[derive(Clone)]
pub struct RadianceImages {
    images: Vec<Arc<CustomImage>>,
    sampler: Arc<Sampler>,
}

impl RadianceImages {
    pub fn new(device: Arc<Device>, allocators: Arc<Allocators>) -> Self {
        let dimensions = ImageDimensions::Dim3d {
            width: RADIANCE_SIZE,
            height: RADIANCE_SIZE,
            depth: RADIANCE_SIZE,
        };

        // image for every layer and every spherical harmonic coefficient
        let images = (0..(RADIANCE_SH_COEFS * LM_LAYERS))
            .map(|_| {
                CustomImage::with_usage(
                    &allocators.memory,
                    dimensions,
                    Format::R16G16B16A16_SFLOAT,
                    ImageUsage::STORAGE | ImageUsage::SAMPLED,
                    ImageCreateFlags::empty(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                address_mode: [SamplerAddressMode::ClampToBorder; 3],
                border_color: BorderColor::FloatTransparentBlack,
                ..SamplerCreateInfo::simple_repeat_linear_no_mipmap()
            },
        )
        .unwrap();

        Self { images, sampler }
    }

    pub fn sampler(&self) -> Arc<Sampler> {
        self.sampler.clone()
    }

    pub fn views(&self) -> ImageViews {
        ImageViews::from_images(&self.images)
    }
}

#[derive(Clone)]
pub struct SdfImages {
    images: Vec<Arc<CustomImage>>,
    sampler: Arc<Sampler>,
}

impl SdfImages {
    pub fn new(device: Arc<Device>, allocators: Arc<Allocators>) -> Self {
        let dimensions = ImageDimensions::Dim3d {
            width: LM_SIZE,
            height: LM_SIZE,
            depth: LM_SIZE,
        };

        let images = (0..LM_LAYERS)
            .map(|_| {
                CustomImage::with_usage(
                    &allocators.memory,
                    dimensions,
                    Format::R16_SFLOAT,
                    ImageUsage::STORAGE | ImageUsage::SAMPLED,
                    ImageCreateFlags::empty(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        let sampler = Sampler::new(
            device,
            SamplerCreateInfo {
                address_mode: [SamplerAddressMode::ClampToBorder; 3],
                border_color: BorderColor::FloatTransparentBlack,
                ..SamplerCreateInfo::simple_repeat_linear_no_mipmap()
            },
        )
        .unwrap();

        Self { images, sampler }
    }

    pub fn sampler(&self) -> Arc<Sampler> {
        self.sampler.clone()
    }

    pub fn views(&self) -> ImageViews {
        ImageViews::from_images(&self.images)
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
