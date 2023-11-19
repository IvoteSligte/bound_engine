use std::sync::Arc;

use vulkano::{
    format::Format,
    image::{view::ImageView, ImageCreateFlags, ImageDimensions, ImageUsage, SwapchainImage},
};
use winit::window::Window;

use crate::{allocator::Allocators, shaders};

use self::custom::CustomImage;

#[derive(Clone)]
pub struct Images {
    pub render: Arc<CustomImage>,
    pub depth: Arc<CustomImage>,
    pub swapchain: Vec<Arc<SwapchainImage>>,
    pub energy: Vec<Arc<CustomImage>>,
}

impl Images {
    pub fn new(
        allocators: Arc<Allocators>,
        window: Arc<Window>,
        swapchain_images: Vec<Arc<SwapchainImage>>,
    ) -> Self {
        Self {
            render: create_render(allocators.clone(), window.clone()),
            depth: create_depth(allocators.clone(), window.clone()),
            swapchain: swapchain_images,
            energy: create_energy(allocators.clone()),
        }
    }

    pub fn views(&self) -> ImageViews {
        ImageViews {
            render: ImageView::new_default(self.render.clone()).unwrap(),
            depth: ImageView::new_default(self.depth.clone()).unwrap(),
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

pub fn create_depth(allocators: Arc<Allocators>, window: Arc<Window>) -> Arc<CustomImage> {
    CustomImage::with_usage(
        &allocators.memory,
        ImageDimensions::Dim2d {
            width: window.inner_size().width,
            height: window.inner_size().height,
            array_layers: 1,
        },
        Format::D32_SFLOAT,
        ImageUsage::DEPTH_STENCIL_ATTACHMENT,
        ImageCreateFlags::empty(),
    )
    .unwrap()
}

pub fn create_energy(allocators: Arc<Allocators>) -> Vec<Arc<CustomImage>> {
    (0..3)
        .map(|_| {
            CustomImage::with_usage(
                &allocators.memory,
                ImageDimensions::Dim3d {
                    width: shaders::CELLS,
                    height: shaders::CELLS,
                    depth: shaders::CELLS,
                },
                Format::R32_SFLOAT,
                ImageUsage::STORAGE,
                ImageCreateFlags::empty(),
            )
            .unwrap()
        })
        .collect()
}

#[derive(Clone)]
pub struct ImageViews {
    pub render: Arc<ImageView<CustomImage>>,
    pub depth: Arc<ImageView<CustomImage>>,
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
            true // TODO: improve; copied from [vulkano::image::StorageImage]
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
