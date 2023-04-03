use std::sync::Arc;

use vulkano::{memory::allocator::MemoryAllocator, image::{StorageImage, ImageDimensions, ImageUsage, ImageCreateFlags}, format::Format};
use winit::window::Window;

pub(crate) fn get_color_image(
    allocator: &(impl MemoryAllocator + ?Sized),
    window: Arc<Window>,
    queue_family_index: u32,
) -> Arc<StorageImage> {
    StorageImage::with_usage(
        allocator,
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
        [queue_family_index],
    )
    .unwrap()
}