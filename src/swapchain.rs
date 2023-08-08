use std::sync::Arc;

use vulkano::{
    device::{physical::PhysicalDevice, Device},
    format::Format,
    image::{ImageUsage, SwapchainImage},
    swapchain::{Surface, Swapchain, SwapchainCreateInfo, SwapchainCreationError},
};
use winit::window::Window;
use winit_event_helper::EventHelper;

use crate::{
    command_buffer::{self, PathtraceCommandBuffers},
    descriptor_sets::*,
    event_helper::Data,
    image, pipeline,
    shaders::{self},
    FOV,
};

pub fn create(
    device: Arc<Device>,
    surface: Arc<Surface>,
    window: Arc<Window>,
    physical_device: Arc<PhysicalDevice>,
) -> (Arc<Swapchain>, Vec<Arc<SwapchainImage>>) {
    let capabilities = physical_device
        .surface_capabilities(&surface, Default::default())
        .unwrap();

    let image_format = physical_device
        .surface_formats(&surface, Default::default())
        .unwrap()
        .iter()
        .max_by_key(|(format, _)| match format {
            Format::R8G8B8A8_SRGB | Format::B8G8R8A8_SRGB => 1,
            _ => 0,
        })
        .unwrap()
        .0;

    Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: capabilities.min_image_count + 1, // TODO: improve
            image_format: Some(image_format),
            image_extent: window.inner_size().into(),
            image_usage: ImageUsage::TRANSFER_DST,
            ..Default::default()
        },
    )
    .unwrap()
}

/// Returns if the swapchain was recreated successfully
pub fn recreate(eh: &mut EventHelper<Data>) -> bool {
    eh.recreate_swapchain = false;
    let dimensions = eh.window.inner_size(); // TODO: function input
    let (new_swapchain, new_swapchain_images) =
        match eh.state.swapchain.recreate(SwapchainCreateInfo {
            image_extent: dimensions.into(),
            ..eh.state.swapchain.create_info()
        }) {
            Ok(ok) => ok,
            Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => {
                return false;
            }
            Err(err) => panic!("{}", err),
        };
    eh.state.swapchain = new_swapchain;

    if eh.window_resized {
        eh.window_resized = false;

        if let Some(future) = eh.state.fences.previous() {
            future.wait(None).unwrap();
        }

        eh.state.pipelines.direct = pipeline::compute(
            eh.state.device.clone(),
            eh.state.shaders.direct.clone(),
            &shaders::DirectSpecializationConstants {
                RATIO_X: FOV,
                RATIO_Y: -FOV * (dimensions.height as f32) / (dimensions.width as f32),
            },
        );

        eh.state.images.color = image::color(eh.state.allocators.clone(), eh.window.clone());

        eh.state.descriptor_sets = DescriptorSets::new(
            eh.state.allocators.clone(),
            eh.state.pipelines.clone(),
            eh.state.buffers.clone(),
            eh.state.images.clone(),
        );

        eh.state.command_buffers.pathtraces.direct = PathtraceCommandBuffers::direct(
            eh.state.allocators.clone(),
            eh.state.queue.clone(),
            eh.state.pipelines.clone(),
            eh.state.descriptor_sets.clone(),
            eh.window.clone(),
        );

        eh.state.images.swapchain = new_swapchain_images;

        eh.state.command_buffers.swapchains = command_buffer::swapchain(
            eh.state.allocators.clone(),
            eh.state.queue.clone(),
            eh.state.images.clone(),
        );
    }

    true
}
