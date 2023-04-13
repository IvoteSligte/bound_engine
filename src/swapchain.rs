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
    command_buffers::{create_pathtrace_command_buffers, create_swapchain_command_buffers},
    create_color_image,
    descriptor_sets::*,
    event_helper::Data,
    pipelines::create_compute_pipeline,
    shaders::{self},
    FOV,
};

pub(crate) fn create_swapchain(
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
            image_usage: ImageUsage {
                transfer_dst: true,
                ..ImageUsage::empty()
            },
            ..Default::default()
        },
    )
    .unwrap()
}

pub(crate) fn recreate_swapchain(
    // TODO: refactor eh.state.
    eh: &mut EventHelper<Data>,
) -> bool {
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

        eh.state.pipelines.direct = create_compute_pipeline(
            eh.state.device.clone(),
            eh.state.shaders.direct.clone(),
            &shaders::DirectSpecializationConstants {
                RATIO_X: FOV,
                RATIO_Y: -FOV * (dimensions.height as f32) / (dimensions.width as f32),
            },
        );

        eh.state.images.color = create_color_image(
            eh.state.allocators.clone(),
            eh.window.clone(),
            eh.state.queue.clone(),
        );

        // TODO: move to command buffer init
        let descriptor_sets = create_compute_descriptor_sets(
            eh.state.allocators.clone(),
            eh.state.pipelines.clone(),
            eh.state.buffers.clone(),
            eh.state.images.clone(),
        );

        eh.state.images.swapchain = new_swapchain_images;

        eh.state.command_buffers.swapchains = create_swapchain_command_buffers(
            eh.state.allocators.clone(),
            eh.state.queue.clone(),
            eh.state.images.clone(),
        );

        eh.state.command_buffers.pathtraces = create_pathtrace_command_buffers(
            eh.state.allocators.clone(),
            eh.state.queue.clone(),
            eh.state.pipelines.clone(),
            eh.window.clone(),
            descriptor_sets.clone(),
        );
    }

    true
}