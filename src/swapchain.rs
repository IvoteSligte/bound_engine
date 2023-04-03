use std::sync::Arc;

use vulkano::{
    buffer::{DeviceLocalBuffer, BufferAccess},
    command_buffer::{
        allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{physical::PhysicalDevice, Device},
    format::Format,
    image::{ImageUsage, SwapchainImage},
    memory::allocator::{FreeListAllocator, GenericMemoryAllocator},
    swapchain::{
        PresentFuture, Surface, Swapchain, SwapchainAcquireFuture, SwapchainCreateInfo,
        SwapchainCreationError,
    },
    sync::{FenceSignalFuture, GpuFuture, JoinFuture},
};
use winit::window::Window;
use winit_event_helper::EventHelper;

use crate::{
    command_buffers::{
        get_pathtrace_command_buffers, get_swapchain_command_buffers, CommandBufferCollection,
    },
    descriptor_sets::*,
    event_helper::Data,
    get_color_image,
    lightmap::{LightmapBufferSet, LightmapImages},
    pipelines::{get_compute_pipeline, Pipelines},
    shaders::{self, Shaders},
    FOV,
};

pub(crate) fn get_swapchain(
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
    eh: &mut EventHelper<Data>,
    swapchain: &mut Arc<Swapchain>,
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue: Arc<vulkano::device::Queue>,
    fences: Vec<
        Option<
            Arc<
                FenceSignalFuture<
                    PresentFuture<
                        JoinFuture<
                            vulkano::command_buffer::CommandBufferExecFuture<
                                vulkano::command_buffer::CommandBufferExecFuture<
                                    Box<dyn GpuFuture>,
                                >,
                            >,
                            SwapchainAcquireFuture,
                        >,
                    >,
                >,
            >,
        >,
    >,
    previous_fence_index: usize,
    device: Arc<Device>,
    pipelines: &mut Pipelines,
    shaders: Shaders,
    memory_allocator: &GenericMemoryAllocator<Arc<FreeListAllocator>>,
    queue_family_index: u32,
    descriptor_set_allocator: &StandardDescriptorSetAllocator,
    real_time_buffer: Arc<DeviceLocalBuffer<shaders::ty::RealTimeBuffer>>,
    bvh_buffer: Arc<DeviceLocalBuffer<shaders::ty::GpuBVH>>,
    mutable_buffer: Arc<DeviceLocalBuffer<shaders::ty::MutableData>>,
    lightmap_images: LightmapImages,
    lightmap_buffers: LightmapBufferSet,
    blue_noise_buffer: Arc<dyn BufferAccess>,
    command_buffers: &mut CommandBufferCollection,
) -> bool {
    eh.recreate_swapchain = false;
    let dimensions = eh.window.inner_size(); // TODO: function input
    let (new_swapchain, new_swapchain_images) = match swapchain.recreate(SwapchainCreateInfo {
        image_extent: dimensions.into(),
        ..swapchain.create_info()
    }) {
        Ok(ok) => ok,
        Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => {
            return false;
        }
        Err(err) => panic!("{}", err),
    };
    *swapchain = new_swapchain;

    if eh.window_resized {
        eh.window_resized = false;

        if let Some(future) = fences[previous_fence_index].clone() {
            future
                .then_signal_fence_and_flush()
                .unwrap()
                .wait(None)
                .unwrap();
        }

        pipelines.direct = get_compute_pipeline(
            device.clone(),
            shaders.direct.clone(),
            &shaders::DirectSpecializationConstants {
                RATIO_X: FOV,
                RATIO_Y: -FOV * (dimensions.height as f32) / (dimensions.width as f32),
            },
        );

        let color_image = get_color_image(memory_allocator, eh.window.clone(), queue_family_index);

        let descriptor_sets = get_compute_descriptor_sets(
            &descriptor_set_allocator,
            pipelines.clone(),
            real_time_buffer.clone(),
            bvh_buffer.clone(),
            mutable_buffer.clone(),
            color_image.clone(),
            lightmap_images.clone(),
            lightmap_buffers.clone(),
            blue_noise_buffer.clone(),
        );

        command_buffers.swapchains = get_swapchain_command_buffers(
            command_buffer_allocator,
            queue.clone(),
            color_image.clone(),
            new_swapchain_images.clone(),
        );

        command_buffers.pathtraces = get_pathtrace_command_buffers(
            command_buffer_allocator,
            queue.clone(),
            pipelines.clone(),
            eh.window.clone(),
            descriptor_sets.clone(),
            lightmap_buffers.clone(),
        );
    }

    true
}
