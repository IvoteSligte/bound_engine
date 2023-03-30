use std::sync::Arc;

use vulkano::{
    buffer::DeviceLocalBuffer,
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{Device, physical::PhysicalDevice},
    memory::allocator::{FreeListAllocator, GenericMemoryAllocator},
    pipeline::{graphics::viewport::Viewport, ComputePipeline},
    swapchain::{
        PresentFuture, Swapchain, SwapchainAcquireFuture, SwapchainCreateInfo,
        SwapchainCreationError, Surface,
    },
    sync::{self, FenceSignalFuture, GpuFuture, JoinFuture}, image::{ImageUsage, SwapchainImage}, format::Format,
};
use winit::dpi::PhysicalSize;
use winit_event_helper::EventHelper;

use crate::{
    command_buffers::{
        get_pathtrace_command_buffers, get_swapchain_command_buffers, CommandBufferCollection,
    },
    descriptor_sets::*,
    event_helper::Data,
    get_color_image,
    lightmap::{LightmapBufferSet, LightmapImages},
    shaders::{self, Shaders}, pipelines::{PathtracePipelines, get_compute_pipeline}, FOV,
};

pub(crate) fn get_swapchain(
    device: &Arc<Device>,
    surface: Arc<Surface>,
    physical_device: Arc<PhysicalDevice>,
    image_format: Format,
    dimensions: PhysicalSize<u32>,
) -> (Arc<Swapchain>, Vec<Arc<SwapchainImage>>) {
    let capabilities = physical_device
        .surface_capabilities(&surface, Default::default())
        .unwrap();

    Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: capabilities.min_image_count + 1, // TODO: improve
            image_format: Some(image_format),
            image_extent: dimensions.into(),
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
    viewport: &mut Viewport,
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue: Arc<vulkano::device::Queue>,
    constant_buffer: Arc<DeviceLocalBuffer<shaders::ty::ConstantBuffer>>,
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
    pathtrace_pipelines: &mut PathtracePipelines,
    shaders: Shaders,
    memory_allocator: &GenericMemoryAllocator<Arc<FreeListAllocator>>,
    queue_family_index: u32,
    descriptor_set_allocator: &StandardDescriptorSetAllocator,
    lightmap_pipelines: Vec<Arc<ComputePipeline>>,
    real_time_buffer: Arc<DeviceLocalBuffer<shaders::ty::RealTimeBuffer>>,
    bvh_buffer: Arc<DeviceLocalBuffer<shaders::ty::GpuBVH>>,
    mutable_buffer: Arc<DeviceLocalBuffer<shaders::ty::MutableData>>,
    lightmap_images: LightmapImages,
    lightmap_buffers: LightmapBufferSet,
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

        viewport.dimensions = eh.dimensions.to_array();

        let constant_data = shaders::ty::ConstantBuffer {
            ratio: [FOV, -FOV * viewport.dimensions[1] / viewport.dimensions[0]],
        };

        // TODO: make this a lot cleaner
        let mut builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .update_buffer(Box::new(constant_data), constant_buffer.clone(), 0)
            .unwrap();
        let command_buffer = builder.build().unwrap();

        if let Some(future) = fences[previous_fence_index].clone() {
            future
                .then_signal_fence_and_flush()
                .unwrap()
                .wait(None)
                .unwrap();
        }

        sync::now(device.clone())
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        pathtrace_pipelines.direct =
            get_compute_pipeline(device.clone(), shaders.direct.clone(), &());

        let color_image = get_color_image(memory_allocator, dimensions, queue_family_index);

        let descriptor_sets = get_compute_descriptor_sets(
            &descriptor_set_allocator,
            pathtrace_pipelines.clone(),
            lightmap_pipelines.clone(),
            real_time_buffer.clone(),
            bvh_buffer.clone(),
            mutable_buffer.clone(),
            constant_buffer.clone(),
            color_image.clone(),
            lightmap_images.clone(),
            lightmap_buffers.clone(),
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
            pathtrace_pipelines.clone(),
            dimensions,
            descriptor_sets.clone(),
            lightmap_buffers.clone(),
        );
    }

    true
}
