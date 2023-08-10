use std::sync::Arc;

use vulkano::{
    device::{Device, DeviceExtensions, Features, Queue},
    instance::debug::{
        DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessenger,
        DebugUtilsMessengerCreateInfo,
    },
    render_pass::{Framebuffer, RenderPass},
    swapchain::Swapchain,
};
use winit::window::Window;

use crate::{
    allocator::Allocators,
    buffer::Buffers,
    command_buffer::CommandBuffers,
    descriptor_sets::DescriptorSets,
    device::{create_device, select_physical_device},
    fences::Fences,
    image::Images,
    instance::create_instance,
    pipeline::Pipelines,
    render_pass,
    shaders::{self, Shaders},
    swapchain::create,
    FOV,
};

pub struct State {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub swapchain: Arc<Swapchain>,
    pub shaders: Shaders,
    pub render_pass: Arc<RenderPass>,
    pub frame_buffer: Arc<Framebuffer>,
    pub pipelines: Pipelines,
    pub buffers: Buffers,
    pub images: Images,
    pub allocators: Arc<Allocators>,
    pub descriptor_sets: DescriptorSets,
    pub command_buffers: CommandBuffers,
    pub real_time_data: shaders::RealTimeBuffer, // TODO: struct abstraction
    pub fences: Fences,
    #[cfg(debug_assertions)]
    _debugger: DebugUtilsMessenger,
}

impl State {
    pub fn new(window: Arc<Window>) -> Self {
        let instance = create_instance();

        let surface =
            vulkano_win::create_surface_from_winit(window.clone(), instance.clone()).unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ext_shader_atomic_float: true,
            ..DeviceExtensions::empty()
        };

        let device_features = Features {
            shader_buffer_float32_atomic_add: true,
            ..Features::empty()
        };

        let (physical_device, queue_family_index) = select_physical_device(
            instance.clone(),
            &surface,
            &device_extensions,
            &device_features,
        );

        let (device, queue) = create_device(
            physical_device.clone(),
            device_extensions,
            device_features,
            queue_family_index,
        );

        let (swapchain, swapchain_images) = create(
            device.clone(),
            surface.clone(),
            window.clone(),
            physical_device.clone(),
        );

        let allocators = Allocators::new(device.clone());

        let buffers = Buffers::new(allocators.clone(), queue.clone());

        let images = Images::new(
            device.clone(),
            allocators.clone(),
            window.clone(),
            swapchain_images.clone(),
        );

        let shaders = Shaders::load(device.clone());

        let render_pass = render_pass::create(device.clone());
        let frame_buffer = render_pass::frame_buffer(render_pass.clone(), images.views().render);

        let pipelines = Pipelines::new(
            device.clone(),
            shaders.clone(),
            render_pass.clone(),
            window.clone(),
        );

        let descriptor_sets = DescriptorSets::new(
            allocators.clone(),
            pipelines.clone(),
            buffers.clone(),
            images.clone(),
        );

        let command_buffers = CommandBuffers::new(
            allocators.clone(),
            queue.clone(),
            frame_buffer.clone(),
            pipelines.clone(),
            images.clone(),
            descriptor_sets.clone(),
        );

        let real_time_data = shaders::RealTimeBuffer {
            rotation: Default::default(),
            position: Default::default(),
            lightmapOrigin: Default::default(),
            deltaLightmapOrigins: Default::default(),
            screenSize: [window.inner_size().width as f32, window.inner_size().height as f32],
            fov: FOV,
        };

        let fences = Fences::new(images.swapchain.len());

        #[cfg(debug_assertions)]
        let debugger = unsafe {
            DebugUtilsMessenger::new(
                // TODO: add message_type and message_severity marker to print output
                instance,
                DebugUtilsMessengerCreateInfo {
                    message_severity: DebugUtilsMessageSeverity::INFO
                        | DebugUtilsMessageSeverity::WARNING
                        | DebugUtilsMessageSeverity::ERROR,
                    message_type: DebugUtilsMessageType::GENERAL
                        | DebugUtilsMessageType::VALIDATION
                        | DebugUtilsMessageType::PERFORMANCE,
                    ..DebugUtilsMessengerCreateInfo::user_callback(Arc::new(|msg| {
                        println!("[DEBUG]: {:?}", msg.description);
                    }))
                },
            )
        }
        .unwrap();

        Self {
            device,
            queue,
            swapchain,
            shaders,
            render_pass,
            frame_buffer,
            pipelines,
            buffers,
            images,
            allocators,
            descriptor_sets,
            command_buffers,
            real_time_data,
            fences,
            #[cfg(debug_assertions)]
            _debugger: debugger,
        }
    }
}
