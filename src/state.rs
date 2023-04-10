use std::sync::Arc;

use vulkano::{
    device::{Device, DeviceExtensions, Queue},
    swapchain::Swapchain,
};
use winit::window::Window;

use crate::{
    allocators::Allocators,
    buffers::Buffers,
    command_buffers::CommandBuffers,
    device::{get_device, select_physical_device},
    fences::Fences,
    images::Images,
    instance::get_instance,
    pipelines::Pipelines,
    shaders::{self, Shaders},
    swapchain::get_swapchain,
};

pub(crate) struct State {
    pub(crate) device: Arc<Device>,
    pub(crate) queue: Arc<Queue>,
    pub(crate) swapchain: Arc<Swapchain>,
    pub(crate) shaders: Shaders,
    pub(crate) pipelines: Pipelines,
    pub(crate) buffers: Buffers,
    pub(crate) images: Images,
    pub(crate) allocators: Arc<Allocators>,
    pub(crate) command_buffers: CommandBuffers,
    pub(crate) real_time_data: shaders::ty::RealTimeBuffer,
    pub(crate) fences: Fences,
}

impl State {
    pub(crate) fn new(window: Arc<Window>) -> Self {
        let instance = get_instance();

        let surface =
            vulkano_win::create_surface_from_winit(window.clone(), instance.clone()).unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) =
            select_physical_device(instance, &surface, &device_extensions);

        let (device, queue) = get_device(
            physical_device.clone(),
            device_extensions,
            queue_family_index,
        );

        let (swapchain, swapchain_images) = get_swapchain(
            device.clone(),
            surface.clone(),
            window.clone(),
            physical_device.clone(),
        );

        let shaders = Shaders::load(device.clone());

        let pipelines = Pipelines::from_shaders(device.clone(), shaders.clone(), window.clone());
        
        let allocators = Allocators::new(device.clone());
        
        let buffers = Buffers::new(allocators.clone(), queue.clone());

        let images = Images::new(
            allocators.clone(),
            window.clone(),
            queue.clone(),
            swapchain_images.clone(),
        );

        let command_buffers = CommandBuffers::new(
            allocators.clone(),
            queue.clone(),
            pipelines.clone(),
            window.clone(),
            buffers.clone(),
            images.clone(),
        );

        let real_time_data = shaders::ty::RealTimeBuffer::default();

        let fences = Fences::new(images.swapchain.len());

        Self {
            device,
            queue,
            swapchain,
            shaders,
            pipelines,
            buffers,
            images,
            allocators,
            command_buffers,
            real_time_data,
            fences,
        }
    }
}
