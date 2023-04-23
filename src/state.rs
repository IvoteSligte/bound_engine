use std::sync::Arc;

use vulkano::{
    device::{Device, DeviceExtensions, Queue},
    sampler::{BorderColor, Sampler, SamplerAddressMode, SamplerCreateInfo},
    swapchain::Swapchain,
};
use winit::window::Window;

use crate::{
    allocators::Allocators,
    buffers::Buffers,
    command_buffers::CommandBuffers,
    device::{create_device, select_physical_device},
    fences::Fences,
    images::Images,
    instance::create_instance,
    pipelines::Pipelines,
    shaders::{self, Shaders},
    swapchain::create_swapchain,
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
    pub(crate) real_time_data: shaders::RealTimeBuffer, // TODO: struct abstraction
    pub(crate) fences: Fences,
}

impl State {
    pub(crate) fn new(window: Arc<Window>) -> Self {
        let instance = create_instance();

        let surface =
            vulkano_win::create_surface_from_winit(window.clone(), instance.clone()).unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) =
            select_physical_device(instance, &surface, &device_extensions);

        let (device, queue) = create_device(
            physical_device.clone(),
            device_extensions,
            queue_family_index,
        );

        let (swapchain, swapchain_images) = create_swapchain(
            device.clone(),
            surface.clone(),
            window.clone(),
            physical_device.clone(),
        );

        let shaders = Shaders::load(device.clone());

        let pipelines = Pipelines::from_shaders(device.clone(), shaders.clone(), window.clone());

        let allocators = Allocators::new(device.clone());

        let buffers = Buffers::new(allocators.clone(), queue.clone());

        // TODO: clean up
        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                address_mode: [SamplerAddressMode::ClampToBorder; 3],
                border_color: BorderColor::FloatTransparentBlack,
                ..SamplerCreateInfo::simple_repeat_linear_no_mipmap()
            },
        )
        .unwrap();

        let images = Images::new(
            allocators.clone(),
            window.clone(),
            queue.clone(),
            swapchain_images.clone(),
            sampler.clone(),
        );

        let command_buffers = CommandBuffers::new(
            allocators.clone(),
            queue.clone(),
            pipelines.clone(),
            window.clone(),
            buffers.clone(),
            images.clone(),
        );

        let real_time_data = shaders::RealTimeBuffer {
            rotation: Default::default(),
            previousRotation: Default::default(),
            position: Default::default(),
            previousPosition: Default::default(),
            lightmapOrigin: Default::default(),
            deltaLightmapOrigins: Default::default(),
        };

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
