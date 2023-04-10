use std::sync::Arc;

use vulkano::{
    device::{Device, Queue, DeviceExtensions},
    swapchain::{Surface, Swapchain},
};
use winit::window::Window;

use crate::{
    allocators::Allocators,
    buffers::Buffers,
    command_buffers::CommandBufferCollection,
    descriptor_sets::{DescriptorSetCollection, get_compute_descriptor_sets},
    images::Images,
    pipelines::Pipelines,
    shaders::{self, Shaders}, instance::get_instance, device::{select_physical_device, get_device}, swapchain::get_swapchain, fences::Fences,
};

pub(crate) struct State {
    pub(crate) surface: Arc<Surface>,
    pub(crate) device: Arc<Device>,
    pub(crate) queue: Arc<Queue>,
    pub(crate) swapchain: Arc<Swapchain>,
    pub(crate) shaders: Shaders,
    pub(crate) pipelines: Pipelines,
    pub(crate) buffers: Buffers,
    pub(crate) images: Images,
    pub(crate) allocators: Arc<Allocators>,
    pub(crate) descriptor_sets: DescriptorSetCollection,
    pub(crate) command_buffers: CommandBufferCollection,
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

        let (mut swapchain, swapchain_images) = get_swapchain(
            device.clone(),
            surface.clone(),
            window.clone(),
            physical_device.clone(),
        );

        let shaders = Shaders::load(device.clone());

        let mut pipelines =
            Pipelines::from_shaders(device.clone(), shaders.clone(), window.clone());

        let mut real_time_data = shaders::ty::RealTimeBuffer::default();

        let allocators = Allocators::new(device.clone());

        let buffers = Buffers::new(allocators.clone(), queue.clone());

        let mut images = Images::new(
            allocators.clone(),
            window.clone(),
            queue.clone(),
            swapchain_images.clone(),
        );

        let descriptor_sets = get_compute_descriptor_sets(
            allocators.clone(),
            pipelines.clone(),
            buffers.clone(),
            images.clone(),
        );

        let mut command_buffers = CommandBufferCollection::new(
            allocators.clone(),
            queue.clone(),
            pipelines.clone(),
            window.clone(),
            descriptor_sets.clone(),
            buffers.clone(),
            images.clone(),
        );

        let fences = Fences::new(images.swapchain.len());

        Self {
            surface,
            device,
            queue,
            swapchain,
            shaders,
            pipelines,
            buffers,
            images,
            allocators,
            descriptor_sets, // TODO: move to command buffer init
            command_buffers,
            real_time_data, // TODO: move to init
            fences,
        }
    }
}
