use std::sync::Arc;

use vulkano::{
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Features, Queue, QueueCreateInfo, QueueFlags,
    },
    instance::Instance,
    swapchain::Surface,
};

pub(crate) fn select_physical_device<'a>(
    instance: Arc<Instance>,
    surface: &'a Surface,
    extensions: &'a DeviceExtensions,
    features: &'a Features,
) -> (Arc<PhysicalDevice>, u32) {
    instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.properties().device_type == PhysicalDeviceType::DiscreteGpu)
        .filter(|p| p.supported_extensions().contains(extensions))
        .filter(|p| p.supported_features().contains(features))
        .find_map(|p| {
            p.queue_family_properties()
                .iter()
                .position(|q| q.queue_flags.contains(QueueFlags::COMPUTE))
                .map(|q| (p, q as u32))
                .filter(|(p, q)| p.surface_support(*q, surface).unwrap_or(false))
        })
        .unwrap()
}

pub(crate) fn create_device(
    physical_device: Arc<PhysicalDevice>,
    device_extensions: DeviceExtensions,
    queue_family_index: u32,
) -> (Arc<Device>, Arc<Queue>) {
    let (device, mut queues) = Device::new(
        physical_device.clone(),
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..QueueCreateInfo::default()
            }],
            enabled_extensions: device_extensions,
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();
    (device, queue)
}
