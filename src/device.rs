use std::sync::Arc;

use vulkano::{instance::Instance, swapchain::Surface, device::{DeviceExtensions, physical::{PhysicalDevice, PhysicalDeviceType}, Queue, Device, DeviceCreateInfo, QueueCreateInfo}, sync::PipelineStage};

pub(crate) fn select_physical_device<'a>(
    instance: Arc<Instance>,
    surface: &'a Surface,
    device_extensions: &'a DeviceExtensions,
) -> (Arc<PhysicalDevice>, u32) {
    instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(device_extensions))
        .filter(|p| p.properties().device_type == PhysicalDeviceType::DiscreteGpu)
        .find_map(|p| {
            p.queue_family_properties()
                .iter()
                .position(|q| {
                    q.supports_stage(PipelineStage::ComputeShader)
                        && q.supports_stage(PipelineStage::Host)
                    //&& q.supports_stage(PipelineStage::Copy)
                    //&& q.supports_stage(PipelineStage::Blit)
                })
                .map(|q| (p, q as u32))
                .filter(|(p, q)| p.surface_support(*q, surface).unwrap_or(false))
        })
        .unwrap()
}

pub(crate) fn get_device(
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