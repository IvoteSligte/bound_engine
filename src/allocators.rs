use std::sync::Arc;

use vulkano::{
    command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    memory::allocator::StandardMemoryAllocator, device::Device,
};

pub(crate) struct Allocators {
    pub(crate) memory: StandardMemoryAllocator,
    pub(crate) command_buffer: StandardCommandBufferAllocator,
    pub(crate) descriptor_set: StandardDescriptorSetAllocator,
}

impl Allocators {
    pub(crate) fn new(device: Arc<Device>) -> Arc<Self> {
        // TODO: optimize each allocator's settings
        Arc::new(Self {
            memory: StandardMemoryAllocator::new_default(device.clone()),
            command_buffer: StandardCommandBufferAllocator::new(device.clone(), StandardCommandBufferAllocatorCreateInfo::default()),
            descriptor_set: StandardDescriptorSetAllocator::new(device.clone()),
        })
    }
}
