use std::sync::Arc;

use vulkano::{
    command_buffer::allocator::{
        StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::Device,
    memory::allocator::StandardMemoryAllocator,
};

pub struct Allocators {
    pub memory: StandardMemoryAllocator,
    pub command_buffer: StandardCommandBufferAllocator,
    pub descriptor_set: StandardDescriptorSetAllocator,
}

impl Allocators {
    pub fn new(device: Arc<Device>) -> Arc<Self> {
        // TODO: optimize each allocator's settings
        Arc::new(Self {
            memory: StandardMemoryAllocator::new_default(device.clone()),
            command_buffer: StandardCommandBufferAllocator::new(
                device.clone(),
                StandardCommandBufferAllocatorCreateInfo::default(),
            ),
            descriptor_set: StandardDescriptorSetAllocator::new(device.clone()),
        })
    }
}
