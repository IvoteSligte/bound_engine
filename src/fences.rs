use std::{sync::Arc, ops::{Deref, DerefMut}};

use vulkano::{sync::{FenceSignalFuture, GpuFuture, JoinFuture}, swapchain::{SwapchainAcquireFuture, PresentFuture}, command_buffer::CommandBufferExecFuture};

type NestedFence = FenceSignalFuture<PresentFuture<JoinFuture<CommandBufferExecFuture<CommandBufferExecFuture<Box<dyn GpuFuture>>>, SwapchainAcquireFuture>>>;

#[derive(Clone)]
pub(crate) struct Fences {
    inner: Vec<Option<Arc<NestedFence>>>,
    previous_index: usize,
}

impl Fences {
    pub(crate) fn new(count: usize) -> Self {
        Self {
            inner: vec![None; count],
            previous_index: 0,
        }
    }

    pub(crate) fn previous(&self) -> Option<Arc<NestedFence>> {
        self.inner[self.previous_index].clone()
    }

    pub(crate) fn set_previous(&mut self, previous_index: usize) {
        self.previous_index = previous_index;
    }
}

impl Deref for Fences {
    type Target = Vec<Option<Arc<NestedFence>>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for Fences {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}