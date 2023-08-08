use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

use vulkano::sync::{future::FenceSignalFuture, GpuFuture};

type NestedFence = FenceSignalFuture<Box<dyn GpuFuture>>;

#[derive(Clone)]
pub struct Fences {
    inner: Vec<Option<Arc<NestedFence>>>,
    previous_index: usize,
}

impl Fences {
    pub fn new(count: usize) -> Self {
        Self {
            inner: vec![None; count],
            previous_index: 0,
        }
    }

    pub fn previous(&self) -> Option<Arc<NestedFence>> {
        self.inner[self.previous_index].clone()
    }

    pub fn set_previous(&mut self, previous_index: usize) {
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
