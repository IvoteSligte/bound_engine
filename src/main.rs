use std::{sync::Arc};

use wgpu::*;
use fps_counter::FPSCounter;
use glam::*;
use winit::{
    dpi::{PhysicalSize, Size},
    event_loop::{EventLoop},
    window::{WindowBuilder},
};

// field of view
const FOV: f32 = 1.0;

fn main() {
    //let instance = Instance::new(InstanceDescriptor::);

    let event_loop = EventLoop::new();
    let window = Arc::new(
        WindowBuilder::new()
            .with_maximized(true)
            .with_decorations(false)
            .with_visible(true)
            .with_resizable(false)
            .with_inner_size(Size::Physical(PhysicalSize::new(1920, 1080))) // TODO: support for other resolutions
            .build(&event_loop)
            .unwrap(),
    );
    window.set_cursor_visible(false);

    let mut fps_counter = FPSCounter::new();

    event_loop.run(move |event, _, control_flow| {
        
    })
}
