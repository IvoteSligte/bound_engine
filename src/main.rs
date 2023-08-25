use glam::*;
use std::{f32::consts::PI, sync::Arc};

use shaders::LM_LAYERS;
use vulkano::{
    swapchain::{AcquireError, SwapchainPresentInfo},
    sync::{self, FlushError, GpuFuture},
};
use winit::{
    dpi::{PhysicalSize, Size},
    event_loop::{ControlFlow, EventLoop},
    window::{CursorGrabMode, WindowBuilder},
};
use winit_event_helper::KeyCode;

mod allocator;
mod buffer;
mod command_buffer;
mod descriptor_sets;
mod device;
mod event_helper;
mod fences;
mod image;
mod instance;
mod pipeline;
mod render_pass;
mod scene;
mod shaders;
mod state;
mod swapchain;

// field of view
const FOV: f32 = 1.0;

fn main() {
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

    let mut eh = event_helper::create(window);

    let callbacks = event_helper::callbacks();

    event_loop.run(move |event, _, control_flow| {
        if eh.quit {
            *control_flow = ControlFlow::Exit;
        }

        if !eh.update(&callbacks, &event) || eh.window_frozen {
            return;
        }

        if eh.update_count() == 0 {
            eh.window.set_cursor_grab(CursorGrabMode::Confined).unwrap();
        }

        println!("{}", eh.fps_counter.tick());

        let cursor_mov = eh.cursor_delta / eh.dimensions().x * eh.rotation_multiplier;
        eh.rotation += cursor_mov * Vec2::new(1.0, -1.0);

        eh.cursor_delta = Vec2::ZERO;

        let inputs = eh.data.window.inputs.clone();
        let delta_time = eh.time_since_previous_step().as_secs_f32();
        let delta_rot = delta_time * eh.rotation_multiplier;
        let delta_mov = delta_time * eh.movement_multiplier;

        if inputs.pressed(KeyCode::Left) {
            eh.rotation.x -= delta_rot;
        }
        if inputs.pressed(KeyCode::Right) {
            eh.rotation.x += delta_rot;
        }
        if inputs.pressed(KeyCode::Up) {
            eh.rotation.y += delta_rot;
        }
        if inputs.pressed(KeyCode::Down) {
            eh.rotation.y -= delta_rot;
        }

        if inputs.pressed(KeyCode::A) {
            eh.delta_position.x -= delta_mov;
        }
        if inputs.pressed(KeyCode::D) {
            eh.delta_position.x += delta_mov;
        }
        if inputs.pressed(KeyCode::W) {
            eh.delta_position.y += delta_mov;
        }
        if inputs.pressed(KeyCode::S) {
            eh.delta_position.y -= delta_mov;
        }
        if inputs.pressed(KeyCode::Q) {
            eh.delta_position.z -= delta_mov;
        }
        if inputs.pressed(KeyCode::E) {
            eh.delta_position.z += delta_mov;
        }

        eh.rotation.y = eh.rotation.y.clamp(-0.5 * PI, 0.5 * PI);

        let position = Vec3::from_array(eh.state.real_time_data.position) + eh.delta_position();

        eh.state.real_time_data.position = position.to_array().into();
        eh.delta_position = Vec3::ZERO;
        eh.state.real_time_data.projection_view = state::projection_view_matrix(
            position,
            eh.rotation(),
            Vec2::from_array(eh.window.inner_size().into()),
        )
        .to_cols_array_2d();

        // rendering
        if eh.recreate_swapchain || eh.window_resized {
            if !swapchain::recreate(&mut eh) {
                return;
            }
        }

        if let Some(previous_future) = eh.state.fences.previous() {
            previous_future.wait(None).unwrap();
        }

        eh.frame_counter += 1;

        *eh.state.buffers.real_time.write().unwrap() = eh.state.real_time_data;

        let (image_index, suboptimal, image_future) =
            match vulkano::swapchain::acquire_next_image(eh.state.swapchain.clone(), None) {
                Ok(ok) => ok,
                Err(AcquireError::OutOfDate) => {
                    return eh.recreate_swapchain = true;
                }
                Err(err) => panic!("{}", err),
            };
        eh.recreate_swapchain |= suboptimal;

        if let Some(image_fence) = &eh.state.fences[image_index as usize] {
            image_fence.wait(None).unwrap();
        }

        let future = sync::now(eh.state.device.clone())
            .then_execute(
                eh.state.queue.clone(),
                eh.state.command_buffers.pathtraces.next(),
            )
            .unwrap()
            .then_execute(
                eh.state.queue.clone(),
                eh.state.command_buffers.pathtraces.direct.clone(),
            )
            .unwrap()
            .then_execute(
                eh.state.queue.clone(),
                eh.state.command_buffers.swapchains[image_index as usize].clone(),
            )
            .unwrap()
            .join(image_future)
            .then_swapchain_present(
                eh.state.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(
                    eh.state.swapchain.clone(),
                    image_index,
                ),
            )
            .boxed()
            .then_signal_fence_and_flush();

        eh.state.fences[image_index as usize] = match future {
            Ok(ok) => Some(Arc::new(ok)),
            Err(FlushError::OutOfDate) => {
                eh.recreate_swapchain = true;
                None
            }
            Err(err) => {
                eprintln!("{}", err);
                None
            }
        };
        eh.state.fences.set_previous(image_index as usize);
    })
}

// FIXME: BUG |||| Os(OsError { line: 1333, file: "{DIR}/winit-0.27.5/src/platform_impl/linux/x11/window.rs", error: XMisc("Cursor could not be confined: already confined by another client") })', src/main.rs:649:65

// TODO: sort things into different files and collection structs
