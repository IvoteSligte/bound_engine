use std::{f32::consts::PI, sync::Arc};

use event_helper::create_callbacks;
use glam::*;
use images::create_color_image;

use shaders::LM_COUNT;
use vulkano::{
    swapchain::{acquire_next_image, AcquireError, SwapchainPresentInfo},
    sync::{self, FlushError, GpuFuture},
};
use winit::{
    dpi::{PhysicalSize, Size},
    event_loop::{ControlFlow, EventLoop},
    window::{CursorGrabMode, WindowBuilder},
};
use winit_event_helper::*;

use crate::{event_helper::*, swapchain::*};

mod allocators;
mod buffers;
mod command_buffers;
mod descriptor_sets;
mod device;
mod event_helper;
mod fences;
mod images;
mod instance;
mod pipelines;
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

    let mut eh = create_event_helper(window);

    let callbacks = create_callbacks();

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

        let new_position =
            Vec3::from_array(*eh.state.real_time_data.position.as_ref()) + eh.delta_position();

        eh.rotation.y = eh.rotation.y.clamp(-0.5 * PI, 0.5 * PI);
        eh.state.real_time_data.previousRotation = eh.state.real_time_data.rotation;
        eh.state.real_time_data.previousPosition = eh.state.real_time_data.position;
        eh.state.real_time_data.rotation = eh.rotation().to_array();
        eh.state.real_time_data.position = new_position.to_array().into();
        eh.delta_position = Vec3::ZERO;

        let old_pos = IVec3::from_array(*eh.state.real_time_data.lightmapOrigin.as_ref());
        let new_pos = new_position.as_ivec3();

        const SMALLEST_UNIT: f32 = 0.5;
        const LARGEST_UNIT: f32 = (1 << (LM_COUNT - 1)) as f32 * SMALLEST_UNIT;

        let largest_delta_pos = (new_pos - old_pos).as_vec3() / LARGEST_UNIT;

        if largest_delta_pos.abs().cmpge(Vec3::splat(1.0)).any() {
            let delta_pos = largest_delta_pos * LARGEST_UNIT;
            eh.state.real_time_data.lightmapOrigin =
                (old_pos + delta_pos.as_ivec3()).to_array().into();

            for i in 0..(LM_COUNT as usize) {
                let unit_size = (i as f32).exp2() * SMALLEST_UNIT;
                let delta_units = (delta_pos / unit_size).as_ivec3();
                eh.state.real_time_data.deltaLightmapOrigins[i] = delta_units.extend(0).to_array();
            }

            // FIXME: do not redo the rendering of the entire lightmap every time, but only the 'moved' part
            eh.state.command_buffers.pathtraces.restart(); // TODO: combine with move lightmap command buffer
        }

        // rendering
        if eh.recreate_swapchain || eh.window_resized {
            let success = recreate_swapchain(&mut eh);

            if !success {
                return;
            }
        }

        if let Some(previous_future) = eh.state.fences.previous() {
            previous_future.wait(None).unwrap();
            *eh.state.buffers.real_time.write().unwrap() = eh.state.real_time_data;
        }

        let (image_index, suboptimal, image_future) =
            match acquire_next_image(eh.state.swapchain.clone(), None) {
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

        let future = sync::now(eh.state.device.clone()).boxed();

        let future = future
            .then_execute(
                eh.state.queue.clone(),
                eh.state.command_buffers.pathtraces.next(),
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
