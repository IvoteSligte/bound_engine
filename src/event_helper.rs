use std::sync::Arc;

use fps_counter::FPSCounter;
use glam::*;
use winit::window::{CursorGrabMode, Fullscreen, Window};
use winit_event_helper::{Callbacks, EventHelper, KeyCode};

use crate::state::State;

mod rotation {
    use glam::Vec3;

    pub const UP: Vec3 = Vec3::new(0.0, 0.0, 1.0);
    pub const FORWARD: Vec3 = Vec3::new(0.0, 1.0, 0.0);
    pub const RIGHT: Vec3 = Vec3::new(1.0, 0.0, 0.0);
}

pub fn create(window: Arc<Window>) -> EventHelper<Data> {
    EventHelper::new(Data {
        state: State::new(window.clone()),
        window,
        window_frozen: false,
        window_resized: false,
        recreate_swapchain: false,
        cursor_delta: Vec2::ZERO,
        delta_position: Vec3::ZERO,
        rotation: Vec2::ZERO,
        quit: false,
        movement_multiplier: 25.0,
        rotation_multiplier: 1.0,
        fps_counter: FPSCounter::new(),
        frame_counter: 0,
    })
}

pub struct Data {
    pub state: State,
    pub window: Arc<Window>,
    pub window_frozen: bool,
    pub window_resized: bool,
    pub recreate_swapchain: bool,
    /// change in cursor position
    pub cursor_delta: Vec2,
    /// change in position relative to the rotation axes
    pub delta_position: Vec3,
    /// absolute rotation around the x and z axes
    pub rotation: Vec2,
    pub quit: bool,
    pub movement_multiplier: f32,
    pub rotation_multiplier: f32,
    pub fps_counter: FPSCounter,
    pub frame_counter: u64,
}

impl Data {
    pub fn rotation(&self) -> Quat {
        Quat::from_rotation_z(-self.rotation.x) * Quat::from_rotation_x(self.rotation.y)
    }

    pub fn delta_position(&self) -> Vec3 {
        let rotation = self.rotation();

        let right = rotation.mul_vec3(rotation::RIGHT);
        let forward = rotation.mul_vec3(rotation::FORWARD);
        let up = rotation.mul_vec3(rotation::UP);

        self.delta_position.x * right + self.delta_position.y * forward + self.delta_position.z * up
    }

    pub fn dimensions(&self) -> Vec2 {
        Vec2::from_array(self.window.inner_size().into())
    }
}

pub fn callbacks() -> Callbacks<Data> {
    let mut callbacks = Callbacks::<Data>::default();

    callbacks.window.quit(|eh, _| eh.quit = true);
    callbacks.window.inputs.just_pressed(KeyCode::Escape, |eh| {
        eh.quit = true;
    });

    callbacks
        .device
        .mouse_motion(|eh, (dx, dy)| eh.cursor_delta += DVec2::new(dx, dy).as_vec2());

    callbacks.window.focused(|eh, focused| {
        eh.window_frozen = !focused;
        if focused {
            eh.window.set_cursor_grab(CursorGrabMode::Confined).unwrap();
        } else {
            eh.window.set_cursor_grab(CursorGrabMode::None).unwrap();
        }
    });

    callbacks.window.resized(|eh, mut size| {
        eh.window_frozen = size.width == 0 || size.height == 0;
        eh.window_resized = true;

        if size.width < size.height {
            size.height = size.width;
            eh.window.set_inner_size(size);
            eh.state.real_time_data.screenSize = [size.width as f32, size.height as f32];
        }
    });

    callbacks
        .window
        .inputs
        .just_pressed(KeyCode::F11, |eh| match eh.window.fullscreen() {
            Some(_) => eh.window.set_fullscreen(None),
            None => eh.window.set_fullscreen(Some(Fullscreen::Borderless(None))),
        });

    // DEBUG
    callbacks.window.inputs.just_pressed(KeyCode::Equals, |eh| {
        if eh.data.window.inputs.pressed(KeyCode::RAlt) {
            eh.rotation_multiplier *= 2.0;
            println!("{}", eh.rotation_multiplier);
        } else {
            eh.movement_multiplier *= 2.0;
            println!("{}", eh.movement_multiplier);
        }
    });

    // DEBUG
    callbacks.window.inputs.just_pressed(KeyCode::Minus, |eh| {
        if eh.data.window.inputs.pressed(KeyCode::RAlt) {
            eh.rotation_multiplier /= 2.0;
            println!("{}", eh.rotation_multiplier);
        } else {
            eh.movement_multiplier /= 2.0;
            println!("{}", eh.movement_multiplier);
        }
    });
    callbacks
}
