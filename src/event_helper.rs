use std::sync::Arc;

use glam::*;
use winit::window::{CursorGrabMode, Window};
use winit_event_helper::{Callbacks, EventHelper, KeyCode};
use winit_fullscreen::WindowFullScreen;

mod rotation {
    use glam::Vec3;

    pub const UP: Vec3 = Vec3::new(0.0, 0.0, 1.0);
    pub const FORWARD: Vec3 = Vec3::new(0.0, 1.0, 0.0);
    pub const RIGHT: Vec3 = Vec3::new(1.0, 0.0, 0.0);
}

pub(crate) fn get_event_helper(window: Arc<Window>, dimensions: Vec2) -> EventHelper<Data> {
    EventHelper::new(Data {
        window: window,
        window_frozen: false,
        window_resized: false,
        recreate_swapchain: false,
        dimensions,
        cursor_delta: Vec2::ZERO,
        delta_position: Vec3::ZERO,
        rotation: Vec2::ZERO,
        quit: false,
        movement_multiplier: 25.0,
        rotation_multiplier: 1.0,
    })
}

pub(crate) struct Data {
    pub(crate) window: Arc<Window>,
    pub(crate) window_frozen: bool,
    pub(crate) window_resized: bool,
    pub(crate) recreate_swapchain: bool,
    /// viewport dimensions
    pub(crate) dimensions: Vec2,
    /// change in cursor position
    pub(crate) cursor_delta: Vec2,
    /// change in position relative to the rotation axes
    pub(crate) delta_position: Vec3,
    /// absolute rotation around the x and z axes
    pub(crate) rotation: Vec2,
    pub(crate) quit: bool,
    pub(crate) movement_multiplier: f32,
    pub(crate) rotation_multiplier: f32,
}

impl Data {
    pub(crate) fn rotation(&self) -> Quat {
        Quat::from_rotation_z(-self.rotation.x) * Quat::from_rotation_x(self.rotation.y)
    }

    pub(crate) fn position(&self) -> Vec3 {
        let rotation = self.rotation();

        let right = rotation.mul_vec3(rotation::RIGHT);
        let forward = rotation.mul_vec3(rotation::FORWARD);
        let up = rotation.mul_vec3(rotation::UP);

        self.delta_position.x * right + self.delta_position.y * forward + self.delta_position.z * up
    }
}

pub(crate) fn get_callbacks() -> Callbacks<Data> {
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
        }

        eh.dimensions = UVec2::new(size.width, size.height).as_vec2();
    });

    callbacks
        .window
        .inputs
        .just_pressed(KeyCode::F11, |eh| eh.window.toggle_fullscreen());

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
