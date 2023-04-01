mod bvh;

use std::{f32::consts::PI, sync::Arc};

use buffers::{get_bvh_buffer, get_mutable_buffer, get_blue_noise_buffer};
use descriptor_sets::get_compute_descriptor_sets;
use device::{get_device, select_physical_device};
use event_helper::get_callbacks;
use fps_counter::FPSCounter;
use glam::*;
use images::get_color_image;
use instance::get_instance;
use lightmap::{LightmapBufferSet, LightmapImages};
use pipelines::Pipelines;
use shaders::{Shaders, LIGHTMAP_COUNT};
use vulkano::{
    buffer::{BufferUsage, DeviceLocalBuffer},
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::DeviceExtensions,
    format::Format,
    memory::allocator::StandardMemoryAllocator,
    swapchain::{acquire_next_image, AcquireError, SwapchainPresentInfo},
    sync::{self, FenceSignalFuture, FlushError, GpuFuture},
};
use winit::{
    event_loop::{ControlFlow, EventLoop},
    window::{CursorGrabMode, WindowBuilder},
};
use winit_event_helper::*;

use crate::{command_buffers::*, event_helper::*, swapchain::*};

mod buffers;
mod command_buffers;
mod descriptor_sets;
mod device;
mod event_helper;
mod images;
mod instance;
mod lightmap;
mod pipelines;
mod scene;
mod shaders;
mod swapchain;

// field of view
const FOV: f32 = 1.0;

fn main() {
    let instance = get_instance();

    let event_loop = EventLoop::new();
    let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
    window.set_visible(true);
    window.set_cursor_visible(false);
    window.set_resizable(false);
    let surface = vulkano_win::create_surface_from_winit(window.clone(), instance.clone()).unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };

    let (physical_device, queue_family_index) =
        select_physical_device(instance, &surface, &device_extensions);

    let (device, queue) = get_device(&physical_device, device_extensions, queue_family_index);

    // TODO: improve
    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

    let dimensions = window.inner_size();
    let image_format = physical_device
        .surface_formats(&surface, Default::default())
        .unwrap()
        .iter()
        .max_by_key(|(format, _)| match format {
            Format::R8G8B8A8_SRGB | Format::B8G8R8A8_SRGB => 1,
            _ => 0,
        })
        .unwrap()
        .0;

    let (mut swapchain, swapchain_images) =
        get_swapchain(&device, surface, physical_device, image_format, dimensions);

    let shaders = Shaders::load(device.clone());

    let mut pipelines = Pipelines::from_shaders(device.clone(), shaders.clone(), dimensions.cast());

    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );
    let mut alloc_command_buffer_builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue_family_index,
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    let bvh_buffer = get_bvh_buffer(&memory_allocator, &mut alloc_command_buffer_builder);

    let mutable_buffer = get_mutable_buffer(&memory_allocator, &mut alloc_command_buffer_builder);

    // TODO: abstraction using struct
    let mut real_time_data = shaders::ty::RealTimeBuffer {
        rotation: [0.0; 4],
        previousRotation: [0.0; 4],
        position: [0.0; 3],
        previousPosition: [0.0; 3],
        deltaLightmapOrigins: [[0; 4]; LIGHTMAP_COUNT], // FIXME:
        lightmapOrigin: [0; 3],
        frame: 0,
        _dummy0: [0; 4],
        _dummy1: [0; 4],
        _dummy2: [0; 4],
    };

    let real_time_buffer = DeviceLocalBuffer::from_data(
        &memory_allocator,
        real_time_data,
        BufferUsage {
            uniform_buffer: true,
            transfer_dst: true,
            ..BufferUsage::empty()
        },
        &mut alloc_command_buffer_builder,
    )
    .unwrap();

    let blue_noise_buffer = get_blue_noise_buffer(&memory_allocator, &mut alloc_command_buffer_builder);

    let lightmap_buffers = LightmapBufferSet::new(&memory_allocator, queue_family_index, 2);

    alloc_command_buffer_builder
        .build()
        .unwrap()
        .execute(queue.clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let color_image = get_color_image(&memory_allocator, dimensions, queue_family_index);
    let lightmap_images = LightmapImages::new(&memory_allocator, queue_family_index);

    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
    let descriptor_sets = get_compute_descriptor_sets(
        &descriptor_set_allocator,
        pipelines.clone(),
        real_time_buffer.clone(),
        bvh_buffer.clone(),
        mutable_buffer.clone(),
        color_image.clone(),
        lightmap_images.clone(),
        lightmap_buffers.clone(),
        blue_noise_buffer.clone(),
    );

    let mut command_buffers = CommandBufferCollection::new(
        &command_buffer_allocator,
        &queue,
        &pipelines,
        dimensions,
        descriptor_sets,
        &lightmap_buffers,
        color_image,
        &swapchain_images,
        &lightmap_images,
    );

    let mut lightmap_update = false;

    let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; swapchain_images.len()];
    let mut previous_fence_index = 0;

    let mut eh = get_event_helper(window);

    let callbacks = get_callbacks();

    let mut fps_counter = FPSCounter::new();

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

        println!("{}", fps_counter.tick());

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

        let new_position = Vec3::from_array(real_time_data.position) + eh.position();

        eh.rotation.y = eh.rotation.y.clamp(-0.5 * PI, 0.5 * PI);
        real_time_data.previousRotation = real_time_data.rotation;
        real_time_data.previousPosition = real_time_data.position;
        real_time_data.rotation = eh.rotation().to_array();
        real_time_data.position = new_position.to_array();
        eh.delta_position = Vec3::ZERO;

        real_time_data.frame += 1;

        let old_pos = IVec3::from_array(real_time_data.lightmapOrigin);
        let new_pos = new_position.as_ivec3();

        let delta_position = new_pos - old_pos;

        const LARGEST_UNIT: i32 = 1 << (LIGHTMAP_COUNT - 1);
        if delta_position.abs().cmpge(IVec3::splat(LARGEST_UNIT)).any() {
            lightmap_update = true;
            for i in 0..LIGHTMAP_COUNT {
                real_time_data.deltaLightmapOrigins[i] =
                    (delta_position / (1 << i)).extend(0).to_array();
            }
        }

        // rendering
        if eh.recreate_swapchain || eh.window_resized {
            let success = recreate_swapchain(
                &mut eh,
                &mut swapchain,
                &command_buffer_allocator,
                queue.clone(),
                fences.clone(),
                previous_fence_index,
                device.clone(),
                &mut pipelines,
                shaders.clone(),
                &memory_allocator,
                queue_family_index,
                &descriptor_set_allocator,
                real_time_buffer.clone(),
                bvh_buffer.clone(),
                mutable_buffer.clone(),
                lightmap_images.clone(),
                lightmap_buffers.clone(),
                blue_noise_buffer.clone(),
                &mut command_buffers,
            );

            if !success {
                return;
            }
        }

        let (image_index, suboptimal, image_future) =
            match acquire_next_image(swapchain.clone(), None) {
                Ok(ok) => ok,
                Err(AcquireError::OutOfDate) => {
                    return eh.recreate_swapchain = true;
                }
                Err(err) => panic!("{}", err),
            };
        eh.recreate_swapchain |= suboptimal;

        let previous_future = match fences[previous_fence_index].clone() {
            Some(future) => future.boxed(),
            None => {
                let mut future = sync::now(device.clone());
                future.cleanup_finished();
                future.boxed()
            }
        };

        if let Some(image_fence) = &fences[image_index as usize] {
            image_fence.wait(None).unwrap();
        }

        let real_time_command_buffer = get_real_time_command_buffer(
            &command_buffer_allocator,
            queue_family_index,
            real_time_data,
            real_time_buffer.clone(),
        );

        let mut future = previous_future;

        if lightmap_update {
            lightmap_update = false;

            future = future
                .then_execute(queue.clone(), command_buffers.move_lightmap.clone())
                .unwrap()
                .boxed();
        }

        future = future
            .then_execute(queue.clone(), real_time_command_buffer)
            .unwrap()
            .boxed();

        let future = future
            .then_execute(
                queue.clone(),
                command_buffers.pathtraces.next().unwrap().clone(),
            )
            .unwrap()
            .then_execute(
                queue.clone(),
                command_buffers.swapchains[image_index as usize].clone(),
            )
            .unwrap()
            .join(image_future)
            .then_swapchain_present(
                queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush();

        // TODO: remove pointless double/triple buffering (in general, not just here)
        fences[image_index as usize] = match future {
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
        previous_fence_index = image_index as usize;

        // DEBUG, TODO: remove
        fences[image_index as usize]
            .clone()
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    })
}

// FIXME: BUG |||| Os(OsError { line: 1333, file: "{DIR}/winit-0.27.5/src/platform_impl/linux/x11/window.rs", error: XMisc("Cursor could not be confined: already confined by another client") })', src/main.rs:649:65

// TODO: sort things into different files and collection structs
