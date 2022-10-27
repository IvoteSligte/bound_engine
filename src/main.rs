use std::{f32::consts::PI, sync::Arc};

use fps_counter::FPSCounter;
use glam::{DVec2, Quat, UVec2, Vec2, Vec3, UVec3};
use vulkano::{
    buffer::{BufferAccess, BufferUsage, DeviceLocalBuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyImageInfo,
        PrimaryAutoCommandBuffer, ClearColorImageInfo,
    },
    descriptor_set::{DescriptorSetsCollection, PersistentDescriptorSet, WriteDescriptorSet},
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo,
    },
    format::Format,
    image::{
        view::ImageView, ImageAccess, ImageCreateFlags, ImageDimensions, ImageUsage, StorageImage,
        SwapchainImage,
    },
    instance::{Instance, InstanceCreateInfo},
    pipeline::{graphics::viewport::Viewport, ComputePipeline, Pipeline, PipelineBindPoint},
    shader::ShaderModule,
    swapchain::{
        self, AcquireError, PresentInfo, Surface, Swapchain, SwapchainCreateInfo,
        SwapchainCreationError,
    },
    sync::{self, FenceSignalFuture, FlushError, GpuFuture, PipelineStage},
    Version, VulkanLibrary,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    dpi::PhysicalSize,
    event::VirtualKeyCode,
    event_loop::{ControlFlow, EventLoop},
    window::{CursorGrabMode, Fullscreen, Window, WindowBuilder},
};
use winit_event_helper::{EventHelper, ElementState2};

mod shaders {
    vulkano_shaders::shader! {
        shaders: {
            Compute: {
                ty: "compute",
                path: "shaders/compute.glsl",
            }
        },
        types_meta: { #[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)] },
        include: ["shaders/random.glsl"],
    }
}

fn select_physical_device<'a, W>(
    instance: &'a Arc<Instance>,
    surface: &'a Surface<W>,
    device_extensions: &'a DeviceExtensions,
) -> (Arc<PhysicalDevice>, u32) {
    instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .position(|q| {
                    q.supports_stage(PipelineStage::FragmentShader)
                        && q.supports_stage(PipelineStage::VertexShader)
                })
                .map(|q| (p, q as u32))
                .filter(|(p, q)| p.surface_support(*q, surface).unwrap_or(false))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 4,
            PhysicalDeviceType::IntegratedGpu => 3,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 1,
            _ => 0,
        })
        .unwrap()
}

fn get_compute_pipeline(
    device: Arc<Device>,
    compute_shader: Arc<ShaderModule>,
) -> Arc<ComputePipeline> {
    ComputePipeline::new(
        device.clone(),
        compute_shader.entry_point("main").unwrap(),
        &(),
        None,
        |_| {},
    )
    .unwrap()
}

fn get_intermediate_image(
    device: Arc<Device>,
    dimensions: [u32; 2],
    queue_family_index: u32,
) -> Arc<StorageImage> {
    StorageImage::with_usage(
        device.clone(),
        ImageDimensions::Dim2d {
            width: dimensions[0],
            height: dimensions[1],
            array_layers: 1,
        },
        Format::R8G8B8A8_UNORM,
        ImageUsage {
            storage: true,
            transfer_src: true,
            transfer_dst: true,
            ..ImageUsage::empty()
        },
        ImageCreateFlags::empty(),
        [queue_family_index],
    )
    .unwrap()
}

fn get_temporal_images(
    device: Arc<Device>,
    dimensions: PhysicalSize<u32>,
    queue_family_index: u32,
) -> (Arc<StorageImage>, Arc<StorageImage>) {
    let accumulator_image_read = StorageImage::with_usage(
        device.clone(),
        ImageDimensions::Dim2d {
            width: dimensions.width,
            height: dimensions.height,
            array_layers: 1,
        },
        Format::R16G16B16A16_SFLOAT, // TODO: loosely match format with swapchain image format
        ImageUsage {
            storage: true,
            transfer_dst: true,
            ..ImageUsage::empty()
        },
        ImageCreateFlags::empty(),
        [queue_family_index],
    )
    .unwrap();

    let accumulator_image_write = StorageImage::with_usage(
        device.clone(),
        accumulator_image_read.dimensions(),
        accumulator_image_read.format(), // TODO: loosely match format with swapchain image format
        ImageUsage {
            storage: true,
            transfer_src: true,
            ..ImageUsage::empty()
        },
        ImageCreateFlags::empty(),
        [queue_family_index],
    )
    .unwrap();
    
    (accumulator_image_read, accumulator_image_write)
}

fn get_compute_descriptor_set(
    pipeline: Arc<ComputePipeline>,
    mutable_buffer: Arc<dyn BufferAccess>,
    constant_buffer: Arc<dyn BufferAccess>,
    render_image: Arc<dyn ImageAccess>,
    accumulator_images: (Arc<StorageImage>, Arc<StorageImage>),
) -> Arc<PersistentDescriptorSet> {
    let render_image_view = ImageView::new_default(render_image.clone()).unwrap();
    let accumulator_image_view_read = ImageView::new_default(accumulator_images.0).unwrap();
    let accumulator_image_view_write = ImageView::new_default(accumulator_images.1).unwrap();

    PersistentDescriptorSet::new(
        pipeline.layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::buffer(0, mutable_buffer),
            WriteDescriptorSet::buffer(1, constant_buffer),
            WriteDescriptorSet::image_view(2, render_image_view),
            WriteDescriptorSet::image_view(3, accumulator_image_view_read),
            WriteDescriptorSet::image_view(4, accumulator_image_view_write),
        ],
    )
    .unwrap()
}

fn get_command_buffer<S>(
    device: Arc<Device>,
    queue: Arc<Queue>,
    compute_pipeline: Arc<ComputePipeline>,
    push_constants: shaders::ty::PushConstantData,
    compute_descriptor_set: S,
    intermediate_image: Arc<dyn ImageAccess>,
    swapchain_image: Arc<SwapchainImage<Window>>,
    accumulator_images: (Arc<StorageImage>, Arc<StorageImage>),
) -> Arc<PrimaryAutoCommandBuffer>
where
    S: DescriptorSetsCollection + Clone,
{
    let mut builder = AutoCommandBufferBuilder::primary(
        device.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    builder
        .clear_color_image(ClearColorImageInfo::image(intermediate_image.clone()))
        .unwrap()
        .bind_pipeline_compute(compute_pipeline.clone())
        .push_constants(compute_pipeline.layout().clone(), 0, push_constants)
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            0,
            compute_descriptor_set.clone(),
        )
        .dispatch((UVec3::from_array(swapchain_image.dimensions().width_height_depth()).as_vec3() / 8.0).ceil().as_uvec3().to_array())
        .unwrap()
        .copy_image(CopyImageInfo::images(
            intermediate_image.clone(),
            swapchain_image.clone(),
        ))
        .unwrap()
        .copy_image(CopyImageInfo::images(accumulator_images.1, accumulator_images.0))
        .unwrap();

    Arc::new(builder.build().unwrap())
}

#[allow(dead_code)]
mod rotation {
    use glam::Vec3;

    pub const UP: Vec3 = Vec3::new(0.0, 0.0, 1.0);
    pub const FORWARD: Vec3 = Vec3::new(0.0, 1.0, 0.0);
    pub const RIGHT: Vec3 = Vec3::new(1.0, 0.0, 0.0);
}

// field of view
const FOV: f32 = 1.0;

struct Data<W> {
    /// the window surface
    surface: Arc<Surface<W>>,
    window_frozen: bool,
    window_resized: bool,
    recreate_swapchain: bool,
    /// viewport dimensions
    dimensions: Vec2,
    /// change in cursor position
    cursor_delta: Vec2,
    /// change in position relative to the rotation axes
    delta_position: Vec3,
    /// absolute rotation around the x and z axes
    rotation: Vec2,
    quit: bool,
    speed_multiplier: f32,
    rotation_multiplier: f32,
}

impl<W> Data<W> {
    fn window(&self) -> &W {
        self.surface.window()
    }

    fn rotation(&self) -> Quat {
        Quat::from_rotation_z(-self.rotation.x) * Quat::from_rotation_x(self.rotation.y)
    }

    fn position(&self) -> Vec3 {
        let rotation = self.rotation();

        let right = rotation.mul_vec3(rotation::RIGHT);
        let forward = rotation.mul_vec3(rotation::FORWARD);
        let up = rotation.mul_vec3(rotation::UP);

        self.delta_position.x * right + self.delta_position.y * forward + self.delta_position.z * up
    }
}

fn main() {
    let library = VulkanLibrary::new().unwrap();
    let required_extensions = vulkano_win::required_extensions(&library);
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            engine_version: Version::V1_3,
            ..Default::default()
        },
    )
    .unwrap();

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();
    // BUG: spamming any key on application startup will make the window invisible
    surface.window().set_cursor_visible(false);
    surface
        .window()
        // WARNING: not supported on Mac, web and mobile platforms
        .set_cursor_grab(CursorGrabMode::Confined)
        .unwrap();
    surface.window().set_resizable(false);

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };

    let (physical, queue_family_index) =
        select_physical_device(&instance, &surface, &device_extensions);

    let (device, mut queues) = Device::new(
        physical.clone(),
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..QueueCreateInfo::default()
            }],
            enabled_extensions: device_extensions,
            ..Default::default()
        },
    )
    .unwrap();
    let queue = queues.next().unwrap();

    let capabilities = physical
        .surface_capabilities(&surface, Default::default())
        .unwrap();
    let dimensions = surface.window().inner_size();
    let composite_alpha = capabilities
        .supported_composite_alpha
        .iter()
        .next()
        .unwrap();
    let image_format = physical
        .surface_formats(&surface, Default::default())
        .unwrap()
        .iter()
        .max_by_key(|(format, _)| match format {
            Format::R8G8B8_UNORM
            | Format::B8G8R8_UNORM
            | Format::R8G8B8A8_UNORM
            | Format::B8G8R8A8_UNORM => 1,
            _ => 0,
        })
        .unwrap()
        .0;

    let (mut swapchain, mut swapchain_images) = Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: capabilities.min_image_count + 1,
            image_format: Some(image_format),
            image_extent: dimensions.into(),
            image_usage: ImageUsage {
                storage: true,
                transfer_dst: true,
                ..ImageUsage::empty()
            },
            composite_alpha,
            ..Default::default()
        },
    )
    .unwrap();

    let compute_shader = shaders::load_Compute(device.clone()).unwrap();

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: surface.window().inner_size().into(),
        depth_range: 0.0..1.0,
    };

    let mut compute_pipeline = get_compute_pipeline(device.clone(), compute_shader.clone());

    let materials = [
        shaders::ty::Material {
            reflectance: [0.7, 0.7, 0.9],
            emittance: [0.0; 3],
            _dummy0: [0u8; 4],
            _dummy1: [0u8; 4],
        },
        shaders::ty::Material {
            reflectance: [0.1, 0.9, 0.5],
            emittance: [0.0; 3],
            _dummy0: [0u8; 4],
            _dummy1: [0u8; 4],
        },
        shaders::ty::Material {
            reflectance: [0.9, 0.9, 0.1],
            emittance: [0.0; 3],
            _dummy0: [0u8; 4],
            _dummy1: [0u8; 4],
        },
        shaders::ty::Material {
            reflectance: [0.9, 0.1, 0.1],
            emittance: [0.0; 3],
            _dummy0: [0u8; 4],
            _dummy1: [0u8; 4],
        },
        shaders::ty::Material {
            reflectance: [0.0; 3],
            emittance: [10.0; 3],
            _dummy0: [0u8; 4],
            _dummy1: [0u8; 4],
        },
    ];

    let objects = [
        shaders::ty::Object {
            pos: [5.0, 8.0, 2.0],
            sizeSquared: 0.8 * 0.8,
        },
        shaders::ty::Object {
            pos: [10.0, 3.0, 1.0],
            sizeSquared: 6.0 * 6.0,
        },
        shaders::ty::Object {
            pos: [-3.0, 2.0, -4.0],
            sizeSquared: 4.0 * 4.0,
        },
        shaders::ty::Object {
            pos: [3.0, 1.0, 0.0],
            sizeSquared: 3.0 * 3.0,
        },
        shaders::ty::Object {
            pos: [20.0, 20.0, 20.0],
            sizeSquared: 5.0 * 5.0,
        },
    ];

    let mut mutable_data = shaders::ty::MutableData {
        matCount: materials.len() as u32,
        objCount: objects.len() as u32,
        ..Default::default()
    };
    mutable_data.mats[..materials.len()].copy_from_slice(&materials);
    mutable_data.objs[..objects.len()].copy_from_slice(&objects);

    let (mutable_buffer, mutable_future) = DeviceLocalBuffer::from_data(
        mutable_data,
        BufferUsage {
            uniform_buffer: true,
            ..BufferUsage::empty()
        },
        queue.clone(),
    )
    .unwrap();

    // near constant data
    let constant_data = shaders::ty::ConstantBuffer {
        view: viewport.dimensions,
        ratio: [FOV, -FOV * viewport.dimensions[1] / viewport.dimensions[0]],
    };

    let (constant_buffer, constant_future) = DeviceLocalBuffer::from_data(
        constant_data,
        BufferUsage {
            uniform_buffer: true,
            ..BufferUsage::empty()
        },
        queue.clone(),
    )
    .unwrap();

    mutable_future
        .join(constant_future)
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let mut push_constants = shaders::ty::PushConstantData {
        rot: [0.0; 4],
        pos: [0.0; 3],
        time: 0.0,
        ipRot: [0.0; 4],
        pPos: [0.0; 3],
    };

    let mut intermediate_image = get_intermediate_image(
        device.clone(),
        swapchain_images[0].dimensions().width_height(),
        queue_family_index,
    );

    let mut accumulator_images =
        get_temporal_images(device.clone(), dimensions, queue_family_index);

    let mut compute_descriptor_set = get_compute_descriptor_set(
        compute_pipeline.clone(),
        mutable_buffer.clone(),
        constant_buffer.clone(),
        intermediate_image.clone(),
        accumulator_images.clone(),
    );

    let mut command_buffers = vec![None; swapchain_images.len()];

    let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; swapchain_images.len()];
    let mut previous_fence_index = 0;

    let mut eh = EventHelper::new(Data {
        surface,
        window_frozen: false,
        window_resized: false,
        recreate_swapchain: false,
        dimensions: Vec2::from_array(viewport.dimensions),
        cursor_delta: Vec2::ZERO,
        delta_position: Vec3::ZERO,
        rotation: Vec2::ZERO,
        quit: false,
        speed_multiplier: 25.0,
        rotation_multiplier: 1.0,
    });

    let exit = |data: &mut EventHelper<Data<_>>| data.quit = true;
    eh.window_close_requested(exit);
    eh.window_keyboard_input(VirtualKeyCode::Escape, ElementState2::Pressed, exit);

    eh.device_mouse_motion(|data, (dx, dy)| data.cursor_delta += DVec2::new(dx, dy).as_vec2());

    eh.window_focused(|data, focused| {
        data.window_frozen = !focused;
        let window = data.window();
        if focused {
            window.set_cursor_grab(CursorGrabMode::Confined).unwrap();
        } else {
            window.set_cursor_grab(CursorGrabMode::None).unwrap();
        }
    });

    eh.window_resized(|data, mut size| {
        data.window_frozen = size.width == 0 || size.height == 0;
        data.window_resized = true;

        if size.width < size.height {
            size.height = size.width;
            data.window().set_inner_size(size);
        }

        data.dimensions = UVec2::new(size.width, size.height).as_vec2();
    });

    eh.window_keyboard_input(VirtualKeyCode::F11, ElementState2::Pressed, |data| {
        let window = data.window();
        match window.fullscreen() {
            Some(_) => window.set_fullscreen(None),
            None => window.set_fullscreen(Some(Fullscreen::Borderless(None))),
        }
    });

    // DEBUG
    eh.window_keyboard_input(VirtualKeyCode::Equals, ElementState2::Held, |data| {
        if data.key_held(VirtualKeyCode::RAlt).is_some() {
            data.rotation_multiplier *= 2.0;
        } else {
            data.speed_multiplier *= 2.0;
        }
        println!("{}", data.speed_multiplier);
    });
    // DEBUG
    eh.window_keyboard_input(VirtualKeyCode::Minus, ElementState2::Held, |data| {
        if data.key_held(VirtualKeyCode::RAlt).is_some() {
            data.rotation_multiplier /= 2.0;
        } else {
            data.speed_multiplier /= 2.0;
        }
        println!("{}", data.speed_multiplier);
    });

    let mut fps_counter = FPSCounter::new();

    // TODO: remove image data on swapchain recreation

    event_loop.run(move |event, _, control_flow| {
        if eh.quit {
            *control_flow = ControlFlow::Exit;
        }

        if !eh.update(&event) || eh.window_frozen {
            return;
        }

        println!("{}", fps_counter.tick());

        let cursor_mov = eh.cursor_delta / eh.dimensions.x * eh.rotation_multiplier;
        eh.rotation += cursor_mov * Vec2::new(1.0, -1.0);

        eh.cursor_delta = Vec2::ZERO;

        // TODO: make movement independent of framerate

        if eh.key_held(VirtualKeyCode::Left).is_some() {
            eh.rotation.x -= eh.secs_since_last_update() as f32 * eh.rotation_multiplier;
        }
        if eh.key_held(VirtualKeyCode::Right).is_some() {
            eh.rotation.x += eh.secs_since_last_update() as f32 * eh.rotation_multiplier;
        }
        if eh.key_held(VirtualKeyCode::Up).is_some() {
            eh.rotation.y += eh.secs_since_last_update() as f32 * eh.rotation_multiplier;
        }
        if eh.key_held(VirtualKeyCode::Down).is_some() {
            eh.rotation.y -= eh.secs_since_last_update() as f32 * eh.rotation_multiplier;
        }

        if eh.key_held(VirtualKeyCode::A).is_some() {
            eh.delta_position.x -= eh.secs_since_last_update() as f32 * eh.speed_multiplier;
        }
        if eh.key_held(VirtualKeyCode::D).is_some() {
            eh.delta_position.x += eh.secs_since_last_update() as f32 * eh.speed_multiplier;
        }
        if eh.key_held(VirtualKeyCode::W).is_some() {
            eh.delta_position.y += eh.secs_since_last_update() as f32 * eh.speed_multiplier;
        }
        if eh.key_held(VirtualKeyCode::S).is_some() {
            eh.delta_position.y -= eh.secs_since_last_update() as f32 * eh.speed_multiplier;
        }
        if eh.key_held(VirtualKeyCode::Q).is_some() {
            eh.delta_position.z -= eh.secs_since_last_update() as f32 * eh.speed_multiplier;
        }
        if eh.key_held(VirtualKeyCode::E).is_some() {
            eh.delta_position.z += eh.secs_since_last_update() as f32 * eh.speed_multiplier;
        }

        eh.rotation.y = eh.rotation.y.clamp(-0.5 * PI, 0.5 * PI);
        push_constants.ipRot = Quat::from_array(push_constants.rot).conjugate().to_array();
        push_constants.pPos = push_constants.pos;
        push_constants.rot = eh.rotation().to_array();
        push_constants.pos = (Vec3::from(push_constants.pos) + eh.position()).to_array();
        eh.delta_position = Vec3::ZERO;

        push_constants.time = eh.secs_since_start() as f32;

        // rendering
        if eh.recreate_swapchain || eh.window_resized {
            eh.recreate_swapchain = false;

            let dimensions = eh.window().inner_size();

            let (new_swapchain, new_swapchain_images) =
                match swapchain.recreate(SwapchainCreateInfo {
                    image_extent: dimensions.into(),
                    ..swapchain.create_info()
                }) {
                    Ok(ok) => ok,
                    Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                    Err(err) => panic!("{}", err),
                };
            swapchain = new_swapchain;
            swapchain_images = new_swapchain_images.clone();

            if eh.window_resized {
                eh.window_resized = false;

                viewport.dimensions = eh.dimensions.to_array();

                let constant_data = shaders::ty::ConstantBuffer {
                    view: viewport.dimensions,
                    ratio: [FOV, -FOV * viewport.dimensions[1] / viewport.dimensions[0]],
                };

                // TODO: make this a lot cleaner
                let mut builder = AutoCommandBufferBuilder::primary(
                    device.clone(),
                    queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                builder
                    .update_buffer(Box::new(constant_data), constant_buffer.clone(), 0)
                    .unwrap();
                let command_buffer = builder.build().unwrap();

                if let Some(future) = fences[previous_fence_index].clone() {
                    future
                        .then_signal_fence_and_flush()
                        .unwrap()
                        .wait(None)
                        .unwrap();
                }
                sync::now(device.clone())
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_signal_fence_and_flush()
                    .unwrap()
                    .wait(None)
                    .unwrap();

                compute_pipeline = get_compute_pipeline(device.clone(), compute_shader.clone());

                intermediate_image = get_intermediate_image(
                    device.clone(),
                    swapchain_images[0].dimensions().width_height(),
                    queue_family_index,
                );

                accumulator_images =
                    get_temporal_images(device.clone(), dimensions, queue_family_index);

                compute_descriptor_set = get_compute_descriptor_set(
                    compute_pipeline.clone(),
                    mutable_buffer.clone(),
                    constant_buffer.clone(),
                    intermediate_image.clone(),
                    accumulator_images.clone(),
                );
            }
        }

        let (image_index, suboptimal, image_future) =
            match swapchain::acquire_next_image(swapchain.clone(), None) {
                Ok(ok) => ok,
                Err(AcquireError::OutOfDate) => {
                    return eh.recreate_swapchain = true;
                }
                Err(err) => panic!("{}", err),
            };
        eh.recreate_swapchain |= suboptimal;

        if let Some(image_fence) = &fences[image_index] {
            image_fence.wait(None).unwrap();
        }

        command_buffers[image_index] = Some(get_command_buffer(
            device.clone(),
            queue.clone(),
            compute_pipeline.clone(),
            push_constants.clone(),
            compute_descriptor_set.clone(),
            intermediate_image.clone(),
            swapchain_images[image_index].clone(),
            accumulator_images.clone(),
        ));

        let previous_future = match fences[previous_fence_index].clone() {
            Some(future) => future.boxed(),
            None => {
                let mut future = sync::now(device.clone());
                future.cleanup_finished();
                future.boxed()
            }
        };

        let future = previous_future
            .join(image_future)
            .then_execute(queue.clone(), command_buffers[image_index].clone().unwrap())
            .unwrap()
            .then_swapchain_present(
                queue.clone(),
                PresentInfo {
                    index: image_index,
                    ..PresentInfo::swapchain(swapchain.clone())
                },
            )
            .then_signal_fence_and_flush();

        fences[image_index] = match future {
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
        previous_fence_index = image_index;
    })
}
