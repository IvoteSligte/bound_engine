mod bounding_volume_hierarchy;

use std::{f32::consts::PI, sync::Arc};

use fps_counter::FPSCounter;
use glam::{DVec2, Quat, UVec2, Vec2, Vec3};
use image::io::Reader as ImageReader;
use vulkano::{
    buffer::{BufferAccess, BufferUsage, DeviceLocalBuffer},
    command_buffer::{
        allocator::{
            CommandBufferAllocator, StandardCommandBufferAllocator,
            StandardCommandBufferAllocatorCreateInfo,
        },
        AutoCommandBufferBuilder, BlitImageInfo, ClearColorImageInfo, CommandBufferUsage,
        PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
    },
    descriptor_set::{
        allocator::{DescriptorSetAllocator, StandardDescriptorSetAllocator},
        DescriptorSetWithOffsets, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo,
    },
    format::Format,
    image::{
        view::ImageView, ImageAccess, ImageCreateFlags, ImageDimensions, ImageUsage,
        ImmutableImage, MipmapsCount, StorageImage,
    },
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{MemoryAllocator, StandardMemoryAllocator},
    pipeline::{graphics::viewport::Viewport, ComputePipeline, Pipeline, PipelineBindPoint},
    sampler::{Sampler, SamplerCreateInfo},
    shader::ShaderModule,
    swapchain::{
        self, AcquireError, Surface, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
        SwapchainPresentInfo,
    },
    sync::{self, FenceSignalFuture, FlushError, GpuFuture, PipelineStage},
    Version, VulkanLibrary,
};
use winit::{
    dpi::PhysicalSize,
    event::VirtualKeyCode,
    event_loop::{ControlFlow, EventLoop},
    window::{CursorGrabMode, Fullscreen, Window, WindowBuilder},
};
use winit_event_helper::{ElementState2, EventHelper};

use crate::bounding_volume_hierarchy::{BVHNode, BVH};

// TODO: fix green showing up on image occasionally after switching resolution
mod shaders {
    vulkano_shaders::shader! {
        shaders: {
            PathtraceCompute: {
                ty: "compute",
                path: "shaders/pathtrace_compute.glsl",
            },
        },
        types_meta: { #[derive(Clone, Copy, Default, Debug, bytemuck::Pod, bytemuck::Zeroable)] },
        include: ["compute_includes.glsl"]
    }
}

fn select_physical_device<'a>(
    instance: &'a Arc<Instance>,
    surface: &'a Surface,
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
                    q.supports_stage(PipelineStage::ComputeShader)
                        && q.supports_stage(PipelineStage::Host)
                    //&& q.supports_stage(PipelineStage::Copy)
                    //&& q.supports_stage(PipelineStage::Blit)
                })
                .map(|q| (p, q as u32))
                .filter(|(p, q)| p.surface_support(*q, surface).unwrap_or(false))
        })
        .max_by_key(|(p, _)| match p.properties().device_type {
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
    pathtrace_shader: Arc<ShaderModule>,
) -> Arc<ComputePipeline> {
    ComputePipeline::new(
        device.clone(),
        pathtrace_shader.entry_point("main").unwrap(),
        &(),
        None, // TODO: look into caches
        |_| {},
    )
    .unwrap()
}

fn get_data_image(
    allocator: &(impl MemoryAllocator + ?Sized),
    dimensions: PhysicalSize<u32>,
    queue_family_index: u32,
) -> Arc<StorageImage> {
    StorageImage::with_usage(
        allocator,
        ImageDimensions::Dim2d {
            width: dimensions.width,
            height: dimensions.height,
            array_layers: 1,
        },
        Format::R16G16B16A16_SFLOAT, // TODO: loosely match format with swapchain image format
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

fn get_compute_descriptor_set<A>(
    allocator: &A,
    pipeline: Arc<ComputePipeline>,
    real_time_buffer: Arc<dyn BufferAccess>,
    bvh_buffer: Arc<dyn BufferAccess>,
    mutable_buffer: Arc<dyn BufferAccess>,
    constant_buffer: Arc<dyn BufferAccess>,
    data_image: Arc<StorageImage>,
    blue_noise_image: Arc<dyn ImageAccess>,
    sampler: Arc<Sampler>,
) -> Arc<PersistentDescriptorSet<A::Alloc>>
where
    A: DescriptorSetAllocator + ?Sized,
{
    let data_image_view = ImageView::new_default(data_image.clone()).unwrap();

    let blue_noise_image_view = ImageView::new_default(blue_noise_image).unwrap();

    let pathtrace_descriptor_set = PersistentDescriptorSet::new(
        allocator,
        pipeline.layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::buffer(0, real_time_buffer.clone()),
            WriteDescriptorSet::buffer(1, bvh_buffer.clone()),
            WriteDescriptorSet::buffer(2, mutable_buffer.clone()),
            WriteDescriptorSet::buffer(3, constant_buffer.clone()),
            WriteDescriptorSet::image_view_sampler(4, blue_noise_image_view, sampler.clone()),
            WriteDescriptorSet::image_view(5, data_image_view.clone()),
        ],
    )
    .unwrap();

    pathtrace_descriptor_set
}

fn get_command_buffer<A, S>(
    allocator: &A,
    queue: Arc<Queue>,
    pipeline: Arc<ComputePipeline>,
    descriptor_set: S,
    dimensions: PhysicalSize<u32>,
    render_image: Arc<StorageImage>,
) -> Arc<PrimaryAutoCommandBuffer<A::Alloc>>
where
    A: CommandBufferAllocator,
    S: Into<DescriptorSetWithOffsets> + Clone,
{
    let mut builder = AutoCommandBufferBuilder::primary(
        allocator,
        queue.queue_family_index(),
        CommandBufferUsage::SimultaneousUse,
    )
    .unwrap();

    let dispatch_groups = [
        (dimensions.width as f32 / 8.0).ceil() as u32,
        (dimensions.height as f32 / 8.0).ceil() as u32,
        8,
    ];

    builder
        .clear_color_image(ClearColorImageInfo::image(render_image.clone()))
        .unwrap()
        .bind_pipeline_compute(pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipeline.layout().clone(),
            0,
            descriptor_set.clone(),
        )
        .dispatch(dispatch_groups)
        .unwrap();

    Arc::new(builder.build().unwrap())
}

mod rotation {
    use glam::Vec3;

    pub const UP: Vec3 = Vec3::new(0.0, 0.0, 1.0);
    pub const FORWARD: Vec3 = Vec3::new(0.0, 1.0, 0.0);
    pub const RIGHT: Vec3 = Vec3::new(1.0, 0.0, 0.0);
}

// field of view
const FOV: f32 = 1.0;

struct Data {
    window: Arc<Window>,
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

impl Data {
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
    let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
    let surface = vulkano_win::create_surface_from_winit(window.clone(), instance.clone()).unwrap();
    // BUG: spamming any key on application startup will make the window invisible
    window.set_cursor_visible(false);
    window.set_cursor_grab(CursorGrabMode::Confined).unwrap();
    window.set_resizable(false);

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

    // TODO: improve
    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

    let capabilities = physical
        .surface_capabilities(&surface, Default::default())
        .unwrap();
    let dimensions = window.inner_size();
    let image_format = physical
        .surface_formats(&surface, Default::default())
        .unwrap()
        .iter()
        .max_by_key(|(format, _)| match format {
            Format::R8G8B8A8_SRGB | Format::B8G8R8A8_SRGB => 1,
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
                transfer_dst: true,
                ..ImageUsage::empty()
            },
            ..Default::default()
        },
    )
    .unwrap();

    let pathtrace_compute_shader = shaders::load_PathtraceCompute(device.clone()).unwrap();

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: window.inner_size().into(),
        depth_range: 0.0..1.0,
    };

    let compute_pipeline = get_compute_pipeline(device.clone(), pathtrace_compute_shader.clone());

    const BVH_OBJECTS: [BVHNode; 9] = [
        BVHNode {
            center: Vec3::new(-100020.0, 0.0, 0.0),
            radius: 1e5,
            child: None,
            next: None,
            leaf: Some(1),
            parent: None,
        },
        BVHNode {
            center: Vec3::new(100020.0, 0.0, 0.0),
            radius: 1e5,
            child: None,
            next: None,
            leaf: Some(2),
            parent: None,
        },
        BVHNode {
            center: Vec3::new(0.0, -100020.0, 0.0),
            radius: 1e5,
            child: None,
            next: None,
            leaf: Some(3),
            parent: None,
        },
        BVHNode {
            center: Vec3::new(0.0, 100020.0, 0.0),
            radius: 1e5,
            child: None,
            next: None,
            leaf: Some(3),
            parent: None,
        },
        BVHNode {
            center: Vec3::new(0.0, 0.0, -100020.0),
            radius: 1e5,
            child: None,
            next: None,
            leaf: Some(3),
            parent: None,
        },
        BVHNode {
            center: Vec3::new(1.0, 0.0, 100020.0),
            radius: 1e5,
            child: None,
            next: None,
            leaf: Some(3),
            parent: None,
        },
        BVHNode {
            center: Vec3::new(0.0, 0.0, 119.7),
            radius: 100.0,
            child: None,
            next: None,
            leaf: Some(4),
            parent: None,
        },
        BVHNode {
            center: Vec3::new(-3.0, 1.0, -16.0),
            radius: 4.0,
            child: None,
            next: None,
            leaf: Some(5),
            parent: None,
        },
        BVHNode {
            center: Vec3::new(4.0, 3.0, -11.0),
            radius: 2.0,
            child: None,
            next: None,
            leaf: Some(6),
            parent: None,
        },
    ];

    let materials = [
        // dummy material
        shaders::ty::Material {
            reflectance: [0.0; 3],
            emittance: [0.0; 3],
            _dummy0: [0u8; 4],
            _dummy1: [0u8; 4],
        },
        shaders::ty::Material {
            reflectance: [0.95, 0.1, 0.1],
            emittance: [0.0; 3],
            _dummy0: [0u8; 4],
            _dummy1: [0u8; 4],
        },
        shaders::ty::Material {
            reflectance: [0.1, 0.95, 0.1],
            emittance: [0.0; 3],
            _dummy0: [0u8; 4],
            _dummy1: [0u8; 4],
        },
        shaders::ty::Material {
            reflectance: [0.95; 3],
            emittance: [0.0; 3],
            _dummy0: [0u8; 4],
            _dummy1: [0u8; 4],
        },
        shaders::ty::Material {
            reflectance: [0.0; 3],
            emittance: [2.0; 3],
            _dummy0: [0u8; 4],
            _dummy1: [0u8; 4],
        },
        shaders::ty::Material {
            reflectance: [0.7; 3],
            emittance: [0.0; 3],
            _dummy0: [0u8; 4],
            _dummy1: [0u8; 4],
        },
        shaders::ty::Material {
            reflectance: [0.5; 3],
            emittance: [0.5, 0.0, 0.0],
            _dummy0: [0u8; 4],
            _dummy1: [0u8; 4],
        },
    ];

    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo {
            primary_buffer_count: 2,
            secondary_buffer_count: 0,
            ..StandardCommandBufferAllocatorCreateInfo::default()
        },
    );
    let mut alloc_command_buffer_builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue_family_index,
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    let mut bvh = BVH {
        head: 0,
        nodes: vec![BVH_OBJECTS[0].clone()],
    };

    for b in BVH_OBJECTS[1..].into_iter() {
        bvh.merge_in_place(BVH {
            head: 0,
            nodes: vec![b.clone().into()],
        });
    }

    let bvh_buffer = DeviceLocalBuffer::<shaders::ty::BoundingVolumeHierarchy>::from_data(
        &memory_allocator,
        bvh.into(),
        BufferUsage {
            uniform_buffer: true,
            ..BufferUsage::empty()
        },
        &mut alloc_command_buffer_builder,
    )
    .unwrap();

    let mut mutable_data = shaders::ty::MutableData {
        ..Default::default()
    };
    mutable_data.mats[..materials.len()].copy_from_slice(&materials);

    let mutable_buffer = DeviceLocalBuffer::from_data(
        &memory_allocator,
        mutable_data,
        BufferUsage {
            uniform_buffer: true,
            ..BufferUsage::empty()
        },
        &mut alloc_command_buffer_builder,
    )
    .unwrap();

    // near constant data
    let constant_data = shaders::ty::ConstantBuffer {
        ratio: [FOV, -FOV * viewport.dimensions[1] / viewport.dimensions[0]],
    };

    let constant_buffer = DeviceLocalBuffer::from_data(
        &memory_allocator,
        constant_data,
        BufferUsage {
            uniform_buffer: true,
            ..BufferUsage::empty()
        },
        &mut alloc_command_buffer_builder,
    )
    .unwrap();

    let blue_noise_raw_image = ImageReader::open("blue_noise\\HDR_RGB_0_64x64.png")
        .unwrap()
        .decode()
        .unwrap()
        .to_rgba16();

    let blue_noise_data = blue_noise_raw_image
        .chunks(2)
        .map(|v| {
            const M: f32 = u16::MAX as f32;
            [v[0], ((v[1] as f32 / M * 2.0 - 1.0).acos() / PI * M) as u16] // maps v[1] to cosine distribution
        })
        .collect::<Vec<_>>();

    let blue_noise_data_len = blue_noise_data.len();

    let blue_noise_image = ImmutableImage::from_iter(
        &memory_allocator,
        blue_noise_data.into_iter(),
        ImageDimensions::Dim1d {
            width: blue_noise_data_len as u32,
            array_layers: 1,
        },
        MipmapsCount::One,
        Format::R16G16_SNORM,
        &mut alloc_command_buffer_builder,
    )
    .unwrap();

    let mut real_time_data = shaders::ty::RealTimeBuffer {
        rotation: [0.0; 4],
        position: [0.0; 3],
        time: 0.0,
        previousRotation: [0.0; 4],
        inversePreviousRotation: [0.0; 4],
        previousPosition: [0.0; 3],
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

    alloc_command_buffer_builder
        .build()
        .unwrap()
        .execute(queue.clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let sampler = Sampler::new(
        device.clone(),
        SamplerCreateInfo::simple_repeat_linear_no_mipmap(),
    )
    .unwrap();

    let mut data_image = get_data_image(&memory_allocator, dimensions, queue_family_index);

    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
    let descriptor_set = get_compute_descriptor_set(
        &descriptor_set_allocator,
        compute_pipeline.clone(),
        real_time_buffer.clone(),
        bvh_buffer.clone(),
        mutable_buffer.clone(),
        constant_buffer.clone(),
        data_image.clone(),
        blue_noise_image.clone(),
        sampler.clone(),
    );

    let mut main_command_buffer = get_command_buffer(
        &command_buffer_allocator,
        queue.clone(),
        compute_pipeline.clone(),
        descriptor_set.clone(),
        dimensions,
        data_image.clone(),
    );

    let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; swapchain_images.len()];
    let mut previous_fence_index = 0;

    let mut eh = EventHelper::new(Data {
        window: window,
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

    let exit = |data: &mut EventHelper<Data>| data.quit = true;
    eh.window_close_requested(exit);
    eh.window_keyboard_input(VirtualKeyCode::Escape, ElementState2::Pressed, exit);

    eh.device_mouse_motion(|data, (dx, dy)| data.cursor_delta += DVec2::new(dx, dy).as_vec2());

    eh.window_focused(|data, focused| {
        data.window_frozen = !focused;
        if focused {
            data.window
                .set_cursor_grab(CursorGrabMode::Confined)
                .unwrap();
        } else {
            data.window.set_cursor_grab(CursorGrabMode::None).unwrap();
        }
    });

    eh.window_resized(|data, mut size| {
        data.window_frozen = size.width == 0 || size.height == 0;
        data.window_resized = true;

        if size.width < size.height {
            size.height = size.width;
            data.window.set_inner_size(size);
        }

        data.dimensions = UVec2::new(size.width, size.height).as_vec2();
    });

    eh.window_keyboard_input(
        VirtualKeyCode::F11,
        ElementState2::Pressed,
        |data| match data.window.fullscreen() {
            Some(_) => data.window.set_fullscreen(None),
            None => data
                .window
                .set_fullscreen(Some(Fullscreen::Borderless(None))),
        },
    );

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
        real_time_data.previousRotation = real_time_data.rotation;
        real_time_data.inversePreviousRotation = Quat::from_array(real_time_data.rotation)
            .conjugate()
            .to_array();
        real_time_data.previousPosition = real_time_data.position;
        real_time_data.rotation = eh.rotation().to_array();
        real_time_data.position = (Vec3::from(real_time_data.position) + eh.position()).to_array();
        eh.delta_position = Vec3::ZERO;

        real_time_data.time = eh.secs_since_start() as f32;

        // rendering
        if eh.recreate_swapchain || eh.window_resized {
            eh.recreate_swapchain = false;

            let dimensions = eh.window.inner_size();

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
                    ratio: [FOV, -FOV * viewport.dimensions[1] / viewport.dimensions[0]],
                };

                // TODO: make this a lot cleaner
                let mut builder = AutoCommandBufferBuilder::primary(
                    &command_buffer_allocator,
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

                let pipeline =
                    get_compute_pipeline(device.clone(), pathtrace_compute_shader.clone());

                data_image = get_data_image(&memory_allocator, dimensions, queue_family_index);

                let descriptor_sets = get_compute_descriptor_set(
                    &descriptor_set_allocator,
                    pipeline.clone(),
                    real_time_buffer.clone(),
                    bvh_buffer.clone(),
                    mutable_buffer.clone(),
                    constant_buffer.clone(),
                    data_image.clone(),
                    blue_noise_image.clone(),
                    sampler.clone(),
                );

                main_command_buffer = get_command_buffer(
                    &command_buffer_allocator,
                    queue.clone(),
                    pipeline.clone(),
                    descriptor_sets.clone(),
                    dimensions,
                    data_image.clone(),
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

        let mut real_time_command_buffer_builder = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue_family_index,
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        real_time_command_buffer_builder
            .update_buffer(Box::new(real_time_data), real_time_buffer.clone(), 0)
            .unwrap();
        let real_time_command_buffer = real_time_command_buffer_builder.build().unwrap();

        let mut blit_command_buffer_builder = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        blit_command_buffer_builder
            .blit_image(BlitImageInfo::images(
                data_image.clone(),
                swapchain_images[image_index as usize].clone(),
            ))
            .unwrap();
        let blit_command_buffer = blit_command_buffer_builder.build().unwrap();

        if let Some(image_fence) = &fences[image_index as usize] {
            image_fence.wait(None).unwrap();
        }

        let previous_future = match fences[previous_fence_index].clone() {
            Some(future) => future.boxed(),
            None => {
                let mut future = sync::now(device.clone());
                future.cleanup_finished();
                future.boxed()
            }
        };

        let future = previous_future
            .then_execute(queue.clone(), real_time_command_buffer)
            .unwrap()
            .then_execute(queue.clone(), main_command_buffer.clone())
            .unwrap()
            .join(image_future)
            .then_execute(queue.clone(), blit_command_buffer)
            .unwrap()
            .then_swapchain_present(
                queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush();

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
    })
}
