mod bvh;

use std::{f32::consts::PI, sync::Arc};

use fps_counter::FPSCounter;
use glam::{DVec2, IVec3, Quat, UVec2, UVec3, Vec2, Vec3};
use vulkano::{
    buffer::{BufferAccess, BufferUsage, DeviceLocalBuffer},
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, BlitImageInfo, CommandBufferUsage, CopyImageInfo,
        PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo,
    },
    format::Format,
    image::{
        view::ImageView, ImageCreateFlags, ImageDimensions, ImageUsage, ImageViewAbstract,
        StorageImage, SwapchainImage,
    },
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{
        FreeListAllocator, GenericMemoryAllocator, MemoryAllocator, StandardMemoryAllocator,
    },
    pipeline::{graphics::viewport::Viewport, ComputePipeline, Pipeline, PipelineBindPoint},
    sampler::Filter,
    shader::{ShaderModule, SpecializationConstants},
    swapchain::{
        self, AcquireError, Surface, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
        SwapchainPresentInfo,
    },
    sync::{self, FenceSignalFuture, FlushError, GpuFuture, PipelineStage},
    Version, VulkanLibrary,
};
use winit::{
    dpi::PhysicalSize,
    event_loop::{ControlFlow, EventLoop},
    window::{CursorGrabMode, Window, WindowBuilder},
};
use winit_event_helper::*;
use winit_fullscreen::WindowFullScreen;

use crate::bvh::{CpuBVH, CpuNode};

mod shaders {
    vulkano_shaders::shader! {
        shaders: {
            Pathtrace: {
                ty: "compute",
                path: "shaders/pathtrace.glsl",
            },
            Lightmap: {
                ty: "compute",
                path: "shaders/lightmap.glsl",
            }
        },
        types_meta: { #[derive(Clone, Copy, Default, Debug, bytemuck::Pod, bytemuck::Zeroable)] },
        include: ["includes.glsl"]
    }
}

// FIXME: index 3 doesn't show up in final bvh in renders?
const BVH_OBJECTS: [CpuNode; 9] = [
    CpuNode {
        position: Vec3::new(-1000020.0, 0.0, 0.0),
        radius: 1e6,
        child: None,
        next: None,
        leaf: Some(1),
        parent: None,
    },
    CpuNode {
        position: Vec3::new(1000020.0, 0.0, 0.0),
        radius: 1e6,
        child: None,
        next: None,
        leaf: Some(2),
        parent: None,
    },
    CpuNode {
        position: Vec3::new(0.0, -1000020.0, 0.0),
        radius: 1e6,
        child: None,
        next: None,
        leaf: Some(3),
        parent: None,
    },
    CpuNode {
        position: Vec3::new(0.0, 1000020.0, 0.0),
        radius: 1e6,
        child: None,
        next: None,
        leaf: Some(3),
        parent: None,
    },
    CpuNode {
        position: Vec3::new(0.0, 0.0, -1000020.0),
        radius: 1e6,
        child: None,
        next: None,
        leaf: Some(3),
        parent: None,
    },
    CpuNode {
        position: Vec3::new(0.0, 0.0, 1000020.0),
        radius: 1e6,
        child: None,
        next: None,
        leaf: Some(3),
        parent: None,
    },
    CpuNode {
        position: Vec3::new(0.0, 0.0, 119.7),
        radius: 100.0,
        child: None,
        next: None,
        leaf: Some(4),
        parent: None,
    },
    CpuNode {
        position: Vec3::new(-3.0, 1.0, -16.0),
        radius: 4.0,
        child: None,
        next: None,
        leaf: Some(5),
        parent: None,
    },
    CpuNode {
        position: Vec3::new(8.0, 15.0, 0.0),
        radius: 2.0,
        child: None,
        next: None,
        leaf: Some(6),
        parent: None,
    },
];

const MATERIALS: [shaders::ty::Material; 7] = [
    // dummy material
    shaders::ty::Material {
        reflectance: [0.0; 3],
        emittance: [0.0; 3],
        _dummy0: [0u8; 4],
        _dummy1: [0u8; 4],
    },
    shaders::ty::Material {
        reflectance: [0.99, 0.1, 0.1],
        emittance: [0.0; 3],
        _dummy0: [0u8; 4],
        _dummy1: [0u8; 4],
    },
    shaders::ty::Material {
        reflectance: [0.1, 0.99, 0.1],
        emittance: [0.0; 3],
        _dummy0: [0u8; 4],
        _dummy1: [0u8; 4],
    },
    shaders::ty::Material {
        reflectance: [0.99; 3],
        emittance: [0.0; 3],
        _dummy0: [0u8; 4],
        _dummy1: [0u8; 4],
    },
    shaders::ty::Material {
        reflectance: [0.0; 3],
        emittance: [1.0; 3],
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
        emittance: [0.2, 0.0, 0.0],
        _dummy0: [0u8; 4],
        _dummy1: [0u8; 4],
    },
];

const COLOR_IMAGE_DIMENSION_DIV: f32 = 1.0;
const LIGHTMAP_SIZE: u32 = 128;
const LIGHTMAP_COUNT: usize = 6;

#[derive(Clone)]
struct DescriptorSetCollection {
    pathtrace_set: Arc<PersistentDescriptorSet>,
    lightmap_sets: Vec<Arc<PersistentDescriptorSet>>,
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
        .filter(|p| p.properties().device_type == PhysicalDeviceType::DiscreteGpu)
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
        .next()
        .unwrap()
}

fn get_compute_pipeline<Css>(
    device: Arc<Device>,
    shader: Arc<ShaderModule>,
    specialization_constants: &Css,
) -> Arc<ComputePipeline>
where
    Css: SpecializationConstants,
{
    ComputePipeline::new(
        device.clone(),
        shader.entry_point("main").unwrap(),
        specialization_constants,
        None, // TODO: look into caches
        |_| {},
    )
    .unwrap()
}

fn get_color_image(
    allocator: &(impl MemoryAllocator + ?Sized),
    dimensions: PhysicalSize<u32>,
    queue_family_index: u32,
) -> Arc<StorageImage> {
    StorageImage::with_usage(
        allocator,
        ImageDimensions::Dim2d {
            width: (dimensions.width as f32 / COLOR_IMAGE_DIMENSION_DIV) as u32,
            height: (dimensions.height as f32 / COLOR_IMAGE_DIMENSION_DIV) as u32,
            array_layers: 1,
        },
        Format::R16G16B16A16_SFLOAT, // TODO: loosely match format with swapchain image format
        ImageUsage {
            storage: true,
            transfer_src: true,
            ..ImageUsage::empty()
        },
        ImageCreateFlags::empty(),
        [queue_family_index],
    )
    .unwrap()
}

fn get_lightmap_images(
    allocator: &(impl MemoryAllocator + ?Sized),
    queue_family_index: u32,
    main_count: usize,
) -> (Vec<Arc<StorageImage>>, Arc<StorageImage>) {
    let dimensions = ImageDimensions::Dim3d {
        width: LIGHTMAP_SIZE,
        height: LIGHTMAP_SIZE,
        depth: LIGHTMAP_SIZE,
    };

    let mains = (0..main_count)
        .map(|_| {
            StorageImage::with_usage(
                allocator,
                dimensions,
                Format::R32G32B32A32_SFLOAT,
                ImageUsage {
                    storage: true,
                    transfer_dst: true,
                    ..ImageUsage::empty()
                },
                ImageCreateFlags::empty(),
                [queue_family_index],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    let staging = StorageImage::with_usage(
        allocator,
        dimensions,
        Format::R32G32B32A32_SFLOAT,
        ImageUsage {
            storage: true,
            transfer_src: true,
            ..ImageUsage::empty()
        },
        ImageCreateFlags::empty(),
        [queue_family_index],
    )
    .unwrap();

    (mains, staging)
}

fn get_compute_descriptor_sets(
    allocator: &StandardDescriptorSetAllocator,
    pathtrace_pipeline: Arc<ComputePipeline>,
    lightmap_pipelines: Vec<Arc<ComputePipeline>>,
    real_time_buffer: Arc<dyn BufferAccess>,
    bvh_buffer: Arc<dyn BufferAccess>,
    mutable_buffer: Arc<dyn BufferAccess>,
    constant_buffer: Arc<dyn BufferAccess>,
    color_image: Arc<StorageImage>,
    (lightmap_images, lightmap_staging_image): (Vec<Arc<StorageImage>>, Arc<StorageImage>),
) -> DescriptorSetCollection {
    let color_image_view = ImageView::new_default(color_image.clone()).unwrap();

    let lightmap_views = lightmap_images
        .iter()
        .map(|vlm| ImageView::new_default(vlm.clone()).unwrap() as Arc<dyn ImageViewAbstract>)
        .collect::<Vec<_>>();
    let lightmap_staging_view = ImageView::new_default(lightmap_staging_image.clone()).unwrap();

    let pathtrace_set = PersistentDescriptorSet::new(
        allocator,
        pathtrace_pipeline.layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::buffer(0, real_time_buffer.clone()),
            WriteDescriptorSet::buffer(1, bvh_buffer.clone()),
            WriteDescriptorSet::buffer(2, mutable_buffer.clone()),
            WriteDescriptorSet::buffer(3, constant_buffer.clone()),
            WriteDescriptorSet::image_view(4, color_image_view.clone()),
            WriteDescriptorSet::image_view_array(5, 0, lightmap_views.clone()),
        ],
    )
    .unwrap();

    let lightmap_sets = lightmap_views
        .iter()
        .zip(lightmap_pipelines.iter())
        .map(|(lm_view, lm_pipeline)| {
            PersistentDescriptorSet::new(
                allocator,
                lm_pipeline.layout().set_layouts()[0].clone(),
                [
                    WriteDescriptorSet::buffer(0, real_time_buffer.clone()),
                    WriteDescriptorSet::image_view(1, lm_view.clone()),
                    WriteDescriptorSet::image_view(2, lightmap_staging_view.clone()),
                ],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    DescriptorSetCollection {
        pathtrace_set,
        lightmap_sets,
    }
}

fn get_main_command_buffers(
    allocator: &StandardCommandBufferAllocator,
    queue: Arc<Queue>,
    pathtrace_pipeline: Arc<ComputePipeline>,
    descriptor_sets: DescriptorSetCollection,
    dimensions: PhysicalSize<u32>,
    color_image: Arc<StorageImage>,
    swapchain_images: Vec<Arc<SwapchainImage>>,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    let dispatch_pathtrace = [
        (dimensions.width as f32 / 8.0 / COLOR_IMAGE_DIMENSION_DIV).ceil() as u32,
        (dimensions.height as f32 / 4.0 / COLOR_IMAGE_DIMENSION_DIV).ceil() as u32,
        1,
    ];

    swapchain_images
        .clone()
        .into_iter()
        .map(|swapchain_image| {
            let mut builder = AutoCommandBufferBuilder::primary(
                allocator,
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .unwrap();

            builder
                .blit_image(BlitImageInfo {
                    filter: Filter::Linear,
                    ..BlitImageInfo::images(color_image.clone(), swapchain_image.clone())
                })
                .unwrap()
                .bind_pipeline_compute(pathtrace_pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    pathtrace_pipeline.layout().clone(),
                    0,
                    descriptor_sets.pathtrace_set.clone(),
                )
                .dispatch(dispatch_pathtrace)
                .unwrap();

            Arc::new(builder.build().unwrap())
        })
        .collect()
}

fn get_lightmap_command_buffer(
    allocator: &StandardCommandBufferAllocator,
    queue: Arc<Queue>,
    lightmap_pipelines: Vec<Arc<ComputePipeline>>,
    descriptor_sets: DescriptorSetCollection,
    (lightmap_images, lightmap_staging_image): (Vec<Arc<StorageImage>>, Arc<StorageImage>),
) -> Arc<PrimaryAutoCommandBuffer> {
    let dispatch_lightmap = UVec3::splat(LIGHTMAP_SIZE / 4).to_array();

    let mut builder = AutoCommandBufferBuilder::primary(
        allocator,
        queue.queue_family_index(),
        CommandBufferUsage::SimultaneousUse,
    )
    .unwrap();

    for i in 0..lightmap_images.len() {
        builder
            .bind_pipeline_compute(lightmap_pipelines[i].clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                lightmap_pipelines[i].layout().clone(),
                0,
                descriptor_sets.lightmap_sets[i].clone(),
            )
            .dispatch(dispatch_lightmap)
            .unwrap()
            .copy_image(CopyImageInfo::images(
                lightmap_staging_image.clone(),
                lightmap_images[i].clone(),
            ))
            .unwrap();
    }

    Arc::new(builder.build().unwrap())
}

fn get_bvh_buffer(
    memory_allocator: &GenericMemoryAllocator<Arc<FreeListAllocator>>,
    alloc_command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
) -> Arc<DeviceLocalBuffer<shaders::ty::GpuBVH>> {
    let mut bvh: CpuBVH = BVH_OBJECTS[0].clone().into();

    for n in BVH_OBJECTS[1..].iter() {
        bvh.merge_in_place(n.clone().into());
    }

    // DEBUG
    bvh.graphify();

    DeviceLocalBuffer::<shaders::ty::GpuBVH>::from_data(
        memory_allocator,
        bvh.into(),
        BufferUsage {
            uniform_buffer: true,
            ..BufferUsage::empty()
        },
        alloc_command_buffer_builder,
    )
    .unwrap()
}

mod rotation {
    use glam::Vec3;

    pub const UP: Vec3 = Vec3::new(0.0, 0.0, 1.0);
    pub const FORWARD: Vec3 = Vec3::new(0.0, 1.0, 0.0);
    pub const RIGHT: Vec3 = Vec3::new(1.0, 0.0, 0.0);
}

// field of view
const FOV: f32 = 0.5;

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
    movement_multiplier: f32,
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
    window.set_visible(true);
    window.set_cursor_visible(false);
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

    let (mut swapchain, swapchain_images) = Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: capabilities.min_image_count + 1, // TODO: improve
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

    let pathtrace_shader = shaders::load_Pathtrace(device.clone()).unwrap();
    let lightmap_shader = shaders::load_Lightmap(device.clone()).unwrap();

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: window.inner_size().into(),
        depth_range: 0.0..1.0,
    };

    let pathtrace_pipeline = get_compute_pipeline(device.clone(), pathtrace_shader.clone(), &());
    let lightmap_pipelines = (0..LIGHTMAP_COUNT)
        .map(|i| {
            get_compute_pipeline(
                device.clone(),
                lightmap_shader.clone(),
                &shaders::LightmapSpecializationConstants {
                    LIGHTMAP_INDEX: i as u32,
                },
            )
        })
        .collect::<Vec<_>>();

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

    let bvh_buffer = get_bvh_buffer(&memory_allocator, &mut alloc_command_buffer_builder);

    let mut mutable_data = shaders::ty::MutableData {
        ..Default::default()
    };
    mutable_data.mats[..MATERIALS.len()].copy_from_slice(&MATERIALS);

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

    let mut real_time_data = shaders::ty::RealTimeBuffer {
        rotation: [0.0; 4],
        previousRotation: [0.0; 4],
        position: [0.0; 3],
        previousPosition: [0.0; 3],
        deltaLightmapOrigins: [[0; 4]; LIGHTMAP_COUNT],
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
    let lightmap_images =
        get_lightmap_images(&memory_allocator, queue_family_index, LIGHTMAP_COUNT);

    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
    let descriptor_sets = get_compute_descriptor_sets(
        &descriptor_set_allocator,
        pathtrace_pipeline.clone(),
        lightmap_pipelines.clone(),
        real_time_buffer.clone(),
        bvh_buffer.clone(),
        mutable_buffer.clone(),
        constant_buffer.clone(),
        color_image.clone(),
        lightmap_images.clone(),
    );

    let mut pathtrace_command_buffers = get_main_command_buffers(
        &command_buffer_allocator,
        queue.clone(),
        pathtrace_pipeline.clone(),
        descriptor_sets.clone(),
        dimensions,
        color_image.clone(),
        swapchain_images.clone(),
    );

    let lightmap_command_buffer = get_lightmap_command_buffer(
        &command_buffer_allocator,
        queue.clone(),
        lightmap_pipelines.clone(),
        descriptor_sets.clone(),
        lightmap_images.clone(),
    );

    let mut lightmap_update = false;

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
        movement_multiplier: 25.0,
        rotation_multiplier: 1.0,
    });

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

    let mut fps_counter = FPSCounter::new();

    // TODO: clear image data on swapchain recreation

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

        let cursor_mov = eh.cursor_delta / eh.dimensions.x * eh.rotation_multiplier;
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

                let pathtrace_pipeline =
                    get_compute_pipeline(device.clone(), pathtrace_shader.clone(), &());

                let color_image =
                    get_color_image(&memory_allocator, dimensions, queue_family_index);

                let descriptor_sets = get_compute_descriptor_sets(
                    &descriptor_set_allocator,
                    pathtrace_pipeline.clone(),
                    lightmap_pipelines.clone(),
                    real_time_buffer.clone(),
                    bvh_buffer.clone(),
                    mutable_buffer.clone(),
                    constant_buffer.clone(),
                    color_image.clone(),
                    lightmap_images.clone(),
                );

                pathtrace_command_buffers = get_main_command_buffers(
                    &command_buffer_allocator,
                    queue.clone(),
                    pathtrace_pipeline.clone(),
                    descriptor_sets.clone(),
                    dimensions,
                    color_image.clone(),
                    new_swapchain_images.clone(),
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

        let mut future = previous_future;

        if lightmap_update {
            lightmap_update = false;

            future = future
                .then_execute(queue.clone(), lightmap_command_buffer.clone())
                .unwrap()
                .boxed();
        }

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

        future = future
            .then_execute(queue.clone(), real_time_command_buffer)
            .unwrap()
            .boxed();

        let future = future
            .then_execute(
                queue.clone(),
                pathtrace_command_buffers[image_index as usize].clone(),
            )
            .unwrap()
            .join(image_future)
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

// BUGS: Os(OsError { line: 1333, file: "{DIR}/winit-0.27.5/src/platform_impl/linux/x11/window.rs", error: XMisc("Cursor could not be confined: already confined by another client") })', src/main.rs:649:65
