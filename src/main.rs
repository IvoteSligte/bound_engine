use std::{f32::consts::PI, sync::Arc};

use fps_counter::FPSCounter;
use glam::{DVec2, Quat, UVec2, UVec3, Vec2, Vec3};
use image::io::Reader as ImageReader;
use vulkano::{
    buffer::{BufferAccess, BufferUsage, DeviceLocalBuffer},
    command_buffer::{
        allocator::{
            CommandBufferAllocator, StandardCommandBufferAllocator,
            StandardCommandBufferAllocatorCreateInfo,
        },
        AutoCommandBufferBuilder, CommandBufferUsage, CopyImageInfo, PrimaryAutoCommandBuffer,
        PrimaryCommandBufferAbstract, BlitImageInfo,
    },
    descriptor_set::{
        allocator::{DescriptorSetAllocator, StandardDescriptorSetAllocator},
        DescriptorSetsCollection, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo,
    },
    format::Format,
    image::{
        view::ImageView, ImageAccess, ImageCreateFlags, ImageDimensions, ImageUsage,
        ImmutableImage, MipmapsCount, StorageImage, SwapchainImage,
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

// TODO: fix green showing up on image occasionally after switching resolution
mod shaders {
    vulkano_shaders::shader! {
        shaders: {
            PathtraceCompute: {
                ty: "compute",
                path: "shaders/pathtrace_compute.glsl",
            },
            VarianceCompute: {
                ty: "compute",
                path: "shaders/variance_compute.glsl",
            },
            DenoiseCompute: {
                ty: "compute",
                path: "shaders/denoise_compute.glsl",
            },
        },
        types_meta: { #[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)] },
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

fn get_compute_pipelines(
    device: Arc<Device>,
    shaders: impl IntoIterator<Item = Arc<ShaderModule>>,
) -> Vec<Arc<ComputePipeline>> {
    shaders
        .into_iter()
        .map(|shader| {
            ComputePipeline::new(
                device.clone(),
                shader.entry_point("main").unwrap(),
                &(),
                None, // TODO: look into caches
                |_| {},
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

// color + variance images
fn get_data_images(
    allocator: &(impl MemoryAllocator + ?Sized),
    dimensions: PhysicalSize<u32>,
    queue_family_index: u32,
) -> [Arc<StorageImage>; 2] {
    let data_0_image = StorageImage::with_usage(
        allocator,
        ImageDimensions::Dim2d {
            width: dimensions.width,
            height: dimensions.height,
            array_layers: 1,
        },
        Format::R16G16B16A16_SFLOAT, // TODO: loosely match format with swapchain image format
        ImageUsage {
            storage: true,
            sampled: true,
            transfer_src: true,
            transfer_dst: true,
            ..ImageUsage::empty()
        },
        ImageCreateFlags::empty(),
        [queue_family_index],
    )
    .unwrap();

    let data_1_image = StorageImage::with_usage(
        allocator,
        ImageDimensions::Dim2d {
            width: dimensions.width,
            height: dimensions.height,
            array_layers: 1,
        },
        Format::R16G16B16A16_SFLOAT,
        ImageUsage {
            storage: true,
            transfer_src: true,
            ..ImageUsage::empty()
        },
        ImageCreateFlags::empty(),
        [queue_family_index],
    )
    .unwrap();

    [data_0_image, data_1_image]
}

fn get_normals_depth_image(
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
        Format::R32G32B32A32_SFLOAT,
        ImageUsage {
            storage: true,
            ..ImageUsage::empty()
        },
        ImageCreateFlags::empty(),
        [queue_family_index],
    )
    .unwrap()
}

fn get_history_length_image(
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
        Format::R16_SFLOAT,
        ImageUsage {
            storage: true,
            ..ImageUsage::empty()
        },
        ImageCreateFlags::empty(),
        [queue_family_index],
    )
    .unwrap()
}

fn get_compute_descriptor_sets<A>(
    allocator: &A,
    pipelines: Vec<Arc<ComputePipeline>>,
    mutable_buffer: Arc<dyn BufferAccess>,
    constant_buffer: Arc<dyn BufferAccess>,
    data_images: [Arc<StorageImage>; 2],
    blue_noise_image: Arc<dyn ImageAccess>,
    sampler: Arc<Sampler>,
    normals_depth_image: Arc<StorageImage>,
    history_length_image: Arc<StorageImage>,
) -> [Arc<PersistentDescriptorSet<A::Alloc>>; 3]
where
    A: DescriptorSetAllocator + ?Sized,
{
    let data_image_views = data_images
        .iter()
        .map(|image| ImageView::new_default(image.clone()).unwrap())
        .collect::<Vec<_>>();
    let normals_depth_image_view = ImageView::new_default(normals_depth_image.clone()).unwrap();
    let history_length_image_view = ImageView::new_default(history_length_image.clone()).unwrap();

    let blue_noise_image_view = ImageView::new_default(blue_noise_image).unwrap();

    let pathtrace_descriptor_set = PersistentDescriptorSet::new(
        allocator,
        pipelines[0].layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::buffer(0, mutable_buffer.clone()),
            WriteDescriptorSet::buffer(1, constant_buffer.clone()),
            WriteDescriptorSet::image_view_sampler(2, data_image_views[0].clone(), sampler.clone()),
            WriteDescriptorSet::image_view(3, data_image_views[1].clone()),
            WriteDescriptorSet::image_view(4, normals_depth_image_view.clone()),
            WriteDescriptorSet::image_view(5, history_length_image_view.clone()),
            WriteDescriptorSet::image_view_sampler(6, blue_noise_image_view, sampler.clone()),
        ],
    )
    .unwrap();

    let variance_descriptor_set = PersistentDescriptorSet::new(
        allocator,
        pipelines[1].layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::image_view(0, data_image_views[1].clone()),
            WriteDescriptorSet::image_view(1, data_image_views[0].clone()),
            WriteDescriptorSet::image_view(2, normals_depth_image_view.clone()),
            WriteDescriptorSet::image_view(3, history_length_image_view.clone()),
        ],
    )
    .unwrap();

    let denoise_descriptor_set = PersistentDescriptorSet::new(
        allocator,
        pipelines[2].layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::image_view(0, data_image_views[0].clone()),
            WriteDescriptorSet::image_view(1, data_image_views[1].clone()),
            WriteDescriptorSet::image_view(2, normals_depth_image_view),
            WriteDescriptorSet::image_view(3, history_length_image_view.clone()),
        ],
    )
    .unwrap();

    [
        pathtrace_descriptor_set,
        variance_descriptor_set,
        denoise_descriptor_set,
    ]
}

fn get_command_buffers<A, S>(
    allocator: &A,
    queue: Arc<Queue>,
    pipelines: Vec<Arc<ComputePipeline>>,
    pathtrace_push_constants: shaders::ty::PathtracePushConstants,
    descriptor_sets: [S; 3],
    swapchain_image: Arc<SwapchainImage>,
    data_images: [Arc<StorageImage>; 2],
) -> [Arc<PrimaryAutoCommandBuffer<A::Alloc>>; 2]
where
    A: CommandBufferAllocator,
    S: DescriptorSetsCollection + Clone,
{
    let mut builder = AutoCommandBufferBuilder::primary(
        allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    let [pathtrace_pipeline, variance_pipeline, denoise_pipeline] = [
        pipelines[0].clone(),
        pipelines[1].clone(),
        pipelines[2].clone(),
    ];
    let [pathtrace_descriptor_set, variance_descriptor_set, denoise_descriptor_set] =
        descriptor_sets;

    let dispatch_groups =
        (UVec3::from_array(swapchain_image.dimensions().width_height_depth()).as_vec3() / 8.0)
            .ceil()
            .as_uvec3()
            .to_array();

    builder
        .bind_pipeline_compute(pathtrace_pipeline.clone())
        .push_constants(
            pathtrace_pipeline.layout().clone(),
            0,
            pathtrace_push_constants,
        )
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pathtrace_pipeline.layout().clone(),
            0,
            pathtrace_descriptor_set.clone(),
        )
        .dispatch(dispatch_groups)
        .unwrap()
        .bind_pipeline_compute(variance_pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            variance_pipeline.layout().clone(),
            0,
            variance_descriptor_set.clone(),
        )
        .dispatch(dispatch_groups)
        .unwrap()
        .bind_pipeline_compute(denoise_pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            denoise_pipeline.layout().clone(),
            0,
            denoise_descriptor_set.clone(),
        );

    const MAX_STAGE: u32 = 5;
    let mut denoise_push_constants = shaders::ty::DenoisePushConstants { stage: 0 };

    for i in 0..MAX_STAGE {
        denoise_push_constants.stage = i;

        builder
            .push_constants(denoise_pipeline.layout().clone(), 0, denoise_push_constants)
            .dispatch(dispatch_groups)
            .unwrap()
            .copy_image(CopyImageInfo::images(
                data_images[1].clone(),
                data_images[0].clone(),
            ))
            .unwrap();
    }

    // // TODO: turn into multiple use command buffers
    let mut blit_command_buffer_builder = AutoCommandBufferBuilder::primary(
        allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    blit_command_buffer_builder
        .blit_image(BlitImageInfo::images(data_images[0].clone(), swapchain_image)).unwrap();

    [
        Arc::new(builder.build().unwrap()),
        Arc::new(blit_command_buffer_builder.build().unwrap()),
    ]
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
    let variance_compute_shader = shaders::load_VarianceCompute(device.clone()).unwrap();
    let denoise_compute_shader = shaders::load_DenoiseCompute(device.clone()).unwrap();

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: window.inner_size().into(),
        depth_range: 0.0..1.0,
    };

    let mut compute_pipelines = get_compute_pipelines(
        device.clone(),
        [
            pathtrace_compute_shader.clone(),
            variance_compute_shader.clone(),
            denoise_compute_shader.clone(),
        ],
    );

    let materials = [
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
            emittance: [0.5, 0.0, 0.0],
            _dummy0: [0u8; 4],
            _dummy1: [0u8; 4],
        },
    ];

    let objects = [
        shaders::ty::Object {
            pos: [-1010.0, 0.0, 0.0],
            radiusSquared: 1000.0 * 1000.0,
            mat: 0,
            _dummy0: [0u8; 12],
        },
        shaders::ty::Object {
            pos: [1010.0, 0.0, 0.0],
            radiusSquared: 1000.0 * 1000.0,
            mat: 1,
            _dummy0: [0u8; 12],
        },
        shaders::ty::Object {
            pos: [0.0, 1010.0, 0.0],
            radiusSquared: 1000.0 * 1000.0,
            mat: 2,
            _dummy0: [0u8; 12],
        },
        shaders::ty::Object {
            pos: [0.0, -1010.0, 0.0],
            radiusSquared: 1000.0 * 1000.0,
            mat: 2,
            _dummy0: [0u8; 12],
        },
        shaders::ty::Object {
            pos: [0.0, 1010.0, 0.0],
            radiusSquared: 1000.0 * 1000.0,
            mat: 2,
            _dummy0: [0u8; 12],
        },
        shaders::ty::Object {
            pos: [0.0, 0.0, -1010.0],
            radiusSquared: 1000.0 * 1000.0,
            mat: 2,
            _dummy0: [0u8; 12],
        },
        shaders::ty::Object {
            pos: [0.0, 0.0, 1010.0],
            radiusSquared: 1000.0 * 1000.0,
            mat: 2,
            _dummy0: [0u8; 12],
        },
        shaders::ty::Object {
            pos: [0.0, 0.0, 109.95],
            radiusSquared: 100.0 * 100.0,
            mat: 3,
            _dummy0: [0u8; 12],
        },
        shaders::ty::Object {
            pos: [-3.0, 1.0, -6.0],
            radiusSquared: 4.0 * 4.0,
            mat: 4,
            _dummy0: [0u8; 12],
        },
        shaders::ty::Object {
            pos: [4.0, 3.0, -1.0],
            radiusSquared: 2.0 * 2.0,
            mat: 5,
            _dummy0: [0u8; 12],
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

    let mut mutable_data = shaders::ty::MutableData {
        ..Default::default()
    };
    mutable_data.mats[..materials.len()].copy_from_slice(&materials);
    mutable_data.objs[..objects.len()].copy_from_slice(&objects);

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

    let blue_noise_raw_image = ImageReader::open("blue_noise\\HDR_RGBA_0_256x256.png")
        .unwrap()
        .decode()
        .unwrap();

    let blue_noise_data = blue_noise_raw_image
        .to_rgba16()
        .chunks(4)
        .map(|v| {
            const M: f32 = u16::MAX as f32;
            [v[0], ((v[1] as f32 / M).acos() * M) as u16] // maps v[1] to cosine distribution
        })
        .collect::<Vec<_>>()
        .into_iter();

    let blue_noise_image = ImmutableImage::from_iter(
        &memory_allocator,
        blue_noise_data,
        ImageDimensions::Dim2d {
            width: blue_noise_raw_image.width(),
            height: blue_noise_raw_image.height(),
            array_layers: 1,
        },
        MipmapsCount::One,
        Format::R16G16_SNORM,
        &mut alloc_command_buffer_builder,
    )
    .unwrap();

    let sampler = Sampler::new(
        device.clone(),
        SamplerCreateInfo::simple_repeat_linear_no_mipmap(),
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

    let mut pathtrace_push_constants = shaders::ty::PathtracePushConstants {
        rot: [0.0; 4],
        pos: [0.0; 3],
        time: 0.0,
        pRot: [0.0; 4],
        ipRot: [0.0; 4],
        pPos: [0.0; 3],
    };

    let mut data_images = get_data_images(&memory_allocator, dimensions, queue_family_index);

    let history_length_image =
        get_history_length_image(&memory_allocator, dimensions, queue_family_index);

    let normals_depth_image =
        get_normals_depth_image(&memory_allocator, dimensions, queue_family_index);

    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
    let mut descriptor_sets = get_compute_descriptor_sets(
        &descriptor_set_allocator,
        compute_pipelines.clone(),
        mutable_buffer.clone(),
        constant_buffer.clone(),
        data_images.clone(),
        blue_noise_image.clone(),
        sampler.clone(),
        normals_depth_image.clone(),
        history_length_image.clone(),
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
        pathtrace_push_constants.pRot = pathtrace_push_constants.rot;
        pathtrace_push_constants.ipRot = Quat::from_array(pathtrace_push_constants.rot)
            .conjugate()
            .to_array();
        pathtrace_push_constants.pPos = pathtrace_push_constants.pos;
        pathtrace_push_constants.rot = eh.rotation().to_array();
        pathtrace_push_constants.pos =
            (Vec3::from(pathtrace_push_constants.pos) + eh.position()).to_array();
        eh.delta_position = Vec3::ZERO;

        pathtrace_push_constants.time = eh.secs_since_start() as f32;

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

                compute_pipelines = get_compute_pipelines(
                    device.clone(),
                    [
                        pathtrace_compute_shader.clone(),
                        variance_compute_shader.clone(),
                        denoise_compute_shader.clone(),
                    ],
                );

                data_images = get_data_images(&memory_allocator, dimensions, queue_family_index);

                let variance_image =
                    get_history_length_image(&memory_allocator, dimensions, queue_family_index);

                let normals_depth_image =
                    get_normals_depth_image(&memory_allocator, dimensions, queue_family_index);

                descriptor_sets = get_compute_descriptor_sets(
                    &descriptor_set_allocator,
                    compute_pipelines.clone(),
                    mutable_buffer.clone(),
                    constant_buffer.clone(),
                    data_images.clone(),
                    blue_noise_image.clone(),
                    sampler.clone(),
                    normals_depth_image.clone(),
                    variance_image.clone(),
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

        let [command_buffer, copy_command_buffer] = get_command_buffers(
            &command_buffer_allocator,
            queue.clone(),
            compute_pipelines.clone(),
            pathtrace_push_constants.clone(),
            descriptor_sets.clone(),
            swapchain_images[image_index as usize].clone(),
            data_images.clone(),
        );

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
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .join(image_future)
            .then_execute(queue.clone(), copy_command_buffer)
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
