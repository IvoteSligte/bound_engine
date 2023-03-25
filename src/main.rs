mod bvh;

use std::{f32::consts::PI, sync::Arc};

use fps_counter::FPSCounter;
use glam::{DVec2, IVec3, Quat, UVec2, UVec3, Vec2, Vec3};
use vulkano::{
    buffer::{BufferAccess, BufferUsage, DeviceLocalBuffer},
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, BlitImageInfo, CommandBufferUsage, CopyImageInfo, FillBufferInfo,
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
            Direct: {
                ty: "compute",
                path: "shaders/direct.glsl",
            },
            Mid: {
                ty: "compute",
                path: "shaders/mid.glsl",
            },
            Last: {
                ty: "compute",
                path: "shaders/last.glsl",
            },
            MoveLightmap: {
                ty: "compute",
                path: "shaders/move_lightmap.glsl",
            }
        },
        types_meta: { #[derive(Clone, Copy, Default, Debug, bytemuck::Pod, bytemuck::Zeroable)] },
        include: ["includes_trace_ray.glsl", "includes_general.glsl", "includes_random.glsl"],
        define: [
            ("MID_COUNT", "65536"),
            ("LAST_COUNT", "65536*8")
        ] // TODO: sync defines with consts
    }
}

const MID_COUNT: u32 = 65536;
const LAST_COUNT: u32 = 65536*8;

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
        emittance: [0.7, 0.0, 0.0],
        _dummy0: [0u8; 4],
        _dummy1: [0u8; 4],
    },
];

const LIGHTMAP_SIZE: u32 = 128;
const LIGHTMAP_COUNT: usize = 6;

#[derive(Clone)]
struct DescriptorSetCollection {
    direct: Arc<PersistentDescriptorSet>,
    mid: Arc<PersistentDescriptorSet>,
    last: Arc<PersistentDescriptorSet>,
    move_mids: Vec<Arc<PersistentDescriptorSet>>,
    move_lasts: Vec<Arc<PersistentDescriptorSet>>,
}

#[derive(Clone)]
struct Shaders {
    direct: Arc<ShaderModule>,
    mid: Arc<ShaderModule>,
    last: Arc<ShaderModule>,
    lightmap: Arc<ShaderModule>,
}

impl Shaders {
    fn load(device: Arc<Device>) -> Self {
        Self {
            direct: shaders::load_Direct(device.clone()).unwrap(),
            mid: shaders::load_Mid(device.clone()).unwrap(),
            last: shaders::load_Last(device.clone()).unwrap(),
            lightmap: shaders::load_MoveLightmap(device.clone()).unwrap(),
        }
    }
}

#[derive(Clone)]
struct PathtracePipelines {
    direct: Arc<ComputePipeline>,
    mid: Arc<ComputePipeline>,
    last: Arc<ComputePipeline>,
}

impl PathtracePipelines {
    fn from_shaders(device: Arc<Device>, shaders: Shaders) -> Self {
        let direct = get_compute_pipeline(device.clone(), shaders.direct.clone(), &());
        let mid =
            get_compute_pipeline(device.clone(), shaders.mid.clone(), &());
        let last =
            get_compute_pipeline(device.clone(), shaders.last.clone(), &());

        Self {
            direct,
            mid,
            last,
        }
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
        None,
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
            width: dimensions.width,
            height: dimensions.height,
            array_layers: 1,
        },
        Format::R16G16B16A16_UNORM, // double precision for copying to srgb
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

#[derive(Clone)]
struct LightmapImages {
    mids: Vec<Arc<StorageImage>>,
    lasts: Vec<Arc<StorageImage>>,
    staging: Arc<StorageImage>,
    indirect_syncs: Vec<Arc<StorageImage>>,
}

#[derive(Clone)]
struct LightmapImageViews {
    mids: Vec<Arc<dyn ImageViewAbstract>>,
    lasts: Vec<Arc<dyn ImageViewAbstract>>,
    staging: Arc<dyn ImageViewAbstract>,
    indirect_syncs: Vec<Arc<dyn ImageViewAbstract>>,
}

impl LightmapImages {
    fn new(
        allocator: &(impl MemoryAllocator + ?Sized),
        queue_family_index: u32,
        count_per_set: usize,
    ) -> Self {
        let dimensions = ImageDimensions::Dim3d {
            width: LIGHTMAP_SIZE,
            height: LIGHTMAP_SIZE,
            depth: LIGHTMAP_SIZE,
        };

        let create_storage_image = |usage, format| {
            StorageImage::with_usage(
                allocator,
                dimensions,
                format,
                ImageUsage {
                    storage: true,
                    ..usage
                },
                ImageCreateFlags::empty(),
                [queue_family_index],
            )
            .unwrap()
        };

        let mids = (0..count_per_set)
            .map(|_| {
                create_storage_image(
                    ImageUsage {
                        transfer_dst: true,
                        ..ImageUsage::default()
                    },
                    Format::R8G8B8A8_UNORM,
                )
            })
            .collect();

        let lasts = (0..count_per_set)
            .map(|_| {
                create_storage_image(
                    ImageUsage {
                        transfer_dst: true,
                        ..ImageUsage::default()
                    },
                    Format::R8G8B8A8_UNORM,
                )
            })
            .collect();

        let staging = create_storage_image(
            ImageUsage {
                transfer_src: true,
                ..ImageUsage::default()
            },
            Format::R8G8B8A8_UNORM,
        );
        
        let indirect_syncs = (0..count_per_set)
            .map(|_| create_storage_image(ImageUsage::default(), Format::R32_UINT))
            .collect();

        Self {
            mids,
            lasts,
            staging,
            indirect_syncs,
        }
    }

    fn image_views(&self) -> LightmapImageViews {
        LightmapImageViews {
            mids: self
                .mids
                .iter()
                .map(|vlm| {
                    ImageView::new_default(vlm.clone()).unwrap() as Arc<dyn ImageViewAbstract>
                })
                .collect(),
            lasts: self
                .lasts
                .iter()
                .map(|vlm| {
                    ImageView::new_default(vlm.clone()).unwrap() as Arc<dyn ImageViewAbstract>
                })
                .collect(),
            staging: ImageView::new_default(self.staging.clone()).unwrap(),
            indirect_syncs: self
                .indirect_syncs
                .iter()
                .map(|vlm| {
                    ImageView::new_default(vlm.clone()).unwrap() as Arc<dyn ImageViewAbstract>
                })
                .collect(),
        }
    }
}

#[derive(Clone)]
struct LightmapBuffers {
    mid_buffer: Arc<DeviceLocalBuffer<shaders::ty::MidBuffer>>,
    mid_counters: Arc<DeviceLocalBuffer<shaders::ty::MidCounters>>,
    last_buffer: Arc<DeviceLocalBuffer<shaders::ty::LastBuffer>>,
    last_counters: Arc<DeviceLocalBuffer<shaders::ty::LastCounters>>,
}

impl LightmapBuffers {
    fn new(
        memory_allocator: &GenericMemoryAllocator<Arc<FreeListAllocator>>,
        queue_family_index: u32,
    ) -> Self {
        const BUFFER_USAGE: BufferUsage = BufferUsage {
            storage_buffer: true,
            uniform_buffer: true,
            ..BufferUsage::empty()
        };

        const COUNTER_USAGE: BufferUsage = BufferUsage {
            storage_buffer: true,
            uniform_buffer: true,
            transfer_dst: true,
            ..BufferUsage::empty()
        };

        Self {
            mid_buffer: DeviceLocalBuffer::new(
                memory_allocator,
                BUFFER_USAGE,
                [queue_family_index],
            )
            .unwrap(),
            last_buffer: DeviceLocalBuffer::new(
                memory_allocator,
                BUFFER_USAGE,
                [queue_family_index],
            )
            .unwrap(),
            mid_counters: DeviceLocalBuffer::new(
                memory_allocator,
                COUNTER_USAGE,
                [queue_family_index],
            )
            .unwrap(),
            last_counters: DeviceLocalBuffer::new(
                memory_allocator,
                COUNTER_USAGE,
                [queue_family_index],
            )
            .unwrap(),
        }
    }
}

fn get_compute_descriptor_sets(
    allocator: &StandardDescriptorSetAllocator,
    pathtrace_pipelines: PathtracePipelines,
    lightmap_pipelines: Vec<Arc<ComputePipeline>>,
    real_time_buffer: Arc<dyn BufferAccess>,
    bvh_buffer: Arc<dyn BufferAccess>,
    mutable_buffer: Arc<dyn BufferAccess>,
    constant_buffer: Arc<dyn BufferAccess>,
    color_image: Arc<StorageImage>,
    lightmap_images: LightmapImages,
    lightmap_buffers: LightmapBuffers,
) -> DescriptorSetCollection {
    let color_image_view = ImageView::new_default(color_image.clone()).unwrap();

    let lightmap_image_views = lightmap_images.image_views();

    let direct = PersistentDescriptorSet::new(
        allocator,
        pathtrace_pipelines.direct.layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::buffer(0, real_time_buffer.clone()),
            WriteDescriptorSet::buffer(1, bvh_buffer.clone()),
            WriteDescriptorSet::buffer(2, constant_buffer.clone()),
            WriteDescriptorSet::image_view(3, color_image_view.clone()),
            WriteDescriptorSet::image_view_array(
                4,
                0,
                lightmap_image_views.mids.clone(),
            ),
            WriteDescriptorSet::image_view_array(5, 0, lightmap_image_views.indirect_syncs.clone()),
            WriteDescriptorSet::buffer(6, lightmap_buffers.mid_buffer.clone()),
            WriteDescriptorSet::buffer(7, lightmap_buffers.mid_counters.clone()),
            WriteDescriptorSet::buffer(8, lightmap_buffers.last_buffer.clone()),
            WriteDescriptorSet::buffer(9, lightmap_buffers.last_counters.clone()),
        ],
    )
    .unwrap();

    let mid = PersistentDescriptorSet::new(
        allocator,
        pathtrace_pipelines.mid.layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::buffer(0, real_time_buffer.clone()),
            WriteDescriptorSet::buffer(1, bvh_buffer.clone()),
            WriteDescriptorSet::buffer(2, mutable_buffer.clone()),
            WriteDescriptorSet::image_view_array(
                3,
                0,
                lightmap_image_views.mids.clone(),
            ),
            WriteDescriptorSet::image_view_array(4, 0, lightmap_image_views.lasts.clone()),
            WriteDescriptorSet::image_view_array(5, 0, lightmap_image_views.indirect_syncs.clone()),
            WriteDescriptorSet::buffer(6, lightmap_buffers.mid_buffer.clone()),
            WriteDescriptorSet::buffer(7, lightmap_buffers.mid_counters.clone()),
            WriteDescriptorSet::buffer(8, lightmap_buffers.last_buffer.clone()),
            WriteDescriptorSet::buffer(9, lightmap_buffers.last_counters.clone()),
        ],
    )
    .unwrap();

    let last = PersistentDescriptorSet::new(
        allocator,
        pathtrace_pipelines.last.layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::buffer(0, real_time_buffer.clone()),
            WriteDescriptorSet::buffer(1, bvh_buffer.clone()),
            WriteDescriptorSet::buffer(2, mutable_buffer.clone()),
            WriteDescriptorSet::image_view_array(3, 0, lightmap_image_views.lasts.clone()),
            WriteDescriptorSet::image_view_array(4, 0, lightmap_image_views.indirect_syncs.clone()),
            WriteDescriptorSet::buffer(5, lightmap_buffers.last_buffer.clone()),
            WriteDescriptorSet::buffer(6, lightmap_buffers.last_counters.clone()),
        ],
    )
    .unwrap();

    let move_lightmaps = |lightmaps: Vec<Arc<dyn ImageViewAbstract>>| {
        lightmaps
            .iter()
            .zip(lightmap_pipelines.iter())
            .map(|(lm_view, lm_pipeline)| {
                PersistentDescriptorSet::new(
                    allocator,
                    lm_pipeline.layout().set_layouts()[0].clone(),
                    [
                        WriteDescriptorSet::buffer(0, real_time_buffer.clone()),
                        WriteDescriptorSet::image_view(1, lm_view.clone()),
                        WriteDescriptorSet::image_view(2, lightmap_image_views.staging.clone()),
                    ],
                )
                .unwrap()
            })
            .collect::<Vec<_>>()
    };

    let move_mids = move_lightmaps(lightmap_image_views.mids);
    let move_lasts = move_lightmaps(lightmap_image_views.lasts);

    DescriptorSetCollection {
        direct,
        mid,
        last,
        move_mids,
        move_lasts,
    }
}

fn get_pathtrace_command_buffers(
    allocator: &StandardCommandBufferAllocator,
    queue: Arc<Queue>,
    pipelines: PathtracePipelines,
    descriptor_sets: DescriptorSetCollection,
    dimensions: PhysicalSize<u32>,
    color_image: Arc<StorageImage>,
    swapchain_images: Vec<Arc<SwapchainImage>>,
    lightmap_buffers: LightmapBuffers,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    let dispatch_direct = [
        (dimensions.width as f32 / 8.0).ceil() as u32,
        (dimensions.height as f32 / 8.0).ceil() as u32,
        1,
    ];

    let dispatch_mid = [MID_COUNT / 64, 1, 1];
    let dispatch_last = [LAST_COUNT / 64, 1, 1];

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
                .bind_pipeline_compute(pipelines.direct.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    pipelines.direct.layout().clone(),
                    0,
                    descriptor_sets.direct.clone(),
                )
                .dispatch(dispatch_direct)
                .unwrap()
                .bind_pipeline_compute(pipelines.mid.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    pipelines.mid.layout().clone(),
                    0,
                    descriptor_sets.mid.clone(),
                )
                .dispatch(dispatch_mid)
                .unwrap()
                .bind_pipeline_compute(pipelines.last.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    pipelines.last.layout().clone(),
                    0,
                    descriptor_sets.last.clone(),
                )
                .dispatch(dispatch_last)
                .unwrap()
                .fill_buffer(FillBufferInfo::dst_buffer(
                    lightmap_buffers.mid_counters.clone(),
                )) // clear buffer
                .unwrap()
                .fill_buffer(FillBufferInfo::dst_buffer(
                    lightmap_buffers.last_counters.clone(),
                )) // clear buffer
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
    lightmap_images: LightmapImages,
) -> Arc<PrimaryAutoCommandBuffer> {
    let dispatch_lightmap = UVec3::splat(LIGHTMAP_SIZE / 4).to_array();

    let mut builder = AutoCommandBufferBuilder::primary(
        allocator,
        queue.queue_family_index(),
        CommandBufferUsage::SimultaneousUse,
    )
    .unwrap();

    for i in 0..LIGHTMAP_COUNT {
        builder
            .bind_pipeline_compute(lightmap_pipelines[i].clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                lightmap_pipelines[i].layout().clone(),
                0,
                descriptor_sets.move_mids[i].clone(),
            )
            .dispatch(dispatch_lightmap)
            .unwrap()
            .copy_image(CopyImageInfo::images(
                lightmap_images.staging.clone(),
                lightmap_images.mids[i].clone(),
            ))
            .unwrap();
    }

    for i in 0..LIGHTMAP_COUNT {
        builder
            .bind_pipeline_compute(lightmap_pipelines[i].clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                lightmap_pipelines[i].layout().clone(),
                0,
                descriptor_sets.move_lasts[i].clone(),
            )
            .dispatch(dispatch_lightmap)
            .unwrap()
            .copy_image(CopyImageInfo::images(
                lightmap_images.staging.clone(),
                lightmap_images.lasts[i].clone(),
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
    //bvh.graphify();

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

    let shaders = Shaders::load(device.clone());

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: window.inner_size().into(),
        depth_range: 0.0..1.0,
    };

    let mut pathtrace_pipelines = PathtracePipelines::from_shaders(device.clone(), shaders.clone());
    let lightmap_pipelines = (0..LIGHTMAP_COUNT)
        .map(|i| {
            get_compute_pipeline(
                device.clone(),
                shaders.lightmap.clone(),
                &shaders::MoveLightmapSpecializationConstants {
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

    let lightmap_buffers = LightmapBuffers::new(&memory_allocator, queue_family_index);

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
        LightmapImages::new(&memory_allocator, queue_family_index, LIGHTMAP_COUNT);

    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
    let descriptor_sets = get_compute_descriptor_sets(
        &descriptor_set_allocator,
        pathtrace_pipelines.clone(),
        lightmap_pipelines.clone(),
        real_time_buffer.clone(),
        bvh_buffer.clone(),
        mutable_buffer.clone(),
        constant_buffer.clone(),
        color_image.clone(),
        lightmap_images.clone(),
        lightmap_buffers.clone(),
    );

    let mut pathtrace_command_buffers = get_pathtrace_command_buffers(
        &command_buffer_allocator,
        queue.clone(),
        pathtrace_pipelines.clone(),
        descriptor_sets.clone(),
        dimensions,
        color_image.clone(),
        swapchain_images.clone(),
        lightmap_buffers.clone(),
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

                pathtrace_pipelines.direct =
                    get_compute_pipeline(device.clone(), shaders.direct.clone(), &());

                let color_image =
                    get_color_image(&memory_allocator, dimensions, queue_family_index);

                let descriptor_sets = get_compute_descriptor_sets(
                    &descriptor_set_allocator,
                    pathtrace_pipelines.clone(),
                    lightmap_pipelines.clone(),
                    real_time_buffer.clone(),
                    bvh_buffer.clone(),
                    mutable_buffer.clone(),
                    constant_buffer.clone(),
                    color_image.clone(),
                    lightmap_images.clone(),
                    lightmap_buffers.clone(),
                );

                pathtrace_command_buffers = get_pathtrace_command_buffers(
                    &command_buffer_allocator,
                    queue.clone(),
                    pathtrace_pipelines.clone(),
                    descriptor_sets.clone(),
                    dimensions,
                    color_image.clone(),
                    new_swapchain_images.clone(),
                    lightmap_buffers.clone(),
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
    })
}

// FIXME: BUG |||| Os(OsError { line: 1333, file: "{DIR}/winit-0.27.5/src/platform_impl/linux/x11/window.rs", error: XMisc("Cursor could not be confined: already confined by another client") })', src/main.rs:649:65

// TODO: sort things into different files
