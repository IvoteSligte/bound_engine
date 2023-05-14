vulkano_shaders::shader! {
    shaders: {
        Direct: {
            ty: "compute",
            path: "shaders/direct.glsl",
        },
        LmInit: {
            ty: "compute",
            path: "shaders/lm_init.glsl",
        },
        LmRender: {
            ty: "compute",
            path: "shaders/lm_render.glsl",
        },
        LmDenoise: {
            ty: "compute",
            path: "shaders/lm_denoise.glsl",
        },
        LmStore: {
            ty: "compute",
            path: "shaders/lm_store.glsl",
        },
    },
    custom_derives: [Copy, Clone, Debug],
    include: ["includes_march_ray.glsl", "includes_general.glsl"],
    define: [
        ("LM_COUNT", "6"),
        ("LM_SIZE", "128"),
        ("LM_MAX_POINTS", "524288"),
        ("MAX_OBJECTS", "128"),
        ("MAX_MATERIALS", "32"),
        ("NOISE_BUFFER_LENGTH", "1024")
    ], // TODO: sync defines with consts
    vulkan_version: "1.2", // TODO: vulkan 1.3
    spirv_version: "1.6"
}

pub(crate) const LM_COUNT: u32 = 6; // TODO: rename to LIGHTMAP_LAYERS
pub(crate) const LM_SIZE: u32 = 128;
pub(crate) const LM_MAX_POINTS: u32 = 524288;

pub(crate) const MAX_OBJECTS: usize = 128;
pub(crate) const MAX_MATERIALS: usize = 32;

// pub(crate) const NOISE_BUFFER_LENGTH: u32 = ray_directions::VECTORS.len() as u32;

// pub(crate) const MAX_RAY_RADIUS: f32 = crate::ray_directions::MAX_RADIUS;

use vulkano::device::Device;

use vulkano::shader::ShaderModule;

use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct Shaders {
    pub(crate) direct: Arc<ShaderModule>,
    pub(crate) lm_init: Arc<ShaderModule>,
    pub(crate) lm_render: Arc<ShaderModule>,
    pub(crate) lm_denoise: Arc<ShaderModule>,
    pub(crate) lm_store: Arc<ShaderModule>,
}

impl Shaders {
    pub(crate) fn load(device: Arc<Device>) -> Self {
        Self {
            direct: load_direct(device.clone()).unwrap(),
            lm_init: load_lm_init(device.clone()).unwrap(),
            lm_render: load_lm_render(device.clone()).unwrap(),
            lm_denoise: load_lm_denoise(device.clone()).unwrap(),
            lm_store: load_lm_store(device.clone()).unwrap(),
        }
    }
}
