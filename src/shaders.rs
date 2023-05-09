vulkano_shaders::shader! {
    shaders: {
        Direct: {
            ty: "compute",
            path: "shaders/direct.glsl",
        },
        LMInit: {
            ty: "compute",
            path: "shaders/lm_init.glsl",
        },
        LMRoughMarch: {
            ty: "compute",
            path: "shaders/lm_rough_march.glsl",
        },
        LMPrimary: {
            ty: "compute",
            path: "shaders/lm_primary.glsl",
        },
        LMSecondary: {
            ty: "compute",
            path: "shaders/lm_secondary.glsl",
        },
    },
    custom_derives: [Copy, Clone, Debug],
    include: ["includes_march_ray.glsl", "includes_general.glsl"],
    define: [
        ("LM_RAYS", "4"),
        ("LM_SAMPLES", "1024"),
        ("LM_COUNT", "6"),
        ("LM_SIZE", "128"),
        ("LM_VOXELS_PER_FRAME", "32768"),
        ("LM_MAX_POINTS", "262144"),
        ("MAX_OBJECTS", "128"),
        ("MAX_MATERIALS", "32"),
        ("MAX_RAY_RADIUS", "0.317856")
    ], // TODO: sync defines with consts
    vulkan_version: "1.2", // TODO: vulkan 1.3
    spirv_version: "1.6"
}

pub(crate) const LM_RAYS: usize = 4;
pub(crate) const LM_SAMPLES: u32 = 1024;
pub(crate) const LM_COUNT: u32 = 6; // TODO: rename to LIGHTMAP_LAYERS
pub(crate) const LM_SIZE: u32 = 128;
pub(crate) const LM_VOXELS_PER_FRAME: u32 = 32768;
// pub(crate) const LM_MAX_POINTS: u32 = 262144;

pub(crate) const MAX_OBJECTS: usize = 128;
pub(crate) const MAX_MATERIALS: usize = 32;

// pub(crate) const MAX_RAY_RADIUS: f32 = crate::ray_directions::MAX_RADIUS;

use vulkano::device::Device;

use vulkano::shader::ShaderModule;

use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct Shaders {
    pub(crate) direct: Arc<ShaderModule>,
    pub(crate) lm_init: Arc<ShaderModule>,
    pub(crate) lm_rough_march: Arc<ShaderModule>,
    pub(crate) lm_primary: Arc<ShaderModule>,
    pub(crate) lm_secondary: Arc<ShaderModule>,
}

impl Shaders {
    pub(crate) fn load(device: Arc<Device>) -> Self {
        Self {
            direct: load_direct(device.clone()).unwrap(),
            lm_init: load_lm_init(device.clone()).unwrap(),
            lm_rough_march: load_lm_rough_march(device.clone()).unwrap(),
            lm_primary: load_lm_primary(device.clone()).unwrap(),
            lm_secondary: load_lm_secondary(device.clone()).unwrap(),
        }
    }
}
