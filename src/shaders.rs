vulkano_shaders::shader! {
    shaders: {
        Direct: {
            ty: "compute",
            path: "shaders/direct.glsl",
        },
        Sdf: {
            ty: "compute",
            path: "shaders/sdf.glsl",
        },
        RadiancePrecalc: {
            ty: "compute",
            path: "shaders/radiancePrecalc.glsl",
        },
        Radiance: {
            ty: "compute",
            path: "shaders/radiance.glsl",
        },
    },
    custom_derives: [Copy, Clone, Debug],
    include: ["includes_general.glsl", "sh_rotation.glsl"],
    define: [
        ("LM_LAYERS", "6"),
        ("LM_SIZE", "128"),
        ("RADIANCE_SIZE", "32"),
        ("MAX_OBJECTS", "128"),
        ("MAX_MATERIALS", "32")
    ], // TODO: sync defines with consts
    vulkan_version: "1.2", // TODO: vulkan 1.3
    spirv_version: "1.6"
}

pub const LM_LAYERS: u32 = 6;
pub const LM_SIZE: u32 = 128;

pub const RADIANCE_SIZE: u32 = 32;
pub const RADIANCE_SH_COEFS: u32 = 4;

pub const MAX_OBJECTS: usize = 128;
pub const MAX_MATERIALS: usize = 32;

use vulkano::device::Device;

use vulkano::shader::ShaderModule;

use std::sync::Arc;

#[derive(Clone)]
pub struct Shaders {
    pub direct: Arc<ShaderModule>,
    pub sdf: Arc<ShaderModule>,
    pub radiance: Arc<ShaderModule>,
    pub radiance_precalc: Arc<ShaderModule>,
}

impl Shaders {
    pub fn load(device: Arc<Device>) -> Self {
        Self {
            direct: load_direct(device.clone()).unwrap(),
            sdf: load_sdf(device.clone()).unwrap(),
            radiance: load_radiance(device.clone()).unwrap(),
            radiance_precalc: load_radiance_precalc(device.clone()).unwrap(),
        }
    }
}
