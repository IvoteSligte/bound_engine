vulkano_shaders::shader! {
    shaders: {
        DirectVertex: {
            ty: "vertex",
            path: "shaders/direct.vert",
        },
        DirectFragment: {
            ty: "fragment",
            path: "shaders/direct.frag",
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
        ("MAX_MATERIALS", "32"),
        ("SH_CS", "4")
    ], // TODO: sync defines with consts
    vulkan_version: "1.2", // TODO: vulkan 1.3
    spirv_version: "1.6"
}

pub const LM_LAYERS: u32 = 6;
pub const LM_SIZE: u32 = 128;

pub const RADIANCE_SIZE: u32 = 32;
pub const SH_CS: u32 = 4;

pub const MAX_OBJECTS: usize = 128;
pub const MAX_MATERIALS: usize = 32;

use vulkano::device::Device;

use vulkano::shader::ShaderModule;

use std::sync::Arc;

#[derive(Clone)]
pub struct Shaders {
    pub direct: DirectShaders,
    pub sdf: Arc<ShaderModule>,
    pub radiance: Arc<ShaderModule>,
    pub radiance_precalc: Arc<ShaderModule>,
}

impl Shaders {
    pub fn load(device: Arc<Device>) -> Self {
        Self {
            direct: DirectShaders::load(device.clone()),
            sdf: load_sdf(device.clone()).unwrap(),
            radiance: load_radiance(device.clone()).unwrap(),
            radiance_precalc: load_radiance_precalc(device.clone()).unwrap(),
        }
    }
}

#[derive(Clone)]
pub struct DirectShaders {
    pub vertex: Arc<ShaderModule>,
    pub fragment: Arc<ShaderModule>,
}

impl DirectShaders {
    fn load(device: Arc<Device>) -> Self {
        Self {
            vertex: load_direct_vertex(device.clone()).unwrap(),
            fragment: load_direct_fragment(device).unwrap(),
        }
    }
}
