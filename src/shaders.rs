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
        ("LM_LAYERS", "4"),
        ("RADIANCE_SIZE", "128"), // image resolution
        ("RADIANCE_UNIT", "2.0"), // unit size in the world
        ("MAX_MATERIALS", "32"),
        ("SH_CS", "4")
    ], // TODO: sync defines with consts
    vulkan_version: "1.2", // TODO: vulkan 1.3
    spirv_version: "1.6"
}

pub const LM_LAYERS: u32 = 4;

pub const RADIANCE_SIZE: u32 = 128;
pub const SH_CS: u32 = 4;

pub const MAX_MATERIALS: usize = 32;

use vulkano::device::Device;

use vulkano::shader::ShaderModule;

use std::sync::Arc;

#[derive(Clone)]
pub struct Shaders {
    pub direct: DirectShaders,
    pub radiance: Arc<ShaderModule>,
    pub radiance_precalc: Arc<ShaderModule>,
}

impl Shaders {
    pub fn load(device: Arc<Device>) -> Self {
        Self {
            direct: DirectShaders::load(device.clone()),
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
