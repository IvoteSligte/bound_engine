vulkano_shaders::shader! {
    shaders: {
        Direct: {
            ty: "compute",
            path: "shaders/direct.glsl",
        },
        Accumulation: {
            ty: "compute",
            path: "shaders/accumulation.glsl",
        },
    },
    types_meta: { #[derive(Clone, Copy, Default, Debug, bytemuck::Pod, bytemuck::Zeroable)] },
    include: ["includes_trace_ray.glsl", "includes_general.glsl"],
    define: [
        ("LM_RAYS", "4"),
        ("LM_SAMPLES", "1024"),
        ("LM_COUNT", "6"),
        ("LM_SIZE", "128")
    ] // TODO: sync defines with consts
}

pub(crate) const LM_RAYS: usize = 4;
pub(crate) const LM_SAMPLES: u32 = 1024;
pub(crate) const LM_COUNT: u32 = 6; // TODO: rename to LIGHTMAP_LAYERS
pub(crate) const LM_SIZE: u32 = 128;

use vulkano::device::Device;

use vulkano::shader::ShaderModule;

use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct Shaders {
    pub(crate) direct: Arc<ShaderModule>,
    pub(crate) accumulation: Arc<ShaderModule>,
}

impl Shaders {
    pub(crate) fn load(device: Arc<Device>) -> Self {
        Self {
            direct: load_Direct(device.clone()).unwrap(),
            accumulation: load_Accumulation(device.clone()).unwrap(),
        }
    }
}
