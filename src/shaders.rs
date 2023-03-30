vulkano_shaders::shader! {
    shaders: {
        Direct: {
            ty: "compute",
            path: "shaders/direct.glsl",
        },
        BufferRays: {
            ty: "compute",
            path: "shaders/buffer_rays.glsl",
        },
        MoveLightmap: {
            ty: "compute",
            path: "shaders/move_lightmap.glsl",
        }
    },
    types_meta: { #[derive(Clone, Copy, Default, Debug, bytemuck::Pod, bytemuck::Zeroable)] },
    include: ["includes_trace_ray.glsl", "includes_general.glsl", "includes_random.glsl"],
    define: [
        ("ITEM_COUNT", "65536"),
        ("RAYS_INDIRECT", "4"),
        ("LIGHTMAP_COUNT", "6"),
        ("LIGHTMAP_SIZE", "128")
    ] // TODO: sync defines with consts
}

pub(crate) const ITEM_COUNT: u32 = 65536;
pub(crate) const RAYS_INDIRECT: usize = 4;
pub(crate) const LIGHTMAP_COUNT: usize = 6; // TODO: rename to LIGHTMAP_LAYERS
pub(crate) const LIGHTMAP_SIZE: u32 = 128;

use vulkano::device::Device;

use vulkano::shader::ShaderModule;

use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct Shaders {
    pub(crate) direct: Arc<ShaderModule>,
    pub(crate) buffer_rays: Arc<ShaderModule>,
    pub(crate) move_lightmap: Arc<ShaderModule>,
}

impl Shaders {
    pub(crate) fn load(device: Arc<Device>) -> Self {
        Self {
            direct: load_Direct(device.clone()).unwrap(),
            buffer_rays: load_BufferRays(device.clone()).unwrap(),
            move_lightmap: load_MoveLightmap(device.clone()).unwrap(),
        }
    }
}