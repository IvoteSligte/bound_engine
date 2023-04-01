use crate::shaders::Shaders;
use crate::{shaders, LIGHTMAP_COUNT, FOV};

use vulkano::pipeline::ComputePipeline;

use vulkano::device::Device;
use vulkano::shader::{ShaderModule, SpecializationConstants};
use winit::dpi::PhysicalSize;

use std::sync::Arc;

pub(crate) fn get_compute_pipeline<Css>(
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

#[derive(Clone)]
pub(crate) struct Pipelines {
    pub(crate) direct: Arc<ComputePipeline>,
    pub(crate) buffer_rays: Arc<ComputePipeline>,
    pub(crate) move_lightmaps: Vec<Arc<ComputePipeline>>,
}

impl Pipelines {
    pub(crate) fn from_shaders(device: Arc<Device>, shaders: Shaders, dimensions: PhysicalSize<f32>) -> Self {
        let direct = get_compute_pipeline(
            device.clone(),
            shaders.direct.clone(),
            &shaders::DirectSpecializationConstants {
                RATIO_X: FOV,
                RATIO_Y: -FOV * dimensions.height / dimensions.width,
            },
        ); // TODO: specialization constants here

        let buffer_rays = get_compute_pipeline(device.clone(), shaders.buffer_rays.clone(), &());

        let move_lightmaps = (0..LIGHTMAP_COUNT)
            .map(|i| {
                get_compute_pipeline(
                    device.clone(),
                    shaders.move_lightmap.clone(),
                    &shaders::MoveLightmapSpecializationConstants {
                        LIGHTMAP_INDEX: i as u32,
                    },
                )
            })
            .collect::<Vec<_>>();

        Self {
            direct,
            buffer_rays,
            move_lightmaps,
        }
    }
}
