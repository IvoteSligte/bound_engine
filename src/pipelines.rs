use crate::shaders::{Shaders, LIGHTMAP_COUNT};
use crate::{shaders, FOV};

use vulkano::pipeline::ComputePipeline;

use vulkano::device::Device;
use vulkano::shader::{ShaderModule, SpecializationConstants};
use winit::dpi::PhysicalSize;
use winit::window::Window;

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
    pub(crate) lightmap_rays: Vec<Arc<ComputePipeline>>,
}

impl Pipelines {
    pub(crate) fn from_shaders(device: Arc<Device>, shaders: Shaders, window: Arc<Window>) -> Self {
        let dimensions: PhysicalSize<f32> = window.inner_size().cast();

        let direct = get_compute_pipeline(
            device.clone(),
            shaders.direct.clone(),
            &shaders::DirectSpecializationConstants {
                RATIO_X: FOV,
                RATIO_Y: -FOV * dimensions.height / dimensions.width,
            },
        ); // TODO: specialization constants here

        let lightmap_rays = (0..LIGHTMAP_COUNT)
            .map(|i| {
                get_compute_pipeline(
                    device.clone(),
                    shaders.lightmap_rays.clone(),
                    &shaders::LightmapRaysSpecializationConstants {
                        LIGHTMAP_INDEX: i as i32,
                    },
                )
            })
            .collect();

        Self {
            direct,
            lightmap_rays,
        }
    }
}
