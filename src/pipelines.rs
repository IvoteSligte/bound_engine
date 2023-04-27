use crate::shaders::Shaders;
use crate::{shaders, FOV};

use vulkano::pipeline::ComputePipeline;

use vulkano::device::Device;
use vulkano::shader::{ShaderModule, SpecializationConstants};
use winit::dpi::PhysicalSize;
use winit::window::Window;

use std::sync::Arc;

pub(crate) fn create_compute_pipeline<Css>(
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
    pub(crate) lm_init: Arc<ComputePipeline>,
    pub(crate) lm_primary: Arc<ComputePipeline>,
    pub(crate) lm_secondary: Arc<ComputePipeline>,
}

impl Pipelines {
    pub(crate) fn from_shaders(device: Arc<Device>, shaders: Shaders, window: Arc<Window>) -> Self {
        let dimensions: PhysicalSize<f32> = window.inner_size().cast();

        // TODO: refactor into separate function
        let direct = create_compute_pipeline(
            device.clone(),
            shaders.direct.clone(),
            &shaders::DirectSpecializationConstants {
                RATIO_X: FOV,
                RATIO_Y: -FOV * dimensions.height / dimensions.width,
            },
        );

        let lm_init = create_compute_pipeline(device.clone(), shaders.lm_init.clone(), &());

        let lm_primary = create_compute_pipeline(device.clone(), shaders.lm_primary.clone(), &());

        let lm_secondary =
            create_compute_pipeline(device.clone(), shaders.lm_secondary.clone(), &());

        Self {
            direct,
            lm_init,
            lm_primary,
            lm_secondary,
        }
    }
}
