use crate::shaders::Shaders;
use crate::{shaders, FOV};

use vulkano::pipeline::ComputePipeline;

use vulkano::device::Device;
use vulkano::shader::{ShaderModule, SpecializationConstants};
use winit::dpi::PhysicalSize;
use winit::window::Window;

use std::sync::Arc;

pub fn compute<Css>(
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
pub struct Pipelines {
    pub direct: Arc<ComputePipeline>,
    pub sdf: Arc<ComputePipeline>,
    pub radiance: [Arc<ComputePipeline>; 2],
    pub radiance_precalc: Arc<ComputePipeline>,
}

impl Pipelines {
    pub fn new(device: Arc<Device>, shaders: Shaders, window: Arc<Window>) -> Self {
        let dimensions: PhysicalSize<f32> = window.inner_size().cast();

        // TODO: refactor into separate function
        let direct = compute(
            device.clone(),
            shaders.direct.clone(),
            &shaders::DirectSpecializationConstants {
                RATIO_X: FOV,
                RATIO_Y: -FOV * dimensions.height / dimensions.width,
            },
        );

        let sdf = compute(device.clone(), shaders.sdf.clone(), &());

        let radiance = [
            compute(
                device.clone(),
                shaders.radiance.clone(),
                &shaders::RadianceSpecializationConstants {
                    CHECKERBOARD_OFFSET: 0,
                },
            ),
            compute(
                device.clone(),
                shaders.radiance.clone(),
                &shaders::RadianceSpecializationConstants {
                    CHECKERBOARD_OFFSET: 1,
                },
            ),
        ];

        let radiance_precalc = compute(device.clone(), shaders.radiance_precalc.clone(), &());

        Self {
            direct,
            sdf,
            radiance,
            radiance_precalc,
        }
    }
}
