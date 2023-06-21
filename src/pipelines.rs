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
    pub(crate) sdf: Arc<ComputePipeline>,
    pub(crate) radiance: [Arc<ComputePipeline>; 2],
    pub(crate) radiance_precalc: Arc<ComputePipeline>,
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

        let sdf = create_compute_pipeline(device.clone(), shaders.sdf.clone(), &());

        let radiance = [
            create_compute_pipeline(
                device.clone(),
                shaders.radiance.clone(),
                &shaders::RadianceSpecializationConstants {
                    CHECKERBOARD_OFFSET: 0,
                },
            ),
            create_compute_pipeline(
                device.clone(),
                shaders.radiance.clone(),
                &shaders::RadianceSpecializationConstants {
                    CHECKERBOARD_OFFSET: 1,
                },
            ),
        ];

        let radiance_precalc =
            create_compute_pipeline(device.clone(), shaders.radiance_precalc.clone(), &());

        Self {
            direct,
            sdf,
            radiance,
            radiance_precalc,
        }
    }
}
