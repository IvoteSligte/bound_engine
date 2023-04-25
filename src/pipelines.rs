use crate::shaders::{Shaders, LM_BUFFER_SLICES, LM_COUNT, LM_SIZE};
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
    pub(crate) lm_primary: Vec<Arc<ComputePipeline>>,
    pub(crate) lm_secondary: Vec<Arc<ComputePipeline>>,
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

        debug_assert!((LM_SIZE * LM_SIZE * LM_SIZE * LM_COUNT) % LM_BUFFER_SLICES == 0);
        const LM_SLICE_LEN: u32 = LM_SIZE * LM_SIZE * LM_SIZE * LM_COUNT / LM_BUFFER_SLICES;
        
        let offsets = (0..LM_BUFFER_SLICES).map(|x| x * LM_SLICE_LEN);

        let lm_primary = offsets.clone()
            .map(|offset| {
                create_compute_pipeline(
                    device.clone(),
                    shaders.lm_primary.clone(),
                    &shaders::LmPrimarySpecializationConstants {
                        LM_BUFFER_OFFSET: offset,
                    },
                )
            })
            .collect();

        let lm_secondary = offsets
            .map(|offset| {
                create_compute_pipeline(
                    device.clone(),
                    shaders.lm_secondary.clone(),
                    &shaders::LmPrimarySpecializationConstants {
                        LM_BUFFER_OFFSET: offset,
                    },
                )
            })
            .collect();

        Self {
            direct,
            lm_init,
            lm_primary,
            lm_secondary,
        }
    }
}
