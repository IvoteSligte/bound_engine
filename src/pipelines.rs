use crate::{LIGHTMAP_COUNT, shaders};
    use crate::shaders::Shaders;

    use vulkano::pipeline::ComputePipeline;

    use vulkano::device::Device;
    use vulkano::shader::{ShaderModule, SpecializationConstants};

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
    pub(crate) struct PathtracePipelines {
        pub(crate) direct: Arc<ComputePipeline>,
        pub(crate) buffer_rays: Arc<ComputePipeline>,
    }

    impl PathtracePipelines {
        pub(crate) fn from_shaders(device: Arc<Device>, shaders: Shaders) -> Self {
            let direct = get_compute_pipeline(device.clone(), shaders.direct.clone(), &());
            let buffer_rays =
                get_compute_pipeline(device.clone(), shaders.buffer_rays.clone(), &());

            Self {
                direct,
                buffer_rays,
            }
        }
    }

    pub(crate) fn get_move_lightmap_pipelines(
        device: &Arc<Device>,
        shaders: &Shaders,
    ) -> Vec<Arc<ComputePipeline>> {
        let lightmap_pipelines = (0..LIGHTMAP_COUNT)
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
        lightmap_pipelines
    }