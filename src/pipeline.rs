use crate::shaders::Shaders;
use crate::{scene, shaders};

use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::{self};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{ComputePipeline, GraphicsPipeline};

use vulkano::device::Device;
use vulkano::render_pass::{RenderPass, Subpass};
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

pub fn graphics<Css>(
    device: Arc<Device>,
    dimensions: [f32; 2],
    render_pass: Arc<RenderPass>,
    vertex: Arc<ShaderModule>,
    spec_consts_vertex: Css,
    fragment: Arc<ShaderModule>,
    spec_consts_fragment: Css,
) -> Arc<GraphicsPipeline>
where
    Css: SpecializationConstants,
{
    let viewport = Viewport {
        origin: [0.0; 2],
        dimensions,
        depth_range: 0.0..1.0,
    };

    GraphicsPipeline::start()
        .vertex_input_state(<scene::Vertex as vertex_input::Vertex>::per_vertex())
        .vertex_shader(vertex.entry_point("main").unwrap(), spec_consts_vertex)
        .input_assembly_state(InputAssemblyState::default())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .fragment_shader(fragment.entry_point("main").unwrap(), spec_consts_fragment)
        .render_pass(Subpass::from(render_pass, 0).unwrap())
        .build(device)
        .unwrap()
}

#[derive(Clone)]
pub struct Pipelines {
    pub direct: Arc<GraphicsPipeline>,
    pub radiance: [Arc<ComputePipeline>; 2],
    pub radiance_precalc: Arc<ComputePipeline>,
}

impl Pipelines {
    pub fn new(
        device: Arc<Device>,
        shaders: Shaders,
        render_pass: Arc<RenderPass>,
        window: Arc<Window>,
    ) -> Self {
        let dimensions: PhysicalSize<f32> = window.inner_size().cast();

        let direct = graphics(
            device.clone(),
            [dimensions.width, dimensions.height],
            render_pass,
            shaders.direct.vertex.clone(),
            (),
            shaders.direct.fragment.clone(),
            (),
        );

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
            radiance,
            radiance_precalc,
        }
    }
}
