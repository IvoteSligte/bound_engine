use std::borrow::Cow;

use wgpu::*;

use crate::*;

#[derive(Debug)]
pub(crate) struct Shaders {
    pub(crate) direct: ShaderModule,
    pub(crate) buffer_rays: ShaderModule,
}

impl Shaders {
    pub(crate) fn new(device: &Device) -> Self {
        let direct = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("direct"),
            source: Shaders::glsl_to_naga(
                "direct.glsl",
                include_str!("../shaders/direct.glsl"), // TODO: fix this for builds?
                naga::ShaderStage::Compute,
                defines::hashmap(),
            ),
        });
        let buffer_rays = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("buffer_rays"),
            source: Shaders::glsl_to_naga(
                "buffer_rays.glsl",
                include_str!("../shaders/buffer_rays.glsl"), // TODO: fix this for builds?
                naga::ShaderStage::Compute,
                defines::hashmap(),
            ),
        });

        Shaders {
            direct,
            buffer_rays,
        }
    }

    fn glsl_to_naga<'a, 'b>(
        label: &'a str,
        shader: &'b str,
        stage: naga::ShaderStage,
        defines: naga::FastHashMap<String, String>,
    ) -> ShaderSource<'b> {
        #[derive(Debug)]
        #[allow(dead_code)]
        struct Error {
            kind: naga::front::glsl::ErrorKind,
            line_nr: u32,
            column_nr: u32,
            code: String,
            squi: String,
        }

        // Parse the given shader code and store its representation.
        let options = naga::front::glsl::Options {
            stage,
            defines: defines.clone(),
        };
        let mut parser = naga::front::glsl::Parser::default();
        let module = match parser.parse(&options, shader) {
            Ok(module) => module,
            Err(errors) => {
                panic!(
                    "Failed to parse shader '{}'!\nErrors: {:#?}",
                    label,
                    errors
                        .into_iter()
                        .map(|naga::front::glsl::Error { kind, meta }| {
                            let location = meta.location(shader);

                            let code = shader.lines().collect::<Vec<_>>()
                                [location.line_number as usize - 1]
                                .to_string();
                            let squiggles = " ".repeat(location.line_position as usize - 1)
                                + &"~".repeat(location.length as usize);

                            Error {
                                kind,
                                line_nr: location.line_number,
                                column_nr: location.line_position,
                                code,
                                squi: squiggles,
                            }
                        })
                        .collect::<Vec<_>>()
                );
            }
        };

        ShaderSource::Naga(Cow::Owned(module))
    }
}
