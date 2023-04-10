use std::num::NonZeroU32;

use wgpu::*;

use crate::*;

#[derive(Debug)]
pub(crate) struct Pipelines {
    pub(crate) direct: ComputePipeline,
    pub(crate) buffer_rays: ComputePipeline,
}

impl Pipelines {
    pub(crate) fn new(device: &Device, shaders: &shaders::Shaders) -> Self {
        Self {
            direct: Pipelines::create_direct(device, shaders),
            buffer_rays: Pipelines::create_buffer_rays(device, shaders),
        }
    }

    fn create_direct(device: &Device, shaders: &shaders::Shaders) -> ComputePipeline {
        let label = Some("direct");
        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label,
            layout: Some(
                &device.create_pipeline_layout(&PipelineLayoutDescriptor {
                    label,
                    bind_group_layouts: &[&device.create_bind_group_layout(
                        &BindGroupLayoutDescriptor {
                            label,
                            entries: &[
                                BindGroupLayoutEntry {
                                    binding: 0,
                                    visibility: ShaderStages::COMPUTE,
                                    ty: BindingType::Buffer {
                                        ty: BufferBindingType::Uniform,
                                        has_dynamic_offset: false,
                                        min_binding_size: None, // TODO:
                                    },
                                    count: None,
                                },
                                BindGroupLayoutEntry {
                                    binding: 1,
                                    visibility: ShaderStages::COMPUTE,
                                    ty: BindingType::Buffer {
                                        ty: BufferBindingType::Uniform,
                                        has_dynamic_offset: false,
                                        min_binding_size: None, // TODO:
                                    },
                                    count: None,
                                },
                                BindGroupLayoutEntry {
                                    binding: 2,
                                    visibility: ShaderStages::COMPUTE,
                                    ty: BindingType::StorageTexture {
                                        access: StorageTextureAccess::WriteOnly,
                                        format: TextureFormat::Rgba16Float,
                                        view_dimension: TextureViewDimension::D2,
                                    },
                                    count: None,
                                },
                                BindGroupLayoutEntry {
                                    binding: 3,
                                    visibility: ShaderStages::COMPUTE,
                                    ty: BindingType::StorageTexture {
                                        access: StorageTextureAccess::ReadOnly,
                                        format: TextureFormat::Rgba16Float,
                                        view_dimension: TextureViewDimension::D3,
                                    },
                                    count: Some(
                                        NonZeroU32::new(
                                            defines::RAYS_INDIRECT * defines::LIGHTMAP_COUNT,
                                        )
                                        .unwrap(),
                                    ),
                                },
                                BindGroupLayoutEntry {
                                    binding: 4,
                                    visibility: ShaderStages::COMPUTE,
                                    ty: BindingType::StorageTexture {
                                        access: StorageTextureAccess::ReadWrite,
                                        format: TextureFormat::R32Uint,
                                        view_dimension: TextureViewDimension::D3,
                                    },
                                    count: Some(
                                        NonZeroU32::new(
                                            defines::LIGHTMAP_COUNT * defines::LIGHTMAP_COUNT,
                                        )
                                        .unwrap(),
                                    ),
                                },
                                BindGroupLayoutEntry {
                                    binding: 5,
                                    visibility: ShaderStages::COMPUTE,
                                    ty: BindingType::Buffer {
                                        ty: BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None, // TODO:
                                    },
                                    count: None,
                                },
                                BindGroupLayoutEntry {
                                    binding: 6,
                                    visibility: ShaderStages::COMPUTE,
                                    ty: BindingType::Buffer {
                                        ty: BufferBindingType::Storage { read_only: false },
                                        has_dynamic_offset: false,
                                        min_binding_size: None, // TODO:
                                    },
                                    count: None,
                                },
                                BindGroupLayoutEntry {
                                    binding: 7,
                                    visibility: ShaderStages::COMPUTE,
                                    ty: BindingType::Buffer {
                                        ty: BufferBindingType::Uniform,
                                        has_dynamic_offset: false,
                                        min_binding_size: None, // TODO:
                                    },
                                    count: None,
                                },
                            ],
                        },
                    )],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shaders.direct,
            entry_point: "main",
        })
    }

    fn create_buffer_rays(device: &Device, shaders: &shaders::Shaders) -> ComputePipeline {
        let label = Some("buffer_rays");
        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label,
            layout: Some(
                &device.create_pipeline_layout(&PipelineLayoutDescriptor {
                    label,
                    bind_group_layouts: &[&device.create_bind_group_layout(
                        &BindGroupLayoutDescriptor {
                            label,
                            entries: &[
                                BindGroupLayoutEntry {
                                    binding: 0,
                                    visibility: ShaderStages::COMPUTE,
                                    ty: BindingType::Buffer {
                                        ty: BufferBindingType::Uniform,
                                        has_dynamic_offset: false,
                                        min_binding_size: None, // TODO:
                                    },
                                    count: None,
                                },
                                BindGroupLayoutEntry {
                                    binding: 1,
                                    visibility: ShaderStages::COMPUTE,
                                    ty: BindingType::Buffer {
                                        ty: BufferBindingType::Uniform,
                                        has_dynamic_offset: false,
                                        min_binding_size: None, // TODO:
                                    },
                                    count: None,
                                },
                                BindGroupLayoutEntry {
                                    binding: 2,
                                    visibility: ShaderStages::COMPUTE,
                                    ty: BindingType::StorageTexture {
                                        access: StorageTextureAccess::WriteOnly,
                                        format: TextureFormat::Rgba16Float,
                                        view_dimension: TextureViewDimension::D2,
                                    },
                                    count: None,
                                },
                                BindGroupLayoutEntry {
                                    binding: 3,
                                    visibility: ShaderStages::COMPUTE,
                                    ty: BindingType::StorageTexture {
                                        access: StorageTextureAccess::ReadOnly,
                                        format: TextureFormat::Rgba16Float,
                                        view_dimension: TextureViewDimension::D3,
                                    },
                                    count: Some(
                                        NonZeroU32::new(
                                            defines::RAYS_INDIRECT * defines::LIGHTMAP_COUNT,
                                        )
                                        .unwrap(),
                                    ),
                                },
                                BindGroupLayoutEntry {
                                    binding: 4,
                                    visibility: ShaderStages::COMPUTE,
                                    ty: BindingType::StorageTexture {
                                        access: StorageTextureAccess::ReadWrite,
                                        format: TextureFormat::R32Uint,
                                        view_dimension: TextureViewDimension::D3,
                                    },
                                    count: Some(
                                        NonZeroU32::new(
                                            defines::LIGHTMAP_COUNT * defines::LIGHTMAP_COUNT,
                                        )
                                        .unwrap(),
                                    ),
                                },
                                BindGroupLayoutEntry {
                                    binding: 5,
                                    visibility: ShaderStages::COMPUTE,
                                    ty: BindingType::Buffer {
                                        ty: BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None, // TODO:
                                    },
                                    count: None,
                                },
                                BindGroupLayoutEntry {
                                    binding: 6,
                                    visibility: ShaderStages::COMPUTE,
                                    ty: BindingType::Buffer {
                                        ty: BufferBindingType::Storage { read_only: false },
                                        has_dynamic_offset: false,
                                        min_binding_size: None, // TODO:
                                    },
                                    count: None,
                                },
                                BindGroupLayoutEntry {
                                    binding: 7,
                                    visibility: ShaderStages::COMPUTE,
                                    ty: BindingType::Buffer {
                                        ty: BufferBindingType::Storage { read_only: false },
                                        has_dynamic_offset: false,
                                        min_binding_size: None, // TODO:
                                    },
                                    count: None,
                                },
                                BindGroupLayoutEntry {
                                    binding: 8,
                                    visibility: ShaderStages::COMPUTE,
                                    ty: BindingType::Buffer {
                                        ty: BufferBindingType::Storage { read_only: false },
                                        has_dynamic_offset: false,
                                        min_binding_size: None, // TODO:
                                    },
                                    count: None,
                                },
                            ],
                        },
                    )],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shaders.buffer_rays,
            entry_point: "main",
        })
    }
}
