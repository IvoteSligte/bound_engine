use wgpu::*;

use crate::{shader_objects::repeat_zero, *};

#[derive(Debug)]
pub(crate) struct ShaderObjects {
    pub(crate) real_time_buffer: RealTimeBufferWrapper,
    pub(crate) bvh: Buffer,
    pub(crate) color_image: Texture,
    pub(crate) lightmap_images: Vec<Texture>,
    pub(crate) lightmap_sync_images: Vec<Texture>,
    pub(crate) lightmap_buffers: [LightmapBufferSet; 2],
    pub(crate) spec_constants: Buffer,
}

impl ShaderObjects {
    pub(crate) fn new(device: &Device, queue: &Queue, screen_size: UVec2) -> Self {
        Self {
            real_time_buffer: RealTimeBufferWrapper::new(device),
            bvh: shader_objects::zeroed_buffer_sized::<shader_objects::GpuBvh>(device),
            color_image: ShaderObjects::zeroed_color_image(device, queue, screen_size),
            lightmap_images: ShaderObjects::zeroed_lightmap_images(device, queue),
            lightmap_sync_images: ShaderObjects::zeroed_sync_images(device, queue),
            lightmap_buffers: [
                LightmapBufferSet::new(device),
                LightmapBufferSet::new(device),
            ],
            spec_constants: shader_objects::zeroed_buffer_sized::<shader_objects::SpecConstants>(device),
        }
    }

    fn zeroed_color_image(device: &Device, queue: &Queue, size: UVec2) -> Texture {
        device.create_texture_with_data(
            queue,
            &TextureDescriptor {
                label: Some("color_image"),
                size: Extent3d {
                    width: size.x,
                    height: size.y,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::STORAGE_BINDING,
                view_formats: &[TextureFormat::Rgba16Float],
            },
            &repeat_zero((2 * 4 * size.x * size.y) as usize),
        )
    }

    fn zeroed_lightmap_images(device: &Device, queue: &Queue) -> Vec<Texture> {
        const FORMAT: TextureFormat = TextureFormat::Rgba16Float;
        let bytes_per_pixel = FORMAT.describe().block_size as usize;

        (0..(defines::RAYS_INDIRECT * defines::LIGHTMAP_COUNT))
            .map(|_| {
                device.create_texture_with_data(
                    queue,
                    &TextureDescriptor {
                        label: Some("color_image"),
                        size: Extent3d {
                            width: defines::LIGHTMAP_SIZE,
                            height: defines::LIGHTMAP_SIZE,
                            depth_or_array_layers: defines::LIGHTMAP_SIZE,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: TextureDimension::D3,
                        format: FORMAT,
                        usage: TextureUsages::STORAGE_BINDING,
                        view_formats: &[FORMAT],
                    },
                    &repeat_zero(bytes_per_pixel * defines::LIGHTMAP_SIZE.pow(3) as usize),
                )
            })
            .collect()
    }

    fn zeroed_sync_images(device: &Device, queue: &Queue) -> Vec<Texture> {
        const FORMAT: TextureFormat = TextureFormat::R32Uint;
        let bytes_per_pixel = FORMAT.describe().block_size as usize;

        (0..defines::LIGHTMAP_COUNT)
            .map(|_| {
                device.create_texture_with_data(
                    queue,
                    &TextureDescriptor {
                        label: Some("color_image"),
                        size: Extent3d {
                            width: defines::LIGHTMAP_SIZE,
                            height: defines::LIGHTMAP_SIZE,
                            depth_or_array_layers: defines::LIGHTMAP_SIZE,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: TextureDimension::D3,
                        format: FORMAT,
                        usage: TextureUsages::STORAGE_BINDING,
                        view_formats: &[FORMAT],
                    },
                    &repeat_zero(bytes_per_pixel * defines::LIGHTMAP_SIZE.pow(3) as usize),
                )
            })
            .collect()
    }
}

#[derive(Debug)]
pub(crate) struct LightmapBufferSet {
    pub(crate) buffer: Buffer,
    pub(crate) counter: Buffer,
}

impl LightmapBufferSet {
    pub(crate) fn new(device: &Device) -> Self {
        Self {
            buffer: unsafe {
                shader_objects::zeroed_buffer_sized::<shader_objects::LightmapBuffer>(device)
            },
            counter: unsafe {
                shader_objects::zeroed_buffer_sized::<shader_objects::LightmapCounter>(device)
            },
        }
    }
}

#[derive(Debug)]
pub(crate) struct RealTimeBufferWrapper {
    pub(crate) inner: Buffer,
    pub(crate) previous_rotation: Vec4,
    pub(crate) previous_position: Vec3,
}

impl RealTimeBufferWrapper {
    /// Initializes with all zeroes
    pub(crate) fn new(device: &Device) -> Self {
        Self {
            inner: unsafe {
                shader_objects::zeroed_buffer_sized::<shader_objects::RealTimeBuffer>(device)
            },
            previous_rotation: Vec4::ZERO,
            previous_position: Vec3::ZERO,
        }
    }
}
