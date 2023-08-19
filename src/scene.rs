use std::f32::consts::PI;

use crate::shaders::{self, MAX_MATERIALS};

use glam::*;
use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input};

// TODO: loading from file
pub fn load() -> (Vec<Vertex>, Vec<u32>, Vec<u32>, Vec<shaders::Material>) {
    let mut materials = vec![
        CpuMaterial {
            reflectance: Vec3::splat(0.99),
            emittance: Vec3::splat(0.0),
        },
        CpuMaterial {
            reflectance: Vec3::splat(0.0),
            emittance: Vec3::splat(10.0),
        },
        CpuMaterial {
            reflectance: Vec3::splat(0.0),
            emittance: Vec3::new(1.0, 0.0, 0.0),
        },
        CpuMaterial {
            reflectance: Vec3::splat(0.0),
            emittance: Vec3::new(0.0, 1.0, 0.0),
        },
        CpuMaterial {
            reflectance: Vec3::splat(0.0),
            emittance: Vec3::new(0.0, 0.0, 1.0),
        },
    ];

    materials.resize(
        MAX_MATERIALS,
        CpuMaterial {
            reflectance: Vec3::splat(0.0),
            emittance: Vec3::splat(0.0),
        },
    );

    let mut objects: Vec<CpuObject> = vec![
        CpuObject::cube(Vec3::new(0.0, 0.0, -1e5), 1e5 - 10.0, 0),
        CpuObject::cube(Vec3::new(0.0, 0.0, 20.0), 1.0, 1),
    ];

    for q in 1..10 {
        for i in 0..9 {
            let angle = 2.0 * PI * (i as f32 / 9.0);

            objects.push(CpuObject::cube(
                Vec3::new(
                    angle.cos() * 40.0 * q as f32,
                    angle.sin() * 40.0 * q as f32,
                    0.0,
                ),
                4.0,
                2 + i / 3,
            ));
        }
    }

    objects.push(CpuObject::rectangle(Vec3::new(50.0, 0.0, 5.0), Vec3::new(1.0, 30.0, 10.0), 0));

    let (vertices, vertex_idxs, material_idxs): (Vec<_>, Vec<_>, Vec<_>) =
        objects.into_iter().fold(
            (vec![], vec![], vec![]),
            |(mut acc_v, mut acc_i, mut acc_mi), obj| {
                let (v, is, mi) = obj.into_renderable();

                acc_i.extend(is.into_iter().map(|i| acc_v.len() as u32 + i));
                acc_v.extend(v);
                acc_mi.extend(mi);

                (acc_v, acc_i, acc_mi)
            },
        );
    let materials = materials.into_iter().map(|mat| mat.into()).collect();

    (vertices, vertex_idxs, material_idxs, materials)
}

#[derive(Clone, Debug)]
pub struct CpuMaterial {
    reflectance: Vec3,
    emittance: Vec3,
}

impl From<CpuMaterial> for shaders::Material {
    fn from(value: CpuMaterial) -> Self {
        Self {
            reflectance: value.reflectance.to_array().into(),
            emittance: value.emittance.to_array(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CpuObject {
    vertices: Vec<Vec3>,
    indices: Vec<u32>,
    materials: Vec<u32>,
}

impl CpuObject {
    #[rustfmt::skip]
    fn cube(position: Vec3, half_extent: f32, material: u32) -> Self {
        Self::rectangle(position, Vec3::splat(half_extent), material)
    }

    #[rustfmt::skip]
    fn rectangle(position: Vec3, half_extents: Vec3, material: u32) -> Self {
        let Vec3 { x: xr, y: yr, z: zr } = half_extents;
        Self {
            vertices: vec![
                position + Vec3::new(-xr, -yr,  zr),
                position + Vec3::new( xr, -yr,  zr),
                position + Vec3::new( xr,  yr,  zr),
                position + Vec3::new(-xr,  yr,  zr),
                position + Vec3::new(-xr, -yr, -zr),
                position + Vec3::new( xr, -yr, -zr),
                position + Vec3::new( xr,  yr, -zr),
                position + Vec3::new(-xr,  yr, -zr),
            ],
            indices: vec![
                // Front face
                0, 1, 2,
                2, 3, 0,
                // Right face
                1, 5, 6,
                6, 2, 1,
                // Back face
                7, 6, 5,
                5, 4, 7,
                // Left face
                4, 0, 3,
                3, 7, 4,
                // Bottom face
                4, 5, 1,
                1, 0, 4,
                // Top face
                3, 2, 6,
                6, 7, 3,
            ],
            materials: vec![material; 12],
        }
    }
}

impl CpuObject {
    fn into_renderable(self) -> (Vec<Vertex>, Vec<u32>, Vec<u32>) {
        let vertices = self
            .vertices
            .into_iter()
            .map(|v| Vertex {
                position: v.extend(0.0).to_array(),
            })
            .collect();

        (vertices, self.indices, self.materials)
    }
}

#[derive(Debug, BufferContents, vertex_input::Vertex)]
#[repr(C)]
pub struct Vertex {
    #[format(R32G32B32A32_SFLOAT)]
    position: [f32; 4],
}
