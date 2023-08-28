use crate::shaders::{self, MAX_MATERIALS};

use glam::*;

// TODO: loading from file
pub fn load() -> (Vec<[f32; 4]>, Vec<u32>, Vec<u32>, Vec<shaders::Material>) {
    let mut materials = vec![
        CpuMaterial {
            reflectance: Vec3::splat(0.0),
            emittance: Vec3::splat(100.0),
        },
        CpuMaterial {
            reflectance: Vec3::splat(0.99),
            emittance: Vec3::splat(0.0),
        },
        CpuMaterial {
            reflectance: Vec3::new(0.99, 0.0, 0.0),
            emittance: Vec3::splat(0.0),
        },
        CpuMaterial {
            reflectance: Vec3::new(0.0, 0.99, 0.0),
            emittance: Vec3::splat(0.0),
        },
        CpuMaterial {
            reflectance: Vec3::new(0.0, 0.0, 0.99),
            emittance: Vec3::splat(0.0),
        },
        CpuMaterial {
            reflectance: Vec3::new(0.99, 0.99, 0.0),
            emittance: Vec3::splat(0.0),
        },
    ];

    materials.resize(
        MAX_MATERIALS,
        CpuMaterial {
            reflectance: Vec3::splat(0.0),
            emittance: Vec3::splat(0.0),
        },
    );

    let objects: Vec<CpuObject> = vec![
        CpuObject::cuboid(
            Vec3::new(0.0, 0.0, -20.0),
            Vec3::new(1000.0, 1000.0, 10.0),
            1,
        ),
        CpuObject::cube(Vec3::new(-10.0, 30.0, -5.0), 5.0, 2),
        CpuObject::cube(Vec3::new(35.0, 20.0, -3.0), 7.0, 3),
        CpuObject::cube(Vec3::new(20.0, -30.0, -7.0), 3.0, 4),
        CpuObject::cube(Vec3::new(20.0, 40.0, -4.0), 6.0, 5),
        CpuObject::cube(Vec3::new(0.0, 0.0, 20.0), 1.0, 0),
    ];

    let (vertices, vertex_idxs, material_idxs) =
        CpuObject::flatten_parts(objects.into_iter().map(|obj| obj.into_parts()));
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
        Self::cuboid(position, Vec3::splat(half_extent), material)
    }

    #[rustfmt::skip]
    fn cuboid(position: Vec3, half_extents: Vec3, material: u32) -> Self {
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
    fn into_parts(self) -> (Vec<[f32; 4]>, Vec<u32>, Vec<u32>) {
        (
            self.vertices
                .into_iter()
                .map(|v| v.extend(0.0).to_array())
                .collect(),
            self.indices,
            self.materials,
        )
    }

    fn flatten_parts<I>(iter: I) -> (Vec<[f32; 4]>, Vec<u32>, Vec<u32>)
    where
        I: IntoIterator<Item = (Vec<[f32; 4]>, Vec<u32>, Vec<u32>)>,
    {
        iter.into_iter().fold(
            (vec![], vec![], vec![]),
            |(mut acc_v, mut acc_i, mut acc_mi), (v, is, mi)| {
                acc_i.extend(is.into_iter().map(|i| acc_v.len() as u32 + i));
                acc_v.extend(v);
                acc_mi.extend(mi);

                (acc_v, acc_i, acc_mi)
            },
        )
    }
}
