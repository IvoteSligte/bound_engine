use std::f32::consts::PI;

use crate::shaders::{self, MAX_MATERIALS, MAX_OBJECTS};

use glam::*;
use vulkano::{buffer::BufferContents, padded::Padded};

#[derive(Clone, Debug)]
pub(crate) struct CpuMaterial {
    reflectance: Vec3,
    emittance: Vec3,
}

impl Into<shaders::Material> for CpuMaterial {
    fn into(self) -> shaders::Material {
        shaders::Material {
            reflectance: self.reflectance.to_array().into(),
            emittance: self.emittance.to_array(),
        }
    }
}

fn custom_materials() -> Vec<CpuMaterial> {
    let materials: Vec<CpuMaterial> = vec![
        CpuMaterial {
            reflectance: Vec3::splat(0.99),
            emittance: Vec3::splat(0.0),
        },
        CpuMaterial {
            reflectance: Vec3::splat(0.0),
            emittance: Vec3::splat(1.0),
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

    materials
}

pub(crate) fn load_materials() -> Vec<shaders::Material> {
    let mut materials = custom_materials();

    materials.insert(
        0, // dummy material
        CpuMaterial {
            reflectance: Vec3::ZERO,
            emittance: Vec3::ZERO,
        },
    );

    materials.resize(
        MAX_MATERIALS,
        CpuMaterial {
            reflectance: Vec3::splat(0.0),
            emittance: Vec3::splat(0.0),
        },
    );

    materials.into_iter().map(|m| m.into()).collect()
}

#[derive(Clone, Debug)]
pub(crate) struct CpuObject {
    position: Vec3,
    radius: f32,
    material: usize,
}

impl From<shaders::Object> for CpuObject {
    fn from(value: shaders::Object) -> Self {
        Self {
            position: Vec3::from_array(value.position),
            radius: value.radius,
            material: value.material as usize - 1,
        }
    }
}

impl From<CpuObject> for shaders::Object {
    fn from(value: CpuObject) -> Self {
        Self {
            position: value.position.to_array(),
            radius: value.radius,
            material: value.material as u32 + 1,
        }
    }
}

fn custom_objects() -> Vec<CpuObject> {
    // TODO: triangle objects, voxel objects (?), loading from file, etc
    let mut objects: Vec<CpuObject> = vec![
        CpuObject {
            position: Vec3::new(0.0, 0.0, -1e5),
            radius: 1e5 - 10.0,
            material: 0,
        },
        CpuObject {
            position: Vec3::new(0.0, 0.0, 20.0),
            radius: 10.0,
            material: 1,
        },
    ];

    for q in 1..10 {
        for i in 0..9 {
            let angle = 2.0 * PI * (i as f32 / 9.0);

            objects.push(CpuObject {
                position: Vec3::new(
                    angle.cos() * 40.0 * q as f32,
                    angle.sin() * 40.0 * q as f32,
                    0.0,
                ),
                radius: 4.0,
                material: 2 + i / 3,
            });
        }
    }

    objects
}

pub(crate) fn load_objects() -> Vec<RawObject> {
    let mut objects = custom_objects();

    objects.resize(
        MAX_OBJECTS,
        CpuObject {
            position: Vec3::splat(0.0),
            radius: 0.0,
            material: 0,
        },
    );

    objects
        .into_iter()
        .map(|obj| obj.into())
        .collect::<Vec<_>>()
}

#[derive(Clone, Debug, BufferContents)]
#[repr(C)]
/// Required for proper alignment in the shader. shaders::Object does not work properly.
pub(crate) struct RawObject {
    position: [f32; 3],
    radius: f32,
    material: Padded<usize, 8>,
}

impl From<CpuObject> for RawObject {
    fn from(value: CpuObject) -> Self {
        Self {
            position: value.position.to_array(),
            radius: value.radius,
            material: (value.material + 1).into(),
        }
    }
}
