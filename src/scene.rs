use std::f32::consts::PI;

use crate::{
    bvh::{self},
    shaders,
};

use glam::*;

#[derive(Clone)]
pub(crate) struct CpuMaterial {
    reflectance: Vec3,
    emittance: Vec3,
}

impl From<shaders::ty::Material> for CpuMaterial {
    fn from(value: shaders::ty::Material) -> Self {
        Self {
            reflectance: Vec3::from_array(value.reflectance),
            emittance: Vec3::from_array(value.emittance),
        }
    }
}

impl Into<shaders::ty::Material> for CpuMaterial {
    fn into(self) -> shaders::ty::Material {
        shaders::ty::Material {
            reflectance: self.reflectance.to_array(),
            emittance: self.emittance.to_array(),
            _dummy0: [0; 4],
            _dummy1: [0; 4],
        }
    }
}

impl CpuMaterial {
    pub(crate) fn to_pod(vec: Vec<Self>) -> Vec<shaders::ty::Material> {
        vec.into_iter().map(|x| x.into()).collect()
    }
}

fn custom_materials() -> Vec<CpuMaterial> {
        let mut materials: Vec<CpuMaterial> = vec![
        CpuMaterial {
            reflectance: Vec3::splat(0.99),
            emittance: Vec3::splat(0.0),
        },
        CpuMaterial {
            reflectance: Vec3::splat(0.0),
            emittance: Vec3::splat(1.0),
        },
    ];

    for i in 1..10 {
        let color = if i < 4 {
            Vec3::new(i as f32 / 3.0, 0.0, 0.0)
        } else if i < 7 {
            Vec3::new(0.0, (i - 2) as f32 / 3.0, 0.0)
        } else {
            Vec3::new(0.0, 0.0, (i - 5) as f32 / 3.0)
        };

        materials.push(CpuMaterial {
            reflectance: Vec3::splat(0.1),
            emittance: color,
        });
    }

    materials
}

pub(crate) fn get_materials() -> Vec<shaders::ty::Material> {
    let mut materials = custom_materials();

    materials.insert(
        0, // dummy material
        CpuMaterial {
            reflectance: Vec3::ZERO,
            emittance: Vec3::ZERO,
        },
    );

    CpuMaterial::to_pod(materials)
}

#[derive(Clone)]
pub(crate) struct CpuObject {
    position: Vec3,
    radius: f32,
    material: usize,
}

impl From<bvh::CpuNode> for CpuObject {
    fn from(value: bvh::CpuNode) -> Self {
        Self {
            position: value.position,
            radius: value.radius,
            material: value.material.unwrap(),
        }
    }
}

impl Into<bvh::CpuNode> for CpuObject {
    fn into(self) -> bvh::CpuNode {
        bvh::CpuNode {
            position: self.position,
            radius: self.radius,
            material: Some(self.material),
            child: None,
            next: None,
            parent: None,
        }
    }
}

fn custom_objects() -> Vec<CpuObject> {
    let mut objects: Vec<CpuObject> = vec![
        CpuObject {
            position: Vec3::new(0.0, 0.0, -1e5),
            radius: 1e5-10.0,
            material: 0,
        },
        CpuObject {
            position: Vec3::new(0.0, 0.0, 20.0),
            radius: 10.0,
            material: 1
        },
    ];
    
    for q in 1..10 {
        for i in 0..9 {
            let angle = 2.0 * PI * (i as f32 / 9.0);
    
            objects.push(CpuObject {
                position: Vec3::new(angle.cos() * 40.0 * q as f32, angle.sin() * 40.0 * q as f32, 0.0),
                radius: 4.0,
                material: 2+i,
            });
        }
    }

    objects
}

pub(crate) fn get_objects() -> shaders::ty::GpuBVH {
    let objects = custom_objects();

    let mut bvh: bvh::CpuBVH =
        bvh::CpuBVH::from(<CpuObject as Into<bvh::CpuNode>>::into(objects[0].clone()));

    for n in objects[1..].iter() {
        bvh.merge_in_place(<CpuObject as Into<bvh::CpuNode>>::into(n.clone()).into());
    }

    bvh.into()
}
