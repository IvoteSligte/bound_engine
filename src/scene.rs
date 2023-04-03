use crate::bvh::CpuNode;

use super::shaders::ty::*;

use glam::*;

pub(crate) const MATERIALS: [Material; 7] = [
    // dummy material
    Material {
        reflectance: [0.0; 3],
        emittance: [0.0; 3],
        _dummy0: [0u8; 4],
        _dummy1: [0u8; 4],
    },
    Material {
        reflectance: [0.99, 0.1, 0.1],
        emittance: [0.0; 3],
        _dummy0: [0u8; 4],
        _dummy1: [0u8; 4],
    },
    Material {
        reflectance: [0.1, 0.99, 0.1],
        emittance: [0.0; 3],
        _dummy0: [0u8; 4],
        _dummy1: [0u8; 4],
    },
    Material {
        reflectance: [0.99; 3],
        emittance: [0.0; 3],
        _dummy0: [0u8; 4],
        _dummy1: [0u8; 4],
    },
    Material {
        reflectance: [0.0; 3],
        emittance: [10.0; 3],
        _dummy0: [0u8; 4],
        _dummy1: [0u8; 4],
    },
    Material {
        reflectance: [0.7; 3],
        emittance: [0.0; 3],
        _dummy0: [0u8; 4],
        _dummy1: [0u8; 4],
    },
    Material {
        reflectance: [0.5; 3],
        emittance: [70.0, 0.0, 0.0],
        _dummy0: [0u8; 4],
        _dummy1: [0u8; 4],
    },
];

pub(crate) const BVH_OBJECTS: [CpuNode; 7] = [
    CpuNode {
        position: Vec3::new(-1000020.0, 0.0, 0.0),
        radius: 1e6,
        child: None,
        next: None,
        material: Some(1),
        parent: None,
    },
    CpuNode {
        position: Vec3::new(1000020.0, 0.0, 0.0),
        radius: 1e6,
        child: None,
        next: None,
        material: Some(2),
        parent: None,
    },
    // CpuNode {
    //     position: Vec3::new(0.0, -1000020.0, 0.0),
    //     radius: 1e6,
    //     child: None,
    //     next: None,
    //     material: Some(3),
    //     parent: None,
    // },
    // CpuNode {
    //     position: Vec3::new(0.0, 1000020.0, 0.0),
    //     radius: 1e6,
    //     child: None,
    //     next: None,
    //     material: Some(3),
    //     parent: None,
    // },
    CpuNode {
        position: Vec3::new(0.0, 0.0, -1000020.0),
        radius: 1e6,
        child: None,
        next: None,
        material: Some(3),
        parent: None,
    },
    CpuNode {
        position: Vec3::new(0.0, 0.0, 1000020.0),
        radius: 1e6,
        child: None,
        next: None,
        material: Some(3),
        parent: None,
    },
    CpuNode {
        position: Vec3::new(0.0, 0.0, 119.7),
        radius: 100.0,
        child: None,
        next: None,
        material: Some(4),
        parent: None,
    },
    CpuNode {
        position: Vec3::new(-3.0, 1.0, -16.0),
        radius: 4.0,
        child: None,
        next: None,
        material: Some(5),
        parent: None,
    },
    CpuNode {
        position: Vec3::new(8.0, 15.0, 0.0),
        radius: 2.0,
        child: None,
        next: None,
        material: Some(6),
        parent: None,
    },
];
