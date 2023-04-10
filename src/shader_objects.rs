use crate::*;

pub(crate) fn repeat_zero(count: usize) -> Vec<u8> {
    ::core::iter::repeat(0u8)
        .take(count)
        .collect()
}

pub(crate) fn zeroed_bytes<T: Sized>() -> Vec<u8> {
    repeat_zero(::core::mem::size_of::<T>())
}

pub(crate) fn zeroed_buffer_sized<T: Sized>(device: &Device) -> Buffer {
    device.create_buffer_init(&BufferInitDescriptor {
        label: Some("real_time"),
        contents: &zeroed_bytes::<T>(),
        usage: BufferUsages::UNIFORM,
    })
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub(crate) struct Bounds {
    position: Vec3,
    radius_squared: f32,
    child: u32,
    next: u32,
    material: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub(crate) struct HitItem {
    position: Vec3,
    object_hit: u32,
    material_hit: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub(crate) struct Material {
    reflectance: Vec3,
    emittance: Vec3,
}

#[repr(C)] // FIXME: repr(align(16)) ?
#[derive(Copy, Clone, Debug)]
pub(crate) struct RealTimeBuffer {
    pub(crate) rotation: Vec4,
    pub(crate) position: Vec3,
    pub(crate) lightmap_origin: IVec3,
    pub(crate) delta_lightmap_origins: [IVec4; defines::LIGHTMAP_COUNT as usize],
}

#[repr(C)]
#[derive(Clone, Debug)]
pub(crate) struct GpuBvh {
    root: u32,
    nodes: [Bounds; 2 * defines::MAX_OBJECTS as usize],
}

#[repr(C)]
#[derive(Clone, Debug)]
pub(crate) struct MutableData {
    mats: [Material; defines::MAX_MATERIALS as usize],
}

#[repr(C)]
#[derive(Clone, Debug)]
pub(crate) struct LightmapBuffer {
    items: [HitItem; (defines::SUBBUFFER_COUNT * defines::SUBBUFFER_LENGTH) as usize],
}

#[repr(C)]
#[derive(Clone, Debug)]
pub(crate) struct LightmapCounter {
    counters: [u32; defines::SUBBUFFER_COUNT as usize],
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub(crate) struct SpecConstants {
    ratio: Vec2,
}
