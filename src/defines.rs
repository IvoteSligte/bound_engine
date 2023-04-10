pub(crate) const MAX_OBJECTS: u32 = 256;
pub(crate) const MAX_MATERIALS: u32 = 256;

pub(crate) const RAYS_INDIRECT: u32 = 4;
pub(crate) const LIGHTMAP_COUNT: u32 = 6;
pub(crate) const LIGHTMAP_SIZE: u32 = 128;

pub(crate) const ITEM_COUNT: u32 = 65536;

pub(crate) const SUBBUFFER_COUNT: u32 = 256;
pub(crate) const SUBBUFFER_LENGTH: u32 = ITEM_COUNT / SUBBUFFER_COUNT; // even division

pub(crate) const ALL_ONES: u32 = 4294967295;
pub(crate) const BIT_USED: u32 = 1 << 30; // bit 31
pub(crate) const BITS_LEVEL: u32 = 63; // bits [0, 6]

pub(crate) const LM_UNIT_SIZE: f32 = 0.5;

mod includes {
    pub(crate) const GENERAL: &str = include_str!("../shaders/includes_general.glsl");
    pub(crate) const TRACE_RAY: &str = include_str!("../shaders/includes_trace_ray.glsl");
}

pub(crate) fn hashmap() -> naga::FastHashMap<String, String> {
    naga::FastHashMap::from_iter([
        ("MAX_OBJECTS".to_string(), MAX_OBJECTS.to_string()),
        ("MAX_MATERIALS".to_string(), MAX_MATERIALS.to_string()),
        ("RAYS_INDIRECT".to_string(), RAYS_INDIRECT.to_string()),
        ("LIGHTMAP_COUNT".to_string(), LIGHTMAP_COUNT.to_string()),
        ("LIGHTMAP_SIZE".to_string(), LIGHTMAP_SIZE.to_string()),
        ("ITEM_COUNT".to_string(), ITEM_COUNT.to_string()),
        ("SUBBUFFER_COUNT".to_string(), SUBBUFFER_COUNT.to_string()),
        ("SUBBUFFER_LENGTH".to_string(), SUBBUFFER_LENGTH.to_string()),
        ("ALL_ONES".to_string(), ALL_ONES.to_string()),
        ("BIT_USED".to_string(), BIT_USED.to_string()),
        ("BITS_LEVEL".to_string(), BITS_LEVEL.to_string()),
        ("LM_UNIT_SIZE".to_string(), LM_UNIT_SIZE.to_string()),
        // misc constants
        ("EPSILON".to_string(), "1e-5".to_string()),
        ("FLT_MAX".to_string(), "3.402823466e+38".to_string()),
        // included files
        ("#include=\"includes_general.glsl\"".to_string(), includes::GENERAL.to_string()),
        ("#include=\"includes_trace_ray.glsl\"".to_string(), includes::TRACE_RAY.to_string()),
    ])
}
