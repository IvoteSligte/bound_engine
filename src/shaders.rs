vulkano_shaders::shader! {
    shaders: {
        DirectVertex: {
            ty: "vertex",
            path: "shaders/direct.vert",
        },
        DirectFragment: {
            ty: "fragment",
            path: "shaders/direct.frag",
        },
        InitDynamicParticles: {
            ty: "compute",
            path: "shaders/init_dyn_particles.glsl",
        },
        ClearGrid: {
            ty: "compute",
            path: "shaders/clear_grid.glsl",
        },
        DynamicParticles: {
            ty: "compute",
            path: "shaders/dyn_particles.glsl",
        },
        DynamicParticles2: {
            ty: "compute",
            path: "shaders/dyn_particles2.glsl",
        },
        StaticParticles: {
            ty: "compute",
            path: "shaders/static_particles.glsl",
        },
        StaticParticles2: {
            ty: "compute",
            path: "shaders/static_particles2.glsl",
        },
    },
    custom_derives: [Copy, Clone, Debug],
    include: ["includes_general.glsl"],
    define: [
        ("DYN_PARTICLES_AXIS", "128"),
        ("DYN_PARTICLES", "2097152"),
        ("DYN_MOVEMENT", "0.5"),
        ("CELLS", "64"),
        ("ENERGY_DISPERSION", "0.5")
    ], // TODO: sync defines with consts
    vulkan_version: "1.2", // TODO: vulkan 1.3
    spirv_version: "1.6"
}

// 2^7 dynamic particles per axis
pub const DYN_PARTICLES_AXIS: u32 = 128;
// 2^21 dynamic particles
// must be a multiple of 2^8 = 64 (workgroup size)
pub const DYN_PARTICLES: u32 = DYN_PARTICLES_AXIS * DYN_PARTICLES_AXIS * DYN_PARTICLES_AXIS;
// movement per update, 1.0 = 1 cell per update
// pub const DYN_MOVEMENT: f32 = 0.5;
// 2^8 cells in a row (CELLS^3 total)
pub const CELLS: u32 = 64;
// how much of a particle's energy is dispersed
// to other particles
// pub const ENERGY_DISPERSION: f32 = 0.5;

// non-shader variables
// static particles / cell (1D)
pub const STATIC_PARTICLE_DENSITY: f32 = 2.0;

pub const TOTAL_CELLS: u32 = CELLS * CELLS * CELLS;

use vulkano::device::Device;

use vulkano::shader::ShaderModule;

use std::sync::Arc;

impl Default for GridCell {
    fn default() -> Self {
        Self {
            vector: [0.0; 3],
            counter: 0,
        }
    }
}

#[derive(Clone)]
pub struct Shaders {
    pub direct: DirectShaders,
    pub init_dynamic_particles: Arc<ShaderModule>,
    pub clear_grid: Arc<ShaderModule>,
    pub dynamic_particles: Arc<ShaderModule>,
    pub dynamic_particles2: Arc<ShaderModule>,
    pub static_particles: Arc<ShaderModule>,
    pub static_particles2: Arc<ShaderModule>,
}

impl Shaders {
    pub fn load(device: Arc<Device>) -> Self {
        Self {
            direct: DirectShaders::load(device.clone()),
            init_dynamic_particles: load_init_dynamic_particles(device.clone()).unwrap(),
            clear_grid: load_clear_grid(device.clone()).unwrap(),
            dynamic_particles: load_dynamic_particles(device.clone()).unwrap(),
            dynamic_particles2: load_dynamic_particles2(device.clone()).unwrap(),
            static_particles: load_static_particles(device.clone()).unwrap(),
            static_particles2: load_static_particles2(device.clone()).unwrap(),
        }
    }
}

#[derive(Clone)]
pub struct DirectShaders {
    pub vertex: Arc<ShaderModule>,
    pub fragment: Arc<ShaderModule>,
}

impl DirectShaders {
    fn load(device: Arc<Device>) -> Self {
        Self {
            vertex: load_direct_vertex(device.clone()).unwrap(),
            fragment: load_direct_fragment(device).unwrap(),
        }
    }
}
