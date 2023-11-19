use crate::shaders;

use glam::*;

// TODO: loading from file
pub fn load() -> (Vec<[f32; 4]>, Vec<u32>, Vec<Vec<shaders::StaticParticle>>) {
    const MATERIALS: &[Material] = &[
        Material {
            reflectance: Vec3::splat(0.0),
            emittance: Vec3::splat(1e2),
        },
        Material {
            reflectance: Vec3::splat(0.99),
            emittance: Vec3::splat(0.0),
        },
        Material {
            reflectance: Vec3::new(0.99, 0.0, 0.0),
            emittance: Vec3::splat(0.0),
        },
        Material {
            reflectance: Vec3::new(0.0, 0.99, 0.0),
            emittance: Vec3::splat(0.0),
        },
        Material {
            reflectance: Vec3::new(0.0, 0.0, 0.99),
            emittance: Vec3::splat(0.0),
        },
        Material {
            reflectance: Vec3::new(0.99, 0.99, 0.0),
            emittance: Vec3::splat(0.0),
        },
    ];

    let objects: Vec<Object> = vec![
        Object::cube(Vec3::new(0.0, 0.0, 20.0), 1.0, MATERIALS[0]),
        Object::cuboid(
            Vec3::new(0.0, 0.0, -20.0),
            Vec3::new(100.0, 100.0, 10.0),
            MATERIALS[1],
        ),
        Object::cube(Vec3::new(-10.0, 30.0, -5.0), 5.0, MATERIALS[2]),
        Object::cube(Vec3::new(35.0, 20.0, -3.0), 7.0, MATERIALS[3]),
        Object::cube(Vec3::new(20.0, -30.0, -7.0), 3.0, MATERIALS[4]),
        Object::cube(Vec3::new(20.0, 40.0, -4.0), 6.0, MATERIALS[5]),
    ];

    let (vertices, vertex_idxs, static_particles) =
        Object::flatten_parts(objects.into_iter().map(|obj| obj.into_parts()));

    (vertices, vertex_idxs, static_particles)
}

#[derive(Copy, Clone, Debug)]
pub struct Material {
    reflectance: Vec3,
    emittance: Vec3,
}

#[derive(Clone, Debug)]
pub struct Object {
    vertices: Vec<Vec3>,
    indices: Vec<u32>,
    particles: Vec<Vec<shaders::StaticParticle>>,
}

impl Object {
    #[rustfmt::skip]
    fn cube(position: Vec3, half_extent: f32, material: Material) -> Self {
        Self::cuboid(position, Vec3::splat(half_extent), material)
    }

    #[rustfmt::skip]
    fn cuboid(position: Vec3, half_extents: Vec3, material: Material) -> Self {
        // TODO: determine the amount of particles based on an upper limit

        // TODO: update the cropping every time the grid moves
        let min_position = (position - half_extents).max(-0.5 * Vec3::splat(shaders::CELLS as f32));
        let max_position = (position + half_extents).min(0.5 * Vec3::splat(shaders::CELLS as f32));
        let extents = max_position - min_position;
        let count = (extents * shaders::STATIC_PARTICLE_DENSITY).as_uvec3();
        let capacity = (count.x * count.y * count.z) as usize;
        let mut particle_positions = Vec::with_capacity(capacity);

        for x in 0..count.x {
            for y in 0..count.y {
                for z in 0..count.z {
                    let position = min_position + UVec3::new(x, y, z).as_vec3() / shaders::STATIC_PARTICLE_DENSITY;
                    particle_positions.push(position);
                }
            }
        }
        let mut static_particles = vec![Vec::with_capacity(capacity), Vec::with_capacity(capacity), Vec::with_capacity(capacity)];

        for i in 0..3 {
            // packed reflectance into the second 16 bits of a u32
            let reflectance = ((material.reflectance[i] * 65535.0).clamp(0.0, 65535.0) as u32) << 16;
            let emittance = material.emittance[i];

            for pos in particle_positions.clone() {
                let pos = pos * ((65536 / shaders::CELLS) as f32) + (65536.0 / 2.0);

                if pos.cmplt(Vec3::ZERO).any() || pos.cmpge(Vec3::splat(65536.0)).any() {
                    continue;
                }
                let upos = pos.as_uvec3();
                let data = [upos.x | (upos.y << 16), upos.z | reflectance];

                static_particles[i].push(shaders::StaticParticle {
                    data,
                    emittance,
                    energy: 0.0,
                });
            }
        }

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
            particles: static_particles,
        }
    }
}

impl Object {
    fn into_parts(self) -> (Vec<[f32; 4]>, Vec<u32>, Vec<Vec<shaders::StaticParticle>>) {
        (
            self.vertices
                .into_iter()
                .map(|v| v.extend(0.0).to_array())
                .collect(),
            self.indices,
            self.particles,
        )
    }

    fn flatten_parts<I>(iter: I) -> (Vec<[f32; 4]>, Vec<u32>, Vec<Vec<shaders::StaticParticle>>)
    where
        I: IntoIterator<Item = (Vec<[f32; 4]>, Vec<u32>, Vec<Vec<shaders::StaticParticle>>)>,
    {
        iter.into_iter().fold(
            (vec![], vec![], vec![vec![], vec![], vec![]]),
            |(mut acc_v, mut acc_i, mut acc_p), (v, is, p)| {
                acc_i.extend(is.into_iter().map(|i| acc_v.len() as u32 + i));
                acc_v.extend(v);
                acc_p[0].extend(p[0].clone());
                acc_p[1].extend(p[1].clone());
                acc_p[2].extend(p[2].clone());

                (acc_v, acc_i, acc_p)
            },
        )
    }
}
