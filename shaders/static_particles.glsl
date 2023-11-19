#version 460

#extension GL_EXT_shader_atomic_float: enable

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

#include "includes_general.glsl"

// cleared every frame
layout(binding = 0) buffer Grid {
    GridCell cells[CELLS][CELLS][CELLS];
} grid;

layout(binding = 1) readonly buffer StaticParticles {
    StaticParticle particles[];
} staticParticles;

layout(binding = 2, r32f) uniform image3D energyGrid;

void main() {
    StaticParticle particle = staticParticles.particles[gl_GlobalInvocationID.x];
    uvec3 position;
    float reflectance;
    float energy;
    float emittance;

    unpackStaticParticle(particle, position, reflectance, energy, emittance);

    vec3 cellPosition = vec3(position % (65536 / CELLS)) * (1.0 / float(65536 / CELLS));
    float energyWeight = emittance + energy * reflectance * STATIC_ENERGY_DISPERSION;
    vec3 corePosition = cellPosition * energyWeight;
    ivec3 index = ivec3(position / (65536 / CELLS));

    imageAtomicAdd(energyGrid, index, energyWeight);
    atomicAdd(grid.cells[index.x][index.y][index.z].counter, STATIC_PARTICLE_WEIGHT);
    atomicAdd(grid.cells[index.x][index.y][index.z].position.x, corePosition.x);
    atomicAdd(grid.cells[index.x][index.y][index.z].position.y, corePosition.y);
    atomicAdd(grid.cells[index.x][index.y][index.z].position.z, corePosition.z);
}
