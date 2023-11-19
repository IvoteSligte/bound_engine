#version 460

#extension GL_EXT_shader_atomic_float: enable

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

#include "includes_general.glsl"

// cleared every frame
layout(binding = 0) buffer Grid {
    GridCell cells[CELLS][CELLS][CELLS];
} grid;

layout(binding = 1) readonly buffer DynamicParticles {
    DynamicParticle particles[DYN_PARTICLES];
} dynamicParticles;

layout(binding = 2, r32f) uniform image3D energyGrid;

void main() {
    DynamicParticle particle = dynamicParticles.particles[gl_GlobalInvocationID.x];
    ivec3 position;
    vec3 direction;
    float energy;

    unpackDynamicParticle(particle, position, direction, energy);

    vec3 cellPosition = vec3(position % (65536 / CELLS)) * (1.0 / float(65536 / CELLS));
    float energyWeight = energy * ENERGY_DISPERSION;
    vec3 corePosition = cellPosition * energyWeight;
    ivec3 index = ivec3(position / (65536 / CELLS));

    imageAtomicAdd(energyGrid, index, energyWeight);
    atomicAdd(grid.cells[index.x][index.y][index.z].counter, 1);
    atomicAdd(grid.cells[index.x][index.y][index.z].position.x, corePosition.x);
    atomicAdd(grid.cells[index.x][index.y][index.z].position.y, corePosition.y);
    atomicAdd(grid.cells[index.x][index.y][index.z].position.z, corePosition.z);
}