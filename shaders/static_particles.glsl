#version 460

#extension GL_EXT_shader_atomic_float: enable

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

#include "includes_general.glsl"

// cleared every frame
layout(binding = 0) writeonly buffer Grid {
    GridCell cells[CELLS][CELLS][CELLS];
} grid;

layout(binding = 1) readonly buffer StaticParticles {
    StaticParticle particles[];
} staticParticles;

void main() {
    StaticParticle particle = staticParticles.particles[gl_GlobalInvocationID.x];
    uvec3 position;
    float reflectance;
    float energy;
    float emittance;

    unpackStaticParticle(particle, position, reflectance, energy, emittance);

    vec3 cellPosition = vec3(position / CELLS);
    // energy is the weight
    vec3 vector = cellPosition * energy * reflectance * ENERGY_DISPERSION;
    // position within cells = position / (65536 / CELLS)
    // 65536 is the amount of bits used for position
    // 65536 / CELLS gives the precision within the cell
    ivec3 index = ivec3(vec3(position) * (1.0 / float(65536 / CELLS)));

    atomicAdd(grid.cells[index.x][index.y][index.z].counter, 1);
    atomicAdd(grid.cells[index.x][index.y][index.z].vector.x, vector.x);
    atomicAdd(grid.cells[index.x][index.y][index.z].vector.y, vector.y);
    atomicAdd(grid.cells[index.x][index.y][index.z].vector.z, vector.z);
}
