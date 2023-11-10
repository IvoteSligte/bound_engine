#version 460

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

#include "includes_general.glsl"

layout(binding = 0) uniform restrict readonly RealTimeBuffer {
    mat4 projection_view;
    vec3 position;
} rt;

// cleared every frame
layout(binding = 1) readonly buffer Grid {
    GridCell cells[CELLS][CELLS][CELLS];
} grid;

layout(binding = 2) buffer StaticParticles {
    StaticParticle particles[];
} staticParticles;

void main() {
    StaticParticle particle = staticParticles.particles[gl_GlobalInvocationID.x];
    uvec3 position;
    float reflectance;
    float energy;
    float emittance;

    unpackStaticParticle(particle, position, reflectance, energy, emittance);

    // energy that was dispersed previously
    energy *= 1.0 - (reflectance * ENERGY_DISPERSION);

    vec3 cellPosition = vec3(position / CELLS);
    // position within cells = position / (65536 / CELLS)
    // 65536 is the amount of bits used for position
    // 65536 / CELLS gives the precision within the cell
    ivec3 index = ivec3(vec3(position) * (1.0 / float(65536 / CELLS)));
    GridCell cell = grid.cells[index.x][index.y][index.z];

    if (cell.vector == vec3(0.0)) {
        return;
    }
    // the core of the cell is the average position of all particles
    // in the cell, weighted by their energy
    // (normalized)
    float coreEnergy = length(cell.vector) / float(cell.counter);

    energy += coreEnergy;
    energy += emittance;
    staticParticles.particles[gl_GlobalInvocationID.x].energy = energy;
}
