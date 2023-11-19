#version 460

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

#include "includes_general.glsl"

// cleared every frame
layout(binding = 0) buffer Grid {
    GridCell cells[CELLS][CELLS][CELLS];
} grid;

layout(binding = 1) buffer StaticParticles {
    StaticParticle particles[];
} staticParticles;

layout(binding = 2, r32f) uniform readonly image3D energyGrid;

void main() {
    StaticParticle particle = staticParticles.particles[gl_GlobalInvocationID.x];
    uvec3 position;
    float reflectance;
    float energy;
    float emittance;

    unpackStaticParticle(particle, position, reflectance, energy, emittance);

    // energy that was dispersed previously
    energy -= energy * reflectance * STATIC_ENERGY_DISPERSION;
    
    ivec3 index = ivec3(position / (65536 / CELLS));
    uint cellCounter = grid.cells[index.x][index.y][index.z].counter;

    energy += imageLoad(energyGrid, index).x / float(cellCounter);
    staticParticles.particles[gl_GlobalInvocationID.x].energy = energy;
}
