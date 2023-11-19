#version 460

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

#include "includes_general.glsl"

// cleared every frame
layout(binding = 0) readonly buffer Grid {
    GridCell cells[CELLS][CELLS][CELLS];
} grid;

layout(binding = 1) buffer DynamicParticles {
    DynamicParticle particles[DYN_PARTICLES];
} dynamicParticles;

layout(binding = 2, r32f) uniform readonly image3D energyGrid;

void main() {
    DynamicParticle particle = dynamicParticles.particles[gl_GlobalInvocationID.x];
    ivec3 position;
    vec3 direction;
    float energy;

    unpackDynamicParticle(particle, position, direction, energy);

    // energy that was dispersed previously
    energy -= energy * DYN_ENERGY_DISPERSION;
    
    // position within the cell
    vec3 cellPosition = vec3(position % (65536 / CELLS)) * (1.0 / float(65536 / CELLS));
    ivec3 index = ivec3(position / (65536 / CELLS));
    GridCell cell = grid.cells[index.x][index.y][index.z];
    
    position += ivec3(direction * DYN_MOVEMENT * float(65536 / CELLS));

    // out of bounds
    if (any(lessThan(position, ivec3(0))) || any(greaterThanEqual(position, ivec3(65536)))) {
        position = newParticlePosition(gl_GlobalInvocationID.x);
        energy = 0.0;
        particle = packDynamicParticle(position, -direction, energy);
        dynamicParticles.particles[gl_GlobalInvocationID.x] = particle;
        return;
    }

    // the core of the cell is the average position of all particles
    // in the cell relative to the cell, weighted by their energy
    float coreEnergy = imageLoad(energyGrid, index).x / float(cell.counter);
    vec3 corePosition = cell.position / (float(cell.counter) * coreEnergy + EPSILON2);
    vec3 coreDirection = cellPosition - corePosition;
    vec3 newDirection = direction * energy + coreDirection * coreEnergy;

    energy = length(newDirection);
    
    if (length(newDirection) != 0.0) {
        direction = newDirection / length(newDirection);
    }
    
    particle = packDynamicParticle(position, direction, energy);
    dynamicParticles.particles[gl_GlobalInvocationID.x] = particle;
}