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

void main() {
    DynamicParticle particle = dynamicParticles.particles[gl_GlobalInvocationID.x];
    ivec3 position;
    vec3 direction;
    float energy;

    unpackDynamicParticle(particle, position, direction, energy);

    // energy that was dispersed previously
    energy *= 1.0 - ENERGY_DISPERSION;
    
    vec3 cellPosition = vec3(position / CELLS);
    // position within cells = position / (65536 / CELLS)
    // 65536 is the amount of bits used for position
    // 65536 / CELLS gives the precision within the cell
    ivec3 index = ivec3(vec3(position) * (1.0 / float(65536 / CELLS)));
    GridCell cell = grid.cells[index.x][index.y][index.z];
    
    position += ivec3(direction * DYN_MOVEMENT * float(65536 / CELLS));

    // out of bounds
    if (any(lessThan(position, ivec3(0))) || any(greaterThanEqual(position, ivec3(65536)))) {
        position = newParticlePosition(gl_GlobalInvocationID.x);
        energy = 0.0;
        particle = packDynamicParticle(position, direction, energy);
        dynamicParticles.particles[gl_GlobalInvocationID.x] = particle;
        return;
    }
    if (cell.vector != vec3(0.0)) {
        // the core of the cell is the average position of all particles
        // in the cell relative to the cell, weighted by their energy
        float coreEnergy = length(cell.vector) / float(cell.counter); // FIXME: length(cell.vector) is not accurate as the position within the cell is not a normalized vector
        vec3 corePosition = cell.vector / (float(cell.counter) * length(cell.vector));
        vec3 coreDifference = cellPosition - corePosition;

        if (coreDifference == vec3(0.0)) {
            coreDifference = direction;
        }
        float alpha = coreEnergy / (coreEnergy + energy);
        vec3 newDirection = mix(normalize(coreDifference), direction, alpha);

        if (newDirection != vec3(0.0)) {
            direction = normalize(newDirection);
        }
        energy += coreEnergy;
    }
    particle = packDynamicParticle(position, direction, energy);
    dynamicParticles.particles[gl_GlobalInvocationID.x] = particle;
}