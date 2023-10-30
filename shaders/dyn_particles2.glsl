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

layout(binding = 2) buffer Particles {
    DynamicParticle dynamic[DYN_PARTICLES];
    StaticParticle static_[];
} particles;

void main() {
    DynamicParticle particle = particles.dynamic[gl_GlobalInvocationID.x];
    ivec3 position;
    vec3 direction;
    float energy;

    unpackDynamicParticle(particle, position, direction, energy);

    position += ivec3(direction * DYN_MOVEMENT * float(CELLS));
    // energy that was dispersed previously
    energy *= 1.0 - ENERGY_DISPERSION;

    // out of bounds
    if (any(lessThan(position, ivec3(0))) || any(greaterThanEqual(position, ivec3(CELLS)))) {
        // TODO: new random position
        position = ivec3(0);
        energy = 0.0;
        particle = packDynamicParticle(position, direction, energy);
        particles.dynamic[gl_GlobalInvocationID.x] = particle;
        return;
    }

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
    vec3 corePosition = cell.vector / (float(cell.counter) * length(cell.vector));
    vec3 coreDifference = cellPosition - corePosition;

    if (coreDifference != vec3(0.0)) {
        float alpha = coreEnergy / (coreEnergy + energy);
        vec3 newDirection = mix(normalize(coreDifference), direction, alpha);

        if (newDirection != vec3(0.0)) {
            direction = normalize(newDirection);
        }
    }
    energy += coreEnergy;

    DynamicParticle newParticle = packDynamicParticle(position, direction, energy);
    particles.dynamic[gl_GlobalInvocationID.x] = newParticle;
}