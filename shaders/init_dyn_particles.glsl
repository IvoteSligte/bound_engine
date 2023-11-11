#version 460

#extension GL_EXT_shader_atomic_float: enable

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

#include "includes_general.glsl"

layout(binding = 0) buffer DynamicParticles {
    DynamicParticle particles[DYN_PARTICLES];
} dynamicParticles;

ivec3 randomPosition() {
    ivec3 p = ivec3(
        rand(gl_GlobalInvocationID.x),
        rand(gl_GlobalInvocationID.x + 111),
        rand(gl_GlobalInvocationID.x + 7547)
    );
    p %= 65536; // 16 bits storage
    return p;
}

void main() {
    ivec3 position = randomPosition();
    vec3 direction = vec3(1.0, 0.0, 0.0);
    float energy = 0.0;
    DynamicParticle particle = packDynamicParticle(position, direction, energy);
    dynamicParticles.particles[gl_GlobalInvocationID.x] = particle;
}