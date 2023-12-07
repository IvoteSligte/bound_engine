#version 460

#extension GL_EXT_shader_atomic_float: enable

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

#include "includes_general.glsl"

layout(binding = 0) buffer DynamicParticles {
    DynamicParticle particles[DYN_PARTICLES];
} dynamicParticles;

void main() {
    ivec3 position = newParticlePosition(gl_GlobalInvocationID.x);
    vec3 direction = newParticleDirection(position);
    float energy = 0.0;
    DynamicParticle particle = packDynamicParticle(position, direction, energy);
    dynamicParticles.particles[gl_GlobalInvocationID.x] = particle;
}