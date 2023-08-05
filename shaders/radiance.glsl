#version 460

#extension GL_EXT_shader_atomic_float : enable

#include "includes_general.glsl"
#include "sh_rotation.glsl"

layout(constant_id = 0) const int CHECKERBOARD_OFFSET = 0;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) buffer RadianceBuffer {
    Radiance radiances[LM_LAYERS][RADIANCE_SIZE][RADIANCE_SIZE][RADIANCE_SIZE];
    Material materials[LM_LAYERS][RADIANCE_SIZE][RADIANCE_SIZE][RADIANCE_SIZE];
} cache;

shared Material SharedMaterial;
shared vec3[9] SharedCoefs;

// TODO: checkerboard rendering, occlusion
void main() {
    const int LAYER = int(gl_WorkGroupID.x / RADIANCE_SIZE);
    // index in layer
    const ivec3 IIL = ivec3(
        gl_WorkGroupID.x % RADIANCE_SIZE,
        gl_WorkGroupID.y * 2 + (gl_WorkGroupID.x + gl_WorkGroupID.z + CHECKERBOARD_OFFSET) % 2, // checkerboard transform
        gl_WorkGroupID.z);

    if (gl_LocalInvocationID.x < 9) {
        SharedCoefs[gl_LocalInvocationID.x] = vec3(0.0);
    }
    barrier();

    vec3[9] coefs = vec3[](vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0));

    // TODO: diffuse reflections etc

    if (IIL.x != 0) {
        vec3[9] rCoefs = unpackSHCoefs(cache.radiances[LAYER][IIL.x-1][IIL.y][IIL.z].sh);
        for (int i = 0; i < 9; i++) {
            coefs[i] += rCoefs[i];
        }
    }
    if (IIL.y != 0) {
        vec3[9] rCoefs = unpackSHCoefs(cache.radiances[LAYER][IIL.x][IIL.y-1][IIL.z].sh);
        for (int i = 0; i < 9; i++) {
            coefs[i] += rCoefs[i];
        }
    }
    if (IIL.z != 0) {
        vec3[9] rCoefs = unpackSHCoefs(cache.radiances[LAYER][IIL.x][IIL.y][IIL.z-1].sh);
        for (int i = 0; i < 9; i++) {
            coefs[i] += rCoefs[i];
        }
    }

    if (IIL.x != RADIANCE_SIZE - 1) {
        vec3[9] rCoefs = unpackSHCoefs(cache.radiances[LAYER][IIL.x+1][IIL.y][IIL.z].sh);
        for (int i = 0; i < 9; i++) {
            coefs[i] += rCoefs[i];
        }
    }
    if (IIL.y != RADIANCE_SIZE - 1) {
        vec3[9] rCoefs = unpackSHCoefs(cache.radiances[LAYER][IIL.x][IIL.y+1][IIL.z].sh);
        for (int i = 0; i < 9; i++) {
            coefs[i] += rCoefs[i];
        }
    }
    if (IIL.z != RADIANCE_SIZE - 1) {
        vec3[9] rCoefs = unpackSHCoefs(cache.radiances[LAYER][IIL.x][IIL.y][IIL.z+1].sh);
        for (int i = 0; i < 9; i++) {
            coefs[i] += rCoefs[i];
        }
    }

    coefs[0] += cache.materials[LAYER][IIL.x][IIL.y][IIL.z].emittance;

    for (int i = 0; i < 9; i++) {
        vec3 c = coefs[i] * (1.0 / 6.0);
        atomicAdd(SharedCoefs[i].x, c.x);
        atomicAdd(SharedCoefs[i].y, c.y);
        atomicAdd(SharedCoefs[i].z, c.z);
    }
    barrier();

    if (gl_LocalInvocationID.x < 9) {
        cache.radiances[LAYER][IIL.x][IIL.y][IIL.z].sh[gl_LocalInvocationID.x] = packSHCoef(SharedCoefs[gl_LocalInvocationID.x]);
    }
}
