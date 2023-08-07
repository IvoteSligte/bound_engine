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
shared vec3[4] SharedCoefs;

// TODO: checkerboard rendering, occlusion
void main() {
    const int LAYER = int(gl_WorkGroupID.x / RADIANCE_SIZE);
    // index in layer
    const ivec3 IIL = ivec3(
        gl_WorkGroupID.x % RADIANCE_SIZE,
        gl_WorkGroupID.y * 2 + (gl_WorkGroupID.x + gl_WorkGroupID.z + CHECKERBOARD_OFFSET) % 2, // checkerboard transform
        gl_WorkGroupID.z);

    if (gl_LocalInvocationID.x < 4) {
        SharedCoefs[gl_LocalInvocationID.x] = vec3(0.0);
    }
    barrier();

    // stores coefs as array of RGB channels
    vec3[4] coefs = vec3[](vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0));

    // -X
    if (IIL.x != 0) {
        vec3[4] rCoefs = unpackSHCoefs(cache.radiances[LAYER][IIL.x-1][IIL.y][IIL.z].sh);
        for (int i = 0; i < 4; i++) {
            coefs[i] += SH_cosLobe_C0 * rCoefs[i];
        }
        coefs[3] += SH_cosLobe_C1 * rCoefs[3];
    }
    // +X
    if (IIL.x != RADIANCE_SIZE - 1) {
        vec3[4] rCoefs = unpackSHCoefs(cache.radiances[LAYER][IIL.x+1][IIL.y][IIL.z].sh);
        for (int i = 0; i < 4; i++) {
            coefs[i] += SH_cosLobe_C0 * rCoefs[i];
        }
        coefs[3] -= SH_cosLobe_C1 * rCoefs[3];
    }
    // -Y
    if (IIL.y != 0) {
        vec3[4] rCoefs = unpackSHCoefs(cache.radiances[LAYER][IIL.x][IIL.y-1][IIL.z].sh);
        for (int i = 0; i < 4; i++) {
            coefs[i] += SH_cosLobe_C0 * rCoefs[i];
        }
        coefs[1] += SH_cosLobe_C1 * rCoefs[1];
    }
    // +Y
    if (IIL.y != RADIANCE_SIZE - 1) {
        vec3[4] rCoefs = unpackSHCoefs(cache.radiances[LAYER][IIL.x][IIL.y+1][IIL.z].sh);
        for (int i = 0; i < 4; i++) {
            coefs[i] += SH_cosLobe_C0 * rCoefs[i];
        }
        coefs[1] -= SH_cosLobe_C1 * rCoefs[1];
    }
    if (IIL.z != 0) {
        vec3[4] rCoefs = unpackSHCoefs(cache.radiances[LAYER][IIL.x][IIL.y][IIL.z-1].sh);
        for (int i = 0; i < 4; i++) {
            coefs[i] += SH_cosLobe_C0 * rCoefs[i];
        }
        coefs[2] -= SH_cosLobe_C1 * rCoefs[2];
    }
    if (IIL.z != RADIANCE_SIZE - 1) {
        vec3[4] rCoefs = unpackSHCoefs(cache.radiances[LAYER][IIL.x][IIL.y][IIL.z+1].sh);
        for (int i = 0; i < 4; i++) {
            coefs[i] += SH_cosLobe_C0 * rCoefs[i];
        }
        coefs[2] += SH_cosLobe_C1 * rCoefs[2];
    }

    coefs[0] += cache.materials[LAYER][IIL.x][IIL.y][IIL.z].emittance;

    vec3 normal = vec3(0.0, 0.0, 1.0); // TODO: get (average?) normal in voxel from scene

    vec4 cosLobe = dirToCosineLobe(normal);

    vec4[3] tempCoefs = vec4[](
        cosLobe * dot(cosLobe, vec4(coefs[0].x, coefs[1].x, coefs[2].x, coefs[3].x)),
        cosLobe * dot(cosLobe, vec4(coefs[0].y, coefs[1].y, coefs[2].y, coefs[3].y)),
        cosLobe * dot(cosLobe, vec4(coefs[0].z, coefs[1].z, coefs[2].z, coefs[3].z))
    );

    coefs = vec3[](
        vec3(tempCoefs[0].x, tempCoefs[1].x, tempCoefs[2].x),
        vec3(tempCoefs[0].y, tempCoefs[1].y, tempCoefs[2].y),
        vec3(tempCoefs[0].z, tempCoefs[1].z, tempCoefs[2].z),
        vec3(tempCoefs[0].w, tempCoefs[1].w, tempCoefs[2].w)
    );

    for (int i = 0; i < 4; i++) {
        vec3 c = coefs[i] * (1.0 / 6.0);
        c /= 128.0; // TODO: use one global invocation per voxel instead of an entire workgroup
        atomicAdd(SharedCoefs[i].r, c.r);
        atomicAdd(SharedCoefs[i].g, c.g);
        atomicAdd(SharedCoefs[i].b, c.b);
    }
    barrier();

    if (gl_LocalInvocationID.x < 4) {
        cache.radiances[LAYER][IIL.x][IIL.y][IIL.z].sh[gl_LocalInvocationID.x] = packSHCoef(SharedCoefs[gl_LocalInvocationID.x]);
    }
}
