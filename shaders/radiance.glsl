#version 460

#extension GL_EXT_shader_atomic_float : enable

#include "includes_general.glsl"

layout(constant_id = 0) const int CHECKERBOARD_OFFSET = 0;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) buffer RadianceBuffer {
    Radiance radiances[LM_LAYERS][RADIANCE_SIZE][RADIANCE_SIZE][RADIANCE_SIZE];
    Material materials[LM_LAYERS][RADIANCE_SIZE][RADIANCE_SIZE][RADIANCE_SIZE];
} cache;

shared Material SharedMaterial;

// TODO: checkerboard rendering, occlusion, spherical harmonic radiance encoding
void main() {
    const int LAYER = int(gl_WorkGroupID.x / RADIANCE_SIZE);
    const ivec3 IIL = ivec3(
        gl_WorkGroupID.x % RADIANCE_SIZE,
        gl_WorkGroupID.y * 2 + (gl_WorkGroupID.x + gl_WorkGroupID.z + CHECKERBOARD_OFFSET) % 2, // checkerboard transform
        gl_WorkGroupID.z
    ); // index in layer

    vec3 direction = directFibonacciSphere(float(gl_LocalInvocationID.x));

    const float F = 0.15; // arbitrary value, determines how much energy moves to the sides instead of along the `direction`
    const float M = (1.0 / (1.0 + 3.0 * F));
    vec3 weights = direction * direction * M + (F * M);

    ivec3 index;
    for (uint i = 0; i < 3; i++) {
        index[i] = direction[i] >= 0.0 ? (max(IIL[i], 1) - 1) : (min(IIL[i], RADIANCE_SIZE - 2) + 1);
    }

    vec3 color = vec3(0.0);

    if (index.x != IIL.x) {
        uvec2 packedRadiance = cache.radiances[LAYER][index.x][IIL.y][IIL.z].packed[gl_LocalInvocationID.x];
        vec3 radiance = vec3(unpackHalf2x16(packedRadiance.x), unpackHalf2x16(packedRadiance.y).x);
        color += radiance * weights[0];
    }

    if (index.y != IIL.y) {
        uvec2 packedRadiance = cache.radiances[LAYER][IIL.x][index.y][IIL.z].packed[gl_LocalInvocationID.x];
        vec3 radiance = vec3(unpackHalf2x16(packedRadiance.x), unpackHalf2x16(packedRadiance.y).x);
        color += radiance * weights[1];
    }

    if (index.z != IIL.z) {
        uvec2 packedRadiance = cache.radiances[LAYER][IIL.x][IIL.y][index.z].packed[gl_LocalInvocationID.x];
        vec3 radiance = vec3(unpackHalf2x16(packedRadiance.x), unpackHalf2x16(packedRadiance.y).x);
        color += radiance * weights[2];
    }

    color += cache.materials[LAYER][IIL.x][IIL.y][IIL.z].emittance;

    uvec2 packed = uvec2(packHalf2x16(color.rg), packHalf2x16(vec2(color.b, 0.0)));
    cache.radiances[LAYER][IIL.x][IIL.y][IIL.z].packed[gl_LocalInvocationID.x] = packed;
}
