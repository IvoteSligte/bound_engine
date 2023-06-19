#version 460

#extension GL_EXT_shader_atomic_float : enable

#include "includes_general.glsl"

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) buffer RadianceBuffer {
    Radiance radiances[LM_LAYERS][RADIANCE_SIZE][RADIANCE_SIZE][RADIANCE_SIZE];
    Material materials[LM_LAYERS][RADIANCE_SIZE][RADIANCE_SIZE][RADIANCE_SIZE];
} cache;

shared Material SharedMaterial;

// TODO: checkerboard rendering, emission, solid objects
void main() {
    const uint LAYER = gl_WorkGroupID.x / RADIANCE_SIZE;
    const uvec3 IIL = uvec3(gl_WorkGroupID.x % RADIANCE_SIZE, gl_WorkGroupID.yz); // index in layer

    vec3 direction = directFibonacciSphere(float(gl_LocalInvocationID.x));

    const float F = 0.15; // arbitrary value, determines how much energy moves to the sides instead of along the `direction`
    const float M = (1.0 / (1.0 + 3.0 * F));
    vec3 weights = direction * direction * M + (F * M);

    uvec3 index;
    for (uint i = 0; i < 3; i++) {
        index[i] = direction[i] >= 0.0 ? (max(IIL[i], 1) - 1) : (min(IIL[i], RADIANCE_SIZE - 2) + 1);
    }

    uint packedRadiance;
    vec4 radiance;
    vec3 color = vec3(0.0);

    if (index.x != IIL.x) {
        packedRadiance = cache.radiances[LAYER][index.x][IIL.y][IIL.z].packed[gl_LocalInvocationID.x];
        radiance = unpackUnorm4x8(packedRadiance);
        color += radiance.rgb / (radiance.w <= 0.0 ? 1.0 : radiance.w) * weights[0];
    }

    if (index.y != IIL.y) {
        packedRadiance = cache.radiances[LAYER][IIL.x][index.y][IIL.z].packed[gl_LocalInvocationID.x];
        radiance = unpackUnorm4x8(packedRadiance);
        color += radiance.rgb / (radiance.w <= 0.0 ? 1.0 : radiance.w) * weights[1];
    }

    if (index.z != IIL.z) {
        packedRadiance = cache.radiances[LAYER][IIL.x][IIL.y][index.z].packed[gl_LocalInvocationID.x];
        radiance = unpackUnorm4x8(packedRadiance);
        color += radiance.rgb / (radiance.w <= 0.0 ? 1.0 : radiance.w) * weights[2]; // TODO: just initialize radiance.w to 1.0 on the CPU, the check is unnecessary
    }

    color += cache.materials[LAYER][IIL.x][IIL.y][IIL.z].emittance;

    float d = clamp(1.0 / maximum(color), 0.0, 1.0); // TODO: maximum(color) == 0.0
    color *= d;

    uint packed = packUnorm4x8(vec4(color, d));
    cache.radiances[LAYER][IIL.x][IIL.y][IIL.z].packed[gl_LocalInvocationID.x] = packed;
}
