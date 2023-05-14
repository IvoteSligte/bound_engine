#version 460

#extension GL_EXT_shader_atomic_float : enable

#include "includes_general.glsl"

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) uniform restrict readonly RealTimeBuffer {
    vec4 rotation;
    vec4 previousRotation;
    vec3 position;
    vec3 previousPosition;
    ivec3 lightmapOrigin; // TODO: different origin per layer
    ivec4 deltaLightmapOrigins[LM_COUNT];
    vec3 noiseDirection;
} rt;

layout(binding = 1, rgba16) uniform restrict readonly image3D[LM_COUNT] lmInputColorImages;

layout(binding = 2) buffer restrict LmPointBuffer {
    LmPoint points[LM_MAX_POINTS];
} lmPointBuffer;

layout(binding = 3) uniform sampler3D SDFImages[LM_COUNT];

layout(binding = 4, r32ui) uniform restrict readonly uimage3D lmPointIndexImages[LM_COUNT];

#include "includes_march_ray.glsl"

void main() {
    vec3 lmOrigin = rt.lightmapOrigin;
    vec3 randDir = rt.noiseDirection;

    LmPoint point = lmPointBuffer.points[gl_GlobalInvocationID.x];
    ivec4 lmIndex = lmIndexAtPos(point.position, lmOrigin);

    vec3 dir = normalize(point.normal + randDir); // TODO: reservoir based direction sampling

    float totalDist = LM_UNIT_SIZES[lmIndex.w];
    vec3 position = point.position;
    bool isHit = marchRay(position, dir, lmOrigin, 2e-2, 64, totalDist); // bottleneck

    vec3 prevColor = imageLoad(lmInputColorImages[lmIndex.w], lmIndex.xyz).rgb;

    ivec4 lmIndexSample = lmIndexAtPos(position, lmOrigin);
    vec3 color = imageLoad(lmInputColorImages[lmIndexSample.w], lmIndexSample.xyz).rgb;
    uint pointIndexSample = imageLoad(lmPointIndexImages[lmIndexSample.w], lmIndexSample.xyz).x;

    atomicAdd(lmPointBuffer.points[gl_GlobalInvocationID.x].color.x, color.x);
    atomicAdd(lmPointBuffer.points[gl_GlobalInvocationID.x].color.y, color.y);
    atomicAdd(lmPointBuffer.points[gl_GlobalInvocationID.x].color.z, color.z);
    atomicAdd(lmPointBuffer.points[gl_GlobalInvocationID.x].frameSamples, 1);

    atomicAdd(lmPointBuffer.points[pointIndexSample].color.x, prevColor.x);
    atomicAdd(lmPointBuffer.points[pointIndexSample].color.y, prevColor.y);
    atomicAdd(lmPointBuffer.points[pointIndexSample].color.z, prevColor.z);
    atomicAdd(lmPointBuffer.points[pointIndexSample].frameSamples, 1);
}
