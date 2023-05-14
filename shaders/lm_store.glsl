#version 460

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

layout(binding = 1) uniform restrict readonly MutableData {
    Material mats[MAX_MATERIALS];
} buf;

layout(binding = 2, rgba16) uniform restrict image3D[LM_COUNT] lmColorImages;

layout(binding = 3) buffer restrict LmPointBuffer {
    LmPoint points[LM_MAX_POINTS];
} lmPointBuffer;

void main() {
    vec3 lmOrigin = rt.lightmapOrigin;

    LmPoint point = lmPointBuffer.points[gl_GlobalInvocationID.x];
    ivec4 lmIndex = lmIndexAtPos(point.position, lmOrigin);

    vec3 prevColor = imageLoad(lmColorImages[lmIndex.w], lmIndex.xyz).rgb;
    vec3 color = point.color / float(point.frameSamples);

    float sampleCount = min(point.sampleCount + 1.0, 1024.0);

    Material material = buf.mats[point.material];
    color = color * material.reflectance + material.emittance;
    color = mix(prevColor, color, 1.0 / sampleCount);

    imageStore(lmColorImages[lmIndex.w], lmIndex.xyz, vec4(color, 1.0));

    lmPointBuffer.points[gl_GlobalInvocationID.x].sampleCount = sampleCount;
    lmPointBuffer.points[gl_GlobalInvocationID.x].color = vec3(0.0);
    lmPointBuffer.points[gl_GlobalInvocationID.x].frameSamples = 0;
}