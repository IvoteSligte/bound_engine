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
    vec4 denoiseRotation; // FIXME: update this CPU-side
    vec3 noiseDirection;
} rt;

layout(binding = 1) uniform restrict readonly MutableData {
    Material mats[MAX_MATERIALS];
} buf;

layout(binding = 2) uniform sampler3D[LM_COUNT] lmInputColorTextures;

layout(binding = 3) buffer restrict LmPointBuffer {
    LmPoint points[LM_MAX_POINTS];
} lmPointBuffer;

vec3 rotateVec3ByQuat(vec3 v, vec4 q) {
    vec3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

void main() {
    vec3 lmOrigin = rt.lightmapOrigin;
    vec4 denoiseRotation = rt.denoiseRotation;

    LmPoint point = lmPointBuffer.points[gl_GlobalInvocationID.x];
    ivec4 lmIndex = lmIndexAtPos(point.position, lmOrigin);

    vec3 prevColor = texelFetch(lmInputColorTextures[lmIndex.w], lmIndex.xyz, 0).rgb;
    vec3 color = point.color;

    vec3 positionSample = point.position + rotateVec3ByQuat(point.normal, denoiseRotation) * LM_UNIT_SIZES[lmIndex.w];

    int lmLayerSample = lmLayerAtPos(positionSample, lmOrigin); // TODO: lmOrigin varying between layers
    float mult = MULTS[lmLayerSample];
    vec3 texIdxSample = (positionSample - lmOrigin) * mult + 0.5; // TODO: lmOrigin varying between layers
    
    vec4 sampleColor = texture(lmInputColorTextures[lmLayerSample], texIdxSample);
    sampleColor.rgb /= sampleColor.w;

    float sampleCount = min(point.sampleCount + 1.0, 1024.0);

    Material material = buf.mats[point.material];
    color /= float(point.frameSamples);
    color = color * material.reflectance + material.emittance;
    color = mix(prevColor, color, 1.0 / sampleCount);

    if (sampleColor.w > 0.0) { // TODO: remove artifacts (light leaking) around lm image borders
        const float MIX_FACTOR = 1.0 / 32.0;
        color = mix(color, sampleColor.rgb, MIX_FACTOR);
    }

    lmPointBuffer.points[gl_GlobalInvocationID.x].sampleCount = sampleCount;
    lmPointBuffer.points[gl_GlobalInvocationID.x].color = color;
    lmPointBuffer.points[gl_GlobalInvocationID.x].frameSamples = 0;
}