#version 460

#include "includes_general.glsl"

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) uniform restrict readonly RealTimeBuffer {
    vec4 rotation;
    vec4 previousRotation;
    vec3 position;
    vec3 previousPosition;
    ivec3 lightmapOrigin; // TODO: different origin per layer
    uint noiseOffset;
    ivec4 deltaLightmapOrigins[LM_COUNT];
} rt;

layout(binding = 1) uniform restrict readonly MutableData {
    Material mats[MAX_MATERIALS];
} buf;

layout(binding = 2, rgba16) uniform restrict readonly image3D[LM_COUNT] lmInputColorImages;

layout(binding = 3, rgba16) uniform restrict writeonly image3D[LM_COUNT] lmOutputColorImages;

layout(binding = 4) buffer restrict readonly LMPointBuffer {
    LMPoint points[LM_MAX_POINTS];
} lmPointBuffer;

layout(binding = 5) uniform restrict readonly NoiseBuffer {
    vec4 dirs[NOISE_BUFFER_LENGTH];
} noise;

layout(binding = 6) uniform sampler3D SDFImages[LM_COUNT];

#include "includes_march_ray.glsl"


void main() {
    vec3 lmOrigin = rt.lightmapOrigin;

    LMPoint point = lmPointBuffer.points[gl_GlobalInvocationID.x];
    ivec4 lmIndex = ivec4(unpackBytesUint(point.lmIndex));

    vec4 randDir = noise.dirs[rt.noiseOffset];
    vec3 dir = normalize(point.normal + randDir.xyz); // TODO: ReSTIR

    float totalDist = LM_UNIT_SIZES[lmIndex.w];
    vec3 position = point.position;
    bool isHit = marchRay(position, dir, lmOrigin, 2e-2, 64, totalDist); // bottleneck

    ivec4 lmIndexSample = lmIndexAtPos(position, lmOrigin);
    vec3 color = imageLoad(lmInputColorImages[lmIndexSample.w], lmIndexSample.xyz).rgb;

    vec4 prevData = imageLoad(lmInputColorImages[lmIndex.w], lmIndex.xyz).rgba;
    prevData.w = min(prevData.w + 1.0, 1024.0);

    Material material = buf.mats[point.material];
    color = color * material.reflectance + material.emittance;
    color = mix(prevData.rgb, color, 1.0 / 1024.0); // TODO: mix factor = 1.0 / prevData.w

    imageStore(lmOutputColorImages[lmIndex.w], lmIndex.xyz, vec4(color, prevData.w));
}
