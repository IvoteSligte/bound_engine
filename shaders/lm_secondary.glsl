#version 460

#include "includes_general.glsl"

layout(local_size_x = LM_SAMPLES, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) uniform restrict readonly RealTimeBuffer {
    vec4 rotation;
    vec4 previousRotation;
    vec3 position;
    vec3 previousPosition;
    ivec3 lightmapOrigin;
    uint lightmapBufferOffset;
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

layout(binding = 5) buffer restrict LMMarchBuffer {
    float dists[LM_VOXELS_PER_FRAME][64];
} lmMarchBuffer;

layout(binding = 6) uniform restrict readonly NoiseBuffer {
    vec4 dirs[LM_SAMPLES];
    vec4 midDirs[64];
} noise;

layout(binding = 7) uniform sampler3D SDFImages[LM_COUNT];

#include "includes_march_ray.glsl"

shared SharedStruct SharedData;
shared vec3 SharedColors[gl_WorkGroupSize.x];

void main() {
    if (gl_LocalInvocationID.x == 0) {
        SharedData = SharedStruct(lmPointBuffer.points[rt.lightmapBufferOffset + gl_WorkGroupID.x], rt.lightmapOrigin);
    }
    barrier();
    
    SharedStruct sData = SharedData;
    LMPoint point = sData.point;
    ivec4 lmIndex = ivec4(unpackBytesUint(point.lmIndex));

    vec4 randDir = noise.dirs[gl_LocalInvocationID.x];
    vec3 dir = normalize(point.normal + randDir.xyz);

    float totalDist = lmMarchBuffer.dists[gl_WorkGroupID.x][gl_LocalInvocationID.x / (LM_SAMPLES / 64)];
    vec3 position = point.position;
    bool isHit = marchRay(position, dir, sData.lightmapOrigin, 2e-2, 16, totalDist); // bottleneck

    ivec4 lmIndexSample = lmIndexAtPos(position, sData.lightmapOrigin);
    vec3 color = imageLoad(lmInputColorImages[lmIndexSample.w], lmIndexSample.xyz).rgb;

    SharedColors[gl_LocalInvocationID.x] = color;

    barrier();

    if (gl_LocalInvocationID.x < 64) {
        for (uint i = 1; i < gl_WorkGroupSize.x / 64; i++) {
            color += SharedColors[i * 64 + gl_LocalInvocationID.x];
        }
        SharedColors[gl_LocalInvocationID.x] = color;
    }

    barrier();

    if (gl_LocalInvocationID.x == 0) {
        for (uint i = 1; i < 64; i++) {
            color += SharedColors[i];
        }

        Material material = buf.mats[point.material];
        color = color * material.reflectance * (1.0 / float(LM_SAMPLES)) + material.emittance;

        imageStore(lmOutputColorImages[lmIndex.w], lmIndex.xyz, vec4(color, 1.0));
    }
}
