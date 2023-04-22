#version 460

#include "includes_general.glsl"

layout(local_size_x = LM_SAMPLES, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) uniform restrict readonly RealTimeBuffer {
    vec4 rotation;
    vec4 previousRotation;
    vec3 position;
    vec3 previousPosition;
    ivec3 lightmapOrigin;
    ivec4 deltaLightmapOrigins[LM_COUNT];
    uint frame;
} rt;

layout(binding = 1) uniform restrict readonly MutableData {
    Material mats[MAX_MATERIALS];
} buf;

layout(binding = 2, rgba16) uniform restrict writeonly image3D[LM_COUNT] lmOutputColorImages;

layout(binding = 3) buffer restrict readonly LMBuffer {
    Voxel voxels[LM_SIZE * LM_SIZE * LM_SIZE * LM_COUNT];
} lmBuffer;

layout(binding = 4) uniform restrict readonly NoiseBuffer {
    vec4 dirs[gl_WorkGroupSize.x];
} noise;

layout(binding = 5) uniform sampler3D SDFImages[LM_COUNT];

layout(binding = 6, r32ui) uniform restrict readonly uimage3D materialImages[LM_COUNT];

#include "includes_trace_ray.glsl"

shared Material SharedMaterials[MAX_MATERIALS];

shared SharedStruct SharedData;
shared vec3 SharedColors[gl_WorkGroupSize.x];

void main() {
    if (gl_LocalInvocationID.x == 0) {
        SharedData = SharedStruct(lmBuffer.voxels[gl_WorkGroupID.x], rt.lightmapOrigin);
        SharedMaterials = buf.mats;
    }
    barrier();
    
    SharedStruct sData = SharedData;
    Voxel voxel = sData.voxel;

    vec4 randDir = noise.dirs[gl_LocalInvocationID.x];
    vec3 dir = normalize(voxel.normal + randDir.xyz);

    vec3 position = voxel.position + dir;
    // FIXME: the initial offset is roughly zero cause the voxel which intersects an edge is the starting point and first sample
    bool isHit = marchRay(position, dir, sData.lightmapOrigin); // bottleneck

    ivec4 lmIndexSample = lmIndexAtPos(position, sData.lightmapOrigin);
    uint material = imageLoad(materialImages[lmIndexSample.w], lmIndexSample.xyz).x;
    vec3 color = SharedMaterials[material].emittance; // TODO: copy materials to shared memory

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

        Material material = buf.mats[voxel.material];
        color = color * material.reflectance * (1.0 / float(LM_SAMPLES)) + material.emittance;

        imageStore(lmOutputColorImages[voxel.lmIndex.w], voxel.lmIndex.xyz, vec4(color, 0.0));
    }
}
