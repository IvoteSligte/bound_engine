#version 460

#include "includes_general.glsl"

layout(local_size_x = LM_SAMPLES / 4, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) uniform restrict readonly RealTimeBuffer {
    vec4 rotation;
    vec4 previousRotation;
    vec3 position;
    vec3 previousPosition;
    ivec3 lightmapOrigin;
    ivec4 deltaLightmapOrigins[LM_COUNT];
    uint frame;
} rt;

layout(binding = 1) uniform restrict readonly GpuBVH {
    uint root;
    Bounds nodes[2 * MAX_OBJECTS];
} bvh;

layout(binding = 2) uniform restrict readonly MutableData {
    Material mats[MAX_MATERIALS];
} buf;

layout(binding = 3, rgba16) uniform restrict writeonly image3D[LM_COUNT] lmOutputColorImages;

layout(binding = 4) buffer restrict readonly LMBuffer {
    Voxel voxels[LM_SIZE * LM_SIZE * LM_SIZE * LM_COUNT];
} lmBuffer;

layout(binding = 5) uniform restrict readonly BlueNoise {
    vec4 items[LM_SAMPLES];
} bn;

#include "includes_trace_ray.glsl"

/// returns an index into a lightmap image in xyz, and the image index in w
ivec4 lightmapIndexAtPos(vec3 v) {
    const int HALF_LM_SIZE = LM_SIZE / 2;
    const float INV_HALF_LM_SIZE = 1.0 / (float(HALF_LM_SIZE) * LM_UNIT_SIZE);

    v -= rt.lightmapOrigin.xyz;
    uint lightmapNum = uint(log2(max(maximum(abs(v)) * INV_HALF_LM_SIZE, 0.500001)) + 1.0);

    ivec3 index = ivec3(floor(v / LM_UNIT_SIZES[lightmapNum])) + HALF_LM_SIZE;

    return ivec4(index, lightmapNum);
}

vec3 posAtLightmapIndex(ivec4 lmIndex) {
    const int HALF_LM_SIZE = LM_SIZE / 2;

    vec3 v = (lmIndex.xyz - (HALF_LM_SIZE + 0.5)) * LM_UNIT_SIZES[lmIndex.w] + rt.lightmapOrigin.xyz;

    return v;
}

shared vec3 SharedColors[gl_WorkGroupSize.x];

void main() {
    const vec3 LIGHTMAP_ORIGIN = rt.lightmapOrigin.xyz;

    Voxel voxel = lmBuffer.voxels[gl_WorkGroupID.x];
    
    ivec4 lmIndex = ivec4(voxel.lmIndex.x % LM_SIZE, voxel.lmIndex.yz, voxel.lmIndex.x / LM_SIZE);

    Bounds nodeHit = bvh.nodes[voxel.objectHit];
    vec3 point = posAtLightmapIndex(lmIndex, LIGHTMAP_ORIGIN);
    vec3 normal = normalize(point - nodeHit.position); // TODO: add hitPoint, normal, lmIndex, material to 8 byte buffer so it doesn't need to be recalculated every time

    vec3 hitPoint = normal * nodeHit.radius + nodeHit.position;

    vec3[4] rands = vec3[4]( // TODO: use a subarray in bn.items of 4 elements
        bn.items[gl_LocalInvocationID.x * 4].xyz,
        bn.items[gl_LocalInvocationID.x * 4 + 1].xyz,
        bn.items[gl_LocalInvocationID.x * 4 + 2].xyz,
        bn.items[gl_LocalInvocationID.x * 4 + 3].xyz
    );

    mat4x3 randDirs = mat4x3(
        normalize(normal + rands[0]),
        normalize(normal + rands[1]),
        normalize(normal + rands[2]),
        normalize(normal + rands[3])
    );

    RayResult results[4] = traceRayWithBVH4(hitPoint, randDirs); // bottleneck

    vec3 color = vec3(0.0);
    for (uint i = 0; i < 4; i++) {
        color += buf.mats[results[i].materialHit].emittance; // TODO: copy materials to shared memory?
    }

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

        Material material = buf.mats[nodeHit.material];
        color = color * (material.reflectance * (1.0 / LM_SAMPLES)) + material.emittance;

        imageStore(lmOutputColorImages[lmIndex.w], lmIndex.xyz, vec4(color, 0.0));
    }
}