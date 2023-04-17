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

layout(binding = 1) uniform restrict readonly GpuBVH {
    uint root;
    Bounds nodes[2 * MAX_OBJECTS];
} bvh;

layout(binding = 2) uniform restrict readonly MutableData {
    Material mats[MAX_MATERIALS];
} buf;

layout(binding = 3, rgba16) uniform restrict readonly image3D[LM_COUNT] lmInputColorImages;

layout(binding = 4, rgba16) uniform restrict writeonly image3D[LM_COUNT] lmOutputColorImages;

layout(binding = 5, rgba16) uniform restrict image3D[LM_COUNT] lmFinalColorImages;

layout(binding = 6) buffer restrict readonly LMBuffer {
    Voxel voxels[LM_SIZE * LM_SIZE * LM_SIZE];
} lmBuffer;

layout(binding = 7) uniform restrict readonly BlueNoise {
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

shared vec3 SharedColors[LM_SAMPLES];

void main() {
    const vec3 LIGHTMAP_ORIGIN = rt.lightmapOrigin.xyz;

    Voxel voxel = lmBuffer.voxels[gl_WorkGroupID.x];
    
    ivec4 lmIndex = ivec4(voxel.lmIndex.x % LM_SIZE, voxel.lmIndex.yz, voxel.lmIndex.x / LM_SIZE);

    Bounds nodeHit = bvh.nodes[voxel.objectHit];
    vec3 point = posAtLightmapIndex(lmIndex, LIGHTMAP_ORIGIN);
    vec3 normal = normalize(point - nodeHit.position);

    vec3 hitPoint = normal * nodeHit.radius + nodeHit.position;

    vec3 randDir = normalize(normal + bn.items[gl_LocalInvocationID.x].xyz);
    Ray ray = Ray(0, hitPoint, randDir, 0);

    float distanceToHit = traceRayWithBVH(ray); // bottleneck
    ray.origin = (ray.direction * distanceToHit) + ray.origin;

    ivec4 lmIndexSample = lightmapIndexAtPos(ray.origin, LIGHTMAP_ORIGIN);
    vec3 color = imageLoad(lmInputColorImages[lmIndexSample.w], lmIndexSample.xyz).rgb; // TODO: texture access for smoother results

    SharedColors[gl_LocalInvocationID.x] = color;

    barrier();

    if (gl_LocalInvocationID.x < 64) {
        for (uint i = 1; i < LM_SAMPLES / 64; i++) {
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
        color = color * (material.reflectance * (1.0 / LM_SAMPLES));

        imageStore(lmOutputColorImages[lmIndex.w], lmIndex.xyz, vec4(color + material.emittance, 0.0));

        color += imageLoad(lmFinalColorImages[lmIndex.w], lmIndex.xyz).rgb;
        imageStore(lmFinalColorImages[lmIndex.w], lmIndex.xyz, vec4(color, 0.0));
    }
}
