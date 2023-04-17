#version 460

#include "includes_general.glsl"

layout(local_size_x = LM_SAMPLES, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint OFFSET_USED = 0;

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

layout(binding = 4, rgba16) uniform restrict writeonly image3D[LM_COUNT] lmFinalColorImages;

layout(binding = 5, r32ui) uniform restrict readonly uimage3D[LM_COUNT] lmUsedImages;

layout(binding = 6, r32ui) uniform restrict readonly uimage3D[LM_COUNT] lmObjectHitImages;

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
    const uint LIGHTMAP_LAYER = gl_WorkGroupID.x / (LM_SIZE / 32);
    const ivec3 LIGHTMAP_CHUNK = ivec3(gl_WorkGroupID.x % (LM_SIZE / 32), gl_WorkGroupID.yz); // TODO: do not dispatch for ignored chunks (layers 2+ in the middle)

    const vec3 LIGHTMAP_ORIGIN = rt.lightmapOrigin.xyz;

    uint used = imageLoad(lmUsedImages[LIGHTMAP_LAYER], LIGHTMAP_CHUNK).x;
    const uint MASK = ALL_ONES << OFFSET_USED;

    uvec2 lsb = findLSB(uvec2(used & MASK, used)); // prioritizes unexplored lightmap voxels
    uint target = lsb.x == -1 ? lsb.y : lsb.x;

    if (target == -1) {
        return;
    }

    ivec3 lmIndex = ivec3((32 * LIGHTMAP_CHUNK.x) + target, gl_WorkGroupID.yz);

    uint nodeHitIndex = imageLoad(lmObjectHitImages[LIGHTMAP_LAYER], lmIndex.xyz).x;
    Bounds nodeHit = bvh.nodes[nodeHitIndex];
    vec3 point = posAtLightmapIndex(ivec4(lmIndex.xyz, LIGHTMAP_LAYER), LIGHTMAP_ORIGIN);
    vec3 normal = normalize(point - nodeHit.position);

    vec3 hitPoint = normal * nodeHit.radius + nodeHit.position;

    vec3 randDir = normalize(normal + bn.items[gl_LocalInvocationID.x].xyz);
    Ray ray = Ray(0, hitPoint, randDir, 0);

    traceRayWithBVH(ray); // bottleneck

    vec3 color = buf.mats[ray.materialHit].emittance; // TODO: only materialHit is used, optimise the traceRayWithBVH function for this

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
        color = color * (material.reflectance * (1.0 / LM_SAMPLES)) + material.emittance;

        imageStore(lmOutputColorImages[LIGHTMAP_LAYER], lmIndex.xyz, vec4(color, 0.0));
        
        imageStore(lmFinalColorImages[LIGHTMAP_LAYER], lmIndex.xyz, vec4(color, 0.0));
    }
}
