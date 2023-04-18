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

layout(binding = 3, rgba16) uniform restrict readonly image3D[LM_COUNT] lmInputColorImages;

layout(binding = 4, rgba16) uniform restrict writeonly image3D[LM_COUNT] lmOutputColorImages;

layout(binding = 5) buffer restrict readonly LMBuffer {
    Voxel voxels[LM_SIZE * LM_SIZE * LM_SIZE * LM_COUNT];
} lmBuffer;

layout(binding = 6) uniform restrict readonly BlueNoise {
    vec4 items[LM_SAMPLES / 4][4];
} bn;

#include "includes_trace_ray.glsl"

shared vec3 SharedColors[gl_WorkGroupSize.x];

void main() {
    const vec3 LIGHTMAP_ORIGIN = rt.lightmapOrigin.xyz;

    Voxel voxel = lmBuffer.voxels[gl_WorkGroupID.x];
    
    ivec4 lmIndex = ivec4(voxel.lmIndex.x % LM_SIZE, voxel.lmIndex.yz, voxel.lmIndex.x / LM_SIZE);

    Bounds nodeHit = bvh.nodes[voxel.objectHit];
    vec3 point = posAtLightmapIndex(lmIndex, LIGHTMAP_ORIGIN);
    vec3 normal = normalize(point - nodeHit.position);

    vec3 hitPoint = normal * nodeHit.radius + nodeHit.position;

    vec4[4] rands = bn.items[gl_LocalInvocationID.x];

    mat4x3 randDirs = mat4x3( // TODO: optimize?
        normalize(normal + rands[0].xyz),
        normalize(normal + rands[1].xyz),
        normalize(normal + rands[2].xyz),
        normalize(normal + rands[3].xyz)
    );

    RayResult results[4] = traceRayWithBVH4(hitPoint, randDirs); // bottleneck

    vec3 color = vec3(0.0);
    for (uint i = 0; i < 4; i++) {
        vec3 p = (randDirs[i] * results[i].distanceToHit) + hitPoint;
        ivec4 lmIndexSample = lightmapIndexAtPos(p, LIGHTMAP_ORIGIN);
        color += imageLoad(lmInputColorImages[lmIndexSample.w], lmIndexSample.xyz).rgb; // TODO: texture access for smoother results
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
        color = color * (material.reflectance * (1.0 / LM_SAMPLES) + material.emittance);

        imageStore(lmOutputColorImages[lmIndex.w], lmIndex.xyz, vec4(color, 0.0));
    }
}
