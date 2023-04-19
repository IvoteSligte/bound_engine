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

layout(binding = 6) uniform restrict readonly NoiseBuffer {
    vec4 items[LM_SAMPLES / 4][4];
} noise;

struct SharedGpuBVH {
    uint root;
    Bounds nodes[2 * MAX_OBJECTS];
};

shared SharedGpuBVH SharedBVH;

#include "includes_trace_ray.glsl"

shared SharedStruct SharedData;
shared vec3 SharedColors[gl_WorkGroupSize.x];

RayResult[4] traceRayWithBVH4(vec3 origin, mat4x3 dirs) {
    RayResult results[4] = RayResult[4](
        RayResult(FLT_MAX, 0, 0),
        RayResult(FLT_MAX, 0, 0),
        RayResult(FLT_MAX, 0, 0),
        RayResult(FLT_MAX, 0, 0)
    );

    uint currIdx = SharedBVH.root;

    while (currIdx != 0) {
        Bounds curr = SharedBVH.nodes[currIdx];

        if (curr.material == 0) {
            currIdx = hitsBounds4(origin, dirs, curr) ? curr.child : curr.next;
            continue;
        }

        float[4] dists = distanceToObject4(origin, dirs, curr);

        for (uint i = 0; i < 4; i++) {
            if (dists[i] > EPSILON && dists[i] < results[i].distanceToHit) {
                // is a leaf, store data
                results[i] = RayResult(dists[i], currIdx, curr.material);
            }
        }

        // move to next node
        currIdx = curr.next;
    }

    return results;
}

void main() {
    if (gl_LocalInvocationID.x == 0) {
        SharedData = SharedStruct(lmBuffer.voxels[gl_WorkGroupID.x], rt.lightmapOrigin);
        SharedBVH = SharedGpuBVH(bvh.root, bvh.nodes);
    }
    barrier();
    
    SharedStruct sData = SharedData;
    Voxel voxel = sData.voxel;

    vec4[4] rands = noise.items[gl_LocalInvocationID.x];

    mat4x3 randDirs = mat4x3( // TODO: optimize?
        normalize(voxel.normal + rands[0].xyz),
        normalize(voxel.normal + rands[1].xyz),
        normalize(voxel.normal + rands[2].xyz),
        normalize(voxel.normal + rands[3].xyz)
    );

    RayResult results[4] = traceRayWithBVH4(voxel.hitPoint, randDirs); // bottleneck

    vec3 color = vec3(0.0);
    for (uint i = 0; i < 4; i++) {
        vec3 p = (randDirs[i] * results[i].distanceToHit) + voxel.hitPoint;
        ivec4 lmIndexSample = lightmapIndexAtPos(p, sData.lightmapOrigin);
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

        Material material = buf.mats[voxel.materialHit];
        color = color * (material.reflectance * (1.0 / LM_SAMPLES) + material.emittance);

        imageStore(lmOutputColorImages[voxel.lmIndex.w], voxel.lmIndex.xyz, vec4(color, 0.0));
    }
}
