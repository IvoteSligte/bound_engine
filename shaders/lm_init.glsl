#version 460

#include "includes_general.glsl"

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

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

layout(binding = 2, r32ui) uniform restrict uimage3D[LM_COUNT] lmOutputUsedImages;

layout(binding = 3, r32ui) uniform restrict uimage3D[LM_COUNT] lmObjectHitImages;

const float SQRT_2_DIV_2 = 0.7071068;

bool pointContainedInBVHMasked(vec3 position, uint mask) {
    uint currIdx = bvh.root;

    while (currIdx != 0) {
        Bounds curr = bvh.nodes[currIdx];

        float pointDistSquared = dot(position, curr.position);

        if (pointDistSquared > curr.radiusSquared || currIdx == mask) {
            currIdx = curr.next;
            continue;
        }

        if (curr.material == 0) {
            currIdx = curr.child;
        } else {
            return true;
            currIdx = curr.next;
        }
    }

    return false;
}

uint customSphereBVHIntersect(vec3 position, float radius) {
    uint currIdx = bvh.root;

    while (currIdx != 0) {
        Bounds curr = bvh.nodes[currIdx];

        float dist = distance(position, curr.position) - curr.radius;

        // continue if the edge of curr is not within the given sphere
        if (abs(dist) > radius) {
            currIdx = curr.next;
            continue;
        }

        if (curr.material == 0) {
            currIdx = curr.child;
        } else {
            vec3 p = curr.position + normalize(position - curr.position) * curr.radius;

            bool isContained = pointContainedInBVHMasked(p, currIdx);
            if (!isContained) {
                return currIdx;
            }

            currIdx = curr.next;
        }
    }

    return 0;
}

// TODO: break this up into multiple parts to do over time if necessary
void main() {
    const vec3 LIGHTMAP_ORIGIN = rt.lightmapOrigin.xyz;

    const ivec4 LM_INDEX = ivec4(gl_WorkGroupID.xyz, gl_WorkGroupID.x / LM_SIZE);

    vec3 position = posAtLightmapIndex(LM_INDEX, LIGHTMAP_ORIGIN);
    float radius = SQRT_2_DIV_2 * LM_UNIT_SIZES[LM_INDEX.w];

    uint nodeIntersected = customSphereBVHIntersect(position, radius); // bottleneck
    
    imageStore(lmObjectHitImages[LM_INDEX.w], LM_INDEX.xyz, uvec4(nodeIntersected));

    const ivec3 CHUNK = ivec3(LM_INDEX.x / 32, LM_INDEX.yz);
    const uint TARGET = 1 << (LM_INDEX.x % 32);

    if (nodeIntersected == 0) {
        // disable
        imageAtomicAnd(lmOutputUsedImages[LM_INDEX.w], CHUNK.xyz, ALL_ONES ^ TARGET);
    } else {
        // enable
        imageAtomicOr(lmOutputUsedImages[LM_INDEX.w], CHUNK.xyz, TARGET);
    }
}
