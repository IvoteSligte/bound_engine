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

layout(binding = 2) buffer restrict LMBuffer {
    Voxel voxels[LM_SIZE * LM_SIZE * LM_SIZE];
} lmBuffer;

layout(binding = 3) buffer restrict LMDispatches {
    uint dispatches[3];// TODO: clear on restart -> initial values are [0, 1, 1]
} lmDispatches;

const float SQRT_2 = 1.41421356;

bool pointContainedInBVHMasked(vec3 position, uint mask) {
    uint currIdx = bvh.root;

    while (currIdx != 0) {
        Bounds curr = bvh.nodes[currIdx];

        float pointDistSquared = dot(position, curr.position);

        if ((pointDistSquared > curr.radiusSquared) || currIdx == mask) {
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

        if (curr.material == 0) {
            if (dist <= radius) {
                currIdx = curr.child;
                continue;
            }
        } else {
            if (abs(dist) <= radius) {
                // TODO: do not calculate point's lighting if point is inside of an object
                // vec3 p = curr.position + normalize(position - curr.position) * curr.radius;

                // bool isContained = pointContainedInBVHMasked(p, currIdx);
                // if (!isContained) {
                    return currIdx;
                // }
            }
        }
        currIdx = curr.next;
    }

    return 0;
}

// TODO: break this up into multiple parts to do over time if necessary
void main() {
    // TODO: find a way to remove this, binding is declared invalid when not reading from the buffer
    Voxel _USELESS = lmBuffer.voxels[0];

    const vec3 LIGHTMAP_ORIGIN = rt.lightmapOrigin.xyz;

    const ivec4 LM_INDEX = ivec4(gl_GlobalInvocationID.x % LM_SIZE, gl_GlobalInvocationID.yz, gl_GlobalInvocationID.x / LM_SIZE);

    vec3 position = posAtLightmapIndex(LM_INDEX, LIGHTMAP_ORIGIN);
    float radius = SQRT_2 * LM_UNIT_SIZES[LM_INDEX.w]; // FIXME: radius is too large

    uint nodeIntersected = customSphereBVHIntersect(position, radius); // bottleneck

    if (nodeIntersected != 0) {
        uint index = atomicAdd(lmDispatches.dispatches[0], 1);
        lmBuffer.voxels[index] = Voxel(gl_GlobalInvocationID.xyz, nodeIntersected);
    }
}
