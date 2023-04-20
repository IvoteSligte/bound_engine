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

layout(binding = 1) uniform restrict readonly ObjectBuffer {
    Object objects[MAX_OBJECTS];
} objBuffer;

layout(binding = 2) buffer restrict LMBuffer {
    Voxel voxels[LM_SIZE * LM_SIZE * LM_SIZE * LM_COUNT];
} lmBuffer;

layout(binding = 3) buffer restrict LMDispatches {
    uint dispatches[3];
} lmDispatches;

const float SQRT_2 = 1.41421356;

bool customSphereIntersect(vec3 position, float radius, out Object objIntersected) {
    for (uint i = 0; i < MAX_OBJECTS; i++) {
        Object obj = objBuffer.objects[i];

        float dist = distance(position, obj.position) - obj.radius;

        if (abs(dist) <= radius) {
            // TODO: do not calculate point's lighting if point is inside of an object
            objIntersected = obj;
            return true;
        }
    }
    return false;
}

// TODO: break this up into multiple parts to do over time if necessary
void main() {
    // TODO: find a way to remove this, binding is declared invalid when not reading from the buffer
    Voxel _USELESS = lmBuffer.voxels[0];

    const vec3 LIGHTMAP_ORIGIN = rt.lightmapOrigin.xyz;

    const ivec4 LM_INDEX = ivec4(gl_GlobalInvocationID.x % LM_SIZE, gl_GlobalInvocationID.yz, gl_GlobalInvocationID.x / LM_SIZE);

    vec3 position = posAtLightmapIndex(LM_INDEX, LIGHTMAP_ORIGIN);
    float radius = SQRT_2 * LM_UNIT_SIZES[LM_INDEX.w];

    Object objIntersected;
    bool intersected = customSphereIntersect(position, radius, objIntersected); // bottleneck // TODO: copy BVH to shared memory

    if (intersected) {
        vec3 normal = normalize(position - objIntersected.position);
        vec3 hitPoint = normal * objIntersected.radius + objIntersected.position;

        uint index = atomicAdd(lmDispatches.dispatches[0], 1);
        lmBuffer.voxels[index] = Voxel(
            LM_INDEX,
            objIntersected.material,
            hitPoint,
            normal
        );
    }
}
