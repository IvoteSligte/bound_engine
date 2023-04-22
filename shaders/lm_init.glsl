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

layout(binding = 1) uniform restrict readonly InitBuffer {
    Object objects[MAX_OBJECTS];
} objBuffer;

layout(binding = 2) buffer restrict LMBuffer {
    Voxel voxels[LM_SIZE * LM_SIZE * LM_SIZE * LM_COUNT];
} lmBuffer;

layout(binding = 3) buffer restrict LMDispatches {
    uint dispatches[3];
} lmDispatches;

layout(binding = 4, r32f) uniform restrict writeonly image3D SDFImages[LM_COUNT]; // TODO: descriptor set, calculations

layout(binding = 5, r32ui) uniform restrict writeonly uimage3D materialImages[LM_COUNT]; // TODO: descriptor set, r16ui?

const float SQRT_2 = 1.41421356;

shared Object SharedObjects[MAX_OBJECTS];

float calculateSDF(vec3 position, out Object closestObj) {
    float minDist = FLT_MAX;

    for (uint i = 0; i < MAX_OBJECTS; i++) {
        Object obj = SharedObjects[i];

        float dist = distance(position, obj.position) - obj.radius;

        if (dist < minDist) {
            minDist = dist;
            closestObj = obj;
        }
    }

    return minDist;
}

// TODO: break this up into multiple parts to do over time if necessary
void main() {
    // TODO: find a way to remove this, binding is declared invalid when not reading from the buffer
    Voxel _USELESS = lmBuffer.voxels[0];

    if (gl_LocalInvocationID == uvec3(0)) {
        SharedObjects = objBuffer.objects;
    }
    barrier();

    const ivec4 LM_INDEX = ivec4(gl_GlobalInvocationID.x % LM_SIZE, gl_GlobalInvocationID.yz, gl_GlobalInvocationID.x / LM_SIZE);

    vec3 position = posAtLightmapIndex(LM_INDEX, rt.lightmapOrigin.xyz);

    Object closestObj;
    float dist = calculateSDF(position, closestObj); // bottleneck // TODO: copy objects to shared memory (precalc step which separates objects into areas)

    uint material = 0;

    if (abs(dist) < SQRT_2 * LM_UNIT_SIZES[LM_INDEX.w]) {
        material = closestObj.material;

        vec3 normal = normalize(position - closestObj.position);
        vec3 position = normal * closestObj.radius + closestObj.position;

        uint index = atomicAdd(lmDispatches.dispatches[0], 1);
        lmBuffer.voxels[index] = Voxel(
            LM_INDEX,
            closestObj.material,
            position,
            normal
        );
    }

    imageStore(SDFImages[LM_INDEX.w], LM_INDEX.xyz, vec4(dist));
    imageStore(materialImages[LM_INDEX.w], LM_INDEX.xyz, uvec4(material)); // FIXME: materials do not appear to be correct on objects
}
