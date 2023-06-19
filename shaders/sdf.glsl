#version 460

#include "includes_general.glsl"

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(binding = 0) uniform restrict readonly RealTimeBuffer {
    vec4 rotation;
    vec3 position;
    ivec3 lightmapOrigin; // TODO: different origin per layer
    ivec4 deltaLightmapOrigins[LM_LAYERS];
} rt;

layout(binding = 1) uniform restrict readonly ObjectBuffer {
    Object objects[MAX_OBJECTS];
} objBuffer;

layout(binding = 2, r16f) uniform restrict writeonly image3D SDFImages[LM_LAYERS];

float calculateSDF(vec3 position) {
    float minDist = FLT_MAX;

    for (uint i = 0; i < MAX_OBJECTS; i++) {
        Object obj = objBuffer.objects[i];

        float dist = distance(position, obj.position) - obj.radius;

        if (dist < minDist) {
            minDist = dist;
        }
    }

    return minDist;
}

void main() {
    const ivec4 LM_INDEX = ivec4(gl_GlobalInvocationID.x % LM_SIZE, gl_GlobalInvocationID.yz, gl_GlobalInvocationID.x / LM_SIZE);

    vec3 position = posAtLmIndex(LM_INDEX, rt.lightmapOrigin);

    float dist = calculateSDF(position); // bottleneck // TODO: object acceleration structure

    imageStore(SDFImages[LM_INDEX.w], LM_INDEX.xyz, vec4(dist));
}
