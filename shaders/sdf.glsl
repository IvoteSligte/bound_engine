#version 460

#include "includes_general.glsl"

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(binding = 0) uniform restrict readonly RealTimeBuffer {
    vec4 rotation;
    vec3 position;
    ivec3 lightmapOrigin; // TODO: different origin per layer
    ivec4 deltaLightmapOrigins[LM_LAYERS];
    vec2 screenSize;
    float fov;
} rt;

layout(binding = 1) uniform restrict readonly ObjectBuffer {
    Object objects[MAX_OBJECTS];
} objBuffer;

layout(binding = 2, r16f) uniform restrict writeonly image3D SDFImages[LM_LAYERS];

float calculateSDF(vec3 position) {
    float minDist = FLT_MAX;

    for (uint i = 0; i < MAX_OBJECTS; i++) {
        Object obj = objBuffer.objects[i];
        float dist = sdBox(position - obj.position, vec3(obj.radius));
        minDist = min(dist, minDist);
    }

    return minDist;
}

// TODO: move the origin by using a modulo when the point is outside the bounds of the image3D
void main() {
    // index in layer
    const ivec3 IIL = ivec3(gl_GlobalInvocationID.x % LM_SIZE, gl_GlobalInvocationID.yz);
    const int LAYER = int(gl_GlobalInvocationID.x / LM_SIZE);

    vec3 position = posAtLmIndex(IIL, LAYER, rt.lightmapOrigin);
    float dist = calculateSDF(position); // bottleneck // TODO: object acceleration structure

    imageStore(SDFImages[LAYER], IIL, vec4(dist));
}
