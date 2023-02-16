#version 460

#include "includes.glsl"

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(binding = 0) uniform restrict readonly RealTimeBuffer {
    vec4 rotation;
    vec4 previousRotation;
    vec3 position;
    vec3 previousPosition;
    ivec4 lightmapOrigins[LIGHTMAP_CASCADES];
    ivec3 deltaLightmapOrigin;
    uint frame;
} rt;

layout(binding = 1, rgba32f) uniform restrict readonly image3D lightmapImageIn;
layout(binding = 2, rgba32f) uniform restrict writeonly image3D lightmapImageOut;

void main() { // TODO: cascades
    vec4 data = imageLoad(lightmapImageIn, ivec3(gl_GlobalInvocationID));
    imageStore(lightmapImageOut, ivec3(gl_GlobalInvocationID) - rt.deltaLightmapOrigin, data);
}
