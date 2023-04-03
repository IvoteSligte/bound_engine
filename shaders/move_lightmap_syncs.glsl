#version 460

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(constant_id = 0) const uint LIGHTMAP_INDEX = 0;

layout(binding = 0) uniform restrict readonly RealTimeBuffer {
    vec4 rotation;
    vec4 previousRotation;
    vec3 position;
    vec3 previousPosition;
    ivec3 lightmapOrigin;
    ivec4 deltaLightmapOrigins[LIGHTMAP_COUNT];
    uint frame;
} rt;

layout(binding = 1, r32ui) uniform restrict readonly uimage3D lightmapImageIn;
layout(binding = 2, r32ui) uniform restrict writeonly uimage3D lightmapImageStaging;

void main() {
    uint data = imageLoad(lightmapImageIn, ivec3(gl_GlobalInvocationID)).x;
    imageStore(lightmapImageStaging, ivec3(gl_GlobalInvocationID) - rt.deltaLightmapOrigins[LIGHTMAP_INDEX].xyz, uvec4(data, uvec3(0)));
}