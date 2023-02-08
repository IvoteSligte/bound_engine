#version 460

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(binding = 0) uniform restrict readonly RealTimeBuffer {
    vec4 rotation;
    vec4 previousRotation;
    vec3 position;
    vec3 previousPosition;
    ivec3 lightmapOrigin;
    ivec3 deltaLightmapOrigin;
    uint frame;
} rt;

layout(binding = 1, rgba16f) uniform restrict readonly image3D volumetricLightmapImageIn;
layout(binding = 2, rgba16f) uniform restrict writeonly image3D volumetricLightmapImageOut;

void main() { // FIXME:
    vec4 data = imageLoad(volumetricLightmapImageIn, ivec3(gl_GlobalInvocationID));
    imageStore(volumetricLightmapImageOut, ivec3(gl_GlobalInvocationID) - rt.deltaLightmapOrigin, data);
}
