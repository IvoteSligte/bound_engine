#version 460

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#include "compute_includes.glsl"

layout(binding = 0, rgba16f) uniform restrict readonly image2D dataInputImage;
layout(binding = 1, rgba8) uniform restrict writeonly image2D renderImage;

vec3 rgb_to_srgb(vec3 rgb) {
    return pow(rgb, vec3(1.0 / 2.2));
}

void main() {
    vec4 data = imageLoad(dataInputImage, ivec2(gl_GlobalInvocationID.xy));

    imageStore(renderImage, ivec2(gl_GlobalInvocationID.xy), vec4(rgb_to_srgb(data.bgr), 1.0));
}
