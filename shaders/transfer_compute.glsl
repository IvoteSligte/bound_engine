#version 460

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#include "compute_includes.glsl"

layout(binding = 0, rgba16f) uniform restrict readonly image2D denoiseOutputImage;
layout(binding = 1, rgba8) uniform restrict writeonly image2D renderImage; // output

void main() {
    vec4 data = imageLoad(denoiseOutputImage, ivec2(gl_GlobalInvocationID.xy));
    imageStore(renderImage, ivec2(gl_GlobalInvocationID.xy), vec4(data.bgr, 1.0));
}
