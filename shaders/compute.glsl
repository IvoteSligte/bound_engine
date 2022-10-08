#version 460

layout(binding = 0, rgba8) uniform image2D acc;
layout(binding = 1, rgba8) uniform readonly image2D curr;

void main() {
    vec4 accPixel = imageLoad(acc, ivec2(gl_WorkGroupID.xy));
    vec4 currPixel = imageLoad(curr, ivec2(gl_WorkGroupID.xy));

    float blendFactor = 0.05 * dot(currPixel.xyz, vec3(1.0));
    imageStore(acc, ivec2(gl_WorkGroupID.xy), mix(accPixel, currPixel, blendFactor));
}
