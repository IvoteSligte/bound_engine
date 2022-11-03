#version 460

#include "compute_includes.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform DenoisePushConstants {
    uint stage;
} pc;

layout(binding = 0, rgba16f) uniform restrict writeonly image2D accumulatorImageRead; // read from in previous shader
layout(binding = 1, rgba16f) uniform restrict readonly image2D accumulatorImageWrite; // written to in previous shader
layout(binding = 2, rgba32f) uniform restrict readonly image2D normalsDepthImage;
layout(binding = 3, r8ui) uniform restrict readonly uimage2D materialImage;
layout(binding = 4, r16f) uniform restrict writeonly image2D prevMomentImage;
layout(binding = 5, r16f) uniform restrict readonly image2D curMomentImage;
layout(binding = 6, r16f) uniform restrict writeonly image2D prevVarianceImage;
layout(binding = 7, r16f) uniform restrict readonly image2D curVarianceImage;

float luminance(vec3 rgb) {
    return dot(rgb, vec3(0.2126, 0.7152, 0.0722));
}

const float wavelet[3][3] = {
    { 0.140625, 0.09375, 0.0234375 },
    { 0.09375, 0.125, 0.015625 },
    { 0.0234375, 0.015625, 0.00390625 },
};

void main() {
    vec4 data = imageLoad(accumulatorImageWrite, ivec2(gl_GlobalInvocationID.xy));
    data.rgb *= wavelet[0][0];

    vec4 normalDepth = imageLoad(normalsDepthImage, ivec2(gl_GlobalInvocationID.xy)); // xyz = normal, w = depth
    uint material = imageLoad(materialImage, ivec2(gl_GlobalInvocationID.xy)).x;

    float weightSum = wavelet[0][0];
    vec2 moment = imageLoad(curMomentImage, ivec2(gl_GlobalInvocationID.xy)).xy;

    // TEMP;
    float none = imageLoad(curVarianceImage, ivec2(gl_GlobalInvocationID.xy)).x;

    int stride = 1 << pc.stage;

    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
            ivec2 i = ivec2(gl_GlobalInvocationID.xy) + ivec2(x, y) * stride;

            if (ivec2(x, y) == ivec2(0)) {
                continue;
            }

            vec4 cData = imageLoad(accumulatorImageWrite, i);
            vec4 cNormalDepth = imageLoad(normalsDepthImage, i);
            uint cMaterial = imageLoad(materialImage, i).x;

            float l = luminance(cData.rgb);

            float weightDepth = abs(normalDepth.w - cNormalDepth.w);
            float weightNormal = max(dot(normalDepth.xyz, cNormalDepth.xyz), 0.0); // to a power, maybe

            float w = exp(-weightDepth) * weightNormal * float(material == cMaterial) * wavelet[abs(x)][abs(y)];

            weightSum += w;

            data.rgb += cData.rgb * w;
            moment += vec2(l, l * l) * w;
        }
    }

    moment /= weightSum;
    data.rgb /= weightSum;

    imageStore(prevVarianceImage, ivec2(gl_GlobalInvocationID.xy), vec4(max(0.0, moment.y - moment.x * moment.x)));
    imageStore(prevMomentImage, ivec2(gl_GlobalInvocationID.xy), vec4(moment, vec2(0.0)));
    imageStore(accumulatorImageRead, ivec2(gl_GlobalInvocationID.xy), data);
}