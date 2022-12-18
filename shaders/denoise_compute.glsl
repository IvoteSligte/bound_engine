#version 460

#include "compute_includes.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform DenoisePushConstants {
    uint stage;
} pc;

layout(binding = 0, rgba16f) uniform restrict readonly image2D dataInputImage;
layout(binding = 1, rgba16f) uniform restrict writeonly image2D dataOutputImage;
layout(binding = 2) uniform sampler2D dataInputTexture;

layout(set = 1, binding = 0, rgba32f) uniform restrict readonly image2D normalsDepthImage;
layout(set = 1, binding = 1, r8ui) uniform restrict uimage2D historyLengthImage;

// corner of an a trous WAVELET
const float WAVELET[2][2] = {
    { 1.0 / 4.0, 1.0 /  8.0 },
    { 1.0 / 8.0, 1.0 / 16.0 },
};

const ivec2 COORDS[8] = {
    ivec2(-1, -1), ivec2(0, -1), ivec2(1, -1),
    ivec2(-1,  0),               ivec2(1,  0),
    ivec2(-1,  1), ivec2(0,  1), ivec2(1,  1),
};

float varianceGaussianSampled() {
    const vec2 OFFSETS[4] = { vec2(-0.5), vec2(0.5, -0.5), vec2(-0.5, 0.5), vec2(0.5) };
    const vec2 fpos = vec2(gl_GlobalInvocationID.xy);

    float sum = 0.0;

    for (uint i = 0; i < 4; i++) {
        const vec2 p = fpos + OFFSETS[i];
        sum += texture(dataInputTexture, p).a;
    }

    return sum * 0.25;
}

void main() {
    ivec2 ipos = ivec2(gl_GlobalInvocationID.xy);

    vec4 data = imageLoad(dataInputImage, ipos);
    data.rgb *= WAVELET[0][0];

    vec4 normalDepth = imageLoad(normalsDepthImage, ipos); // xyz = normal, w = depth
    float historyLength = imageLoad(historyLengthImage, ipos).x;

    float weightSum = WAVELET[0][0];
    float luminance = luminanceFromRGB(data.rgb);
    float sqrt_variance = sqrt(varianceGaussianSampled());

    int stride = 1 << pc.stage;

    for (uint i = 0; i < 8; i++) {
        const ivec2 c = COORDS[i];
        ivec2 p = ipos + c * stride;

        vec4 cData = imageLoad(dataInputImage, p);
        vec4 cNormalDepth = imageLoad(normalsDepthImage, p);
        float cHistoryLength = imageLoad(historyLengthImage, p).x;

        float cLuminance = luminanceFromRGB(cData.rgb);

        float weightDepth = abs(normalDepth.w - cNormalDepth.w);
        float weightNormal = max(dot(normalDepth.xyz, cNormalDepth.xyz), 0.0);
        float weightLuminance = abs(luminance - cLuminance) / sqrt_variance;
        float weightHistoryLength = cHistoryLength / historyLength;

        const float weightWavelet = WAVELET[abs(c.x)][abs(c.y)];
        float w = exp(-weightDepth - max(weightLuminance, 0.0)) * weightNormal * weightHistoryLength * weightWavelet;

        weightSum += w;
        data += cData * vec4(w.xxx, w * w);
    }

    data /= vec4(weightSum.xxx, weightSum * weightSum);

    imageStore(dataOutputImage, ivec2(gl_GlobalInvocationID.xy), data);
}