#version 460

#include "compute_includes.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(binding = 0, rgba16f) uniform restrict readonly image2D dataInputImage;
layout(binding = 1, rgba16f) uniform restrict writeonly image2D dataOutputImage;
layout(binding = 2, rg32f) uniform restrict readonly image2D momentsInputImage;
layout(binding = 3, rg32f) uniform restrict writeonly image2D momentsOutputImage;

layout(set = 1, binding = 0, rgba32f) uniform restrict readonly image2D normalsDepthImage;
layout(set = 1, binding = 1, r8ui) uniform restrict uimage2D historyLengthImage;

const ivec2 COORDS[40] = {
                                                                      ivec2(0, -64),
                                                     ivec2(-16, -48), ivec2(0, -36), ivec2(16, -48),
                                    ivec2(-32, -32), ivec2(-12, -24), ivec2(0, -16), ivec2(12, -24), ivec2(32, -32),
                   ivec2(-48, -16), ivec2(-24, -12), ivec2(-8,   -8), ivec2(0,  -4), ivec2(8,   -8), ivec2(24, -12), ivec2(48, -16),
    ivec2(-64, 0), ivec2(-36,   0), ivec2(-16,   0), ivec2(-4,    0),                ivec2(4,    0), ivec2(16,   0), ivec2(36,   0), ivec2(64, 0),
                   ivec2(-48,  16), ivec2(-24,  12), ivec2(-8,    8), ivec2(0,   4), ivec2(8,    8), ivec2(24,  12), ivec2(48,  16),
                                    ivec2(-32,  32), ivec2(-12,  24), ivec2(0,  16), ivec2(12,  24), ivec2(32,  32),
                                                     ivec2(-16,  48), ivec2(0,  36), ivec2(16,  48),
                                                                      ivec2(0,  64),
};

void main() {
    const ivec2 ipos = ivec2(gl_GlobalInvocationID.xy);

    float historyLength = imageLoad(historyLengthImage, ipos).x;

    vec2 moments = imageLoad(momentsInputImage, ipos).xy;
    vec4 sumData = imageLoad(dataInputImage, ipos);
    vec2 sumMoments = moments;

    if (historyLength < 4.0) {
        vec4 normalDepth = imageLoad(normalsDepthImage, ipos); // xyz = normal, w = depth

        float sumWeights = 1.0;

        // 7x7 bilateral filter
        for (uint i = 0; i < 40; i++) {
            const ivec2 p = ipos + COORDS[i];

            vec3 cColor = imageLoad(dataInputImage, p).rgb;
            vec4 cNormalDepth = imageLoad(normalsDepthImage, p);
            float cHistoryLength = imageLoad(historyLengthImage, p).x;
            
            float cLuminance = luminanceFromRGB(cColor);
            vec2 cMoments = imageLoad(momentsInputImage, p).xy;

            float weightDepth = abs(normalDepth.w - cNormalDepth.w);
            float weightNormal = max(dot(normalDepth.xyz, cNormalDepth.xyz), 0.0);
            float weightLuminance = abs(moments.x - cLuminance);
            float weightHistoryLength = cHistoryLength / historyLength;

            float w = exp(-weightDepth) * weightNormal * weightHistoryLength;

            sumWeights += w;
            sumMoments += cMoments * w;
            sumData.rgb += cColor * w;
        }

        sumData.rgb /= sumWeights;
        sumMoments /= sumWeights;

        float variance = max(sumMoments.y - sumMoments.x * sumMoments.x, 0.0);
        variance *= 4.0 / historyLength;

        sumData.a = variance;
    }

    imageStore(dataOutputImage, ivec2(gl_GlobalInvocationID.xy), sumData);
    imageStore(momentsOutputImage, ivec2(gl_GlobalInvocationID.xy), vec4(sumMoments, vec2(0.0)));
}
