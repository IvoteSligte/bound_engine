#version 460

#include "compute_includes.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(binding = 0, rgba16f) uniform restrict readonly image2D dataInputImage;
layout(binding = 1, rgba16f) uniform restrict writeonly image2D dataOutputImage;
layout(binding = 2, rgba32f) uniform restrict readonly image2D normalsDepthImage;
layout(binding = 3, r8ui) uniform restrict readonly uimage2D materialImage;
layout(binding = 4, r16f) uniform restrict readonly image2D historyLengthImage;

const ivec2 COORDS[48] = {
    ivec2(-3, -3), ivec2(-2, -3), ivec2(-1, -3), ivec2(0, -3), ivec2(1, -3), ivec2(2, -3), ivec2(3, -3),
    ivec2(-3, -2), ivec2(-2, -2), ivec2(-1, -2), ivec2(0, -2), ivec2(1, -2), ivec2(2, -2), ivec2(3, -2),
    ivec2(-3, -1), ivec2(-2, -1), ivec2(-1, -1), ivec2(0, -1), ivec2(1, -1), ivec2(2, -1), ivec2(3, -1),
    ivec2(-3,  0), ivec2(-2,  0), ivec2(-1,  0),               ivec2(1,  0), ivec2(2,  0), ivec2(3,  0),
    ivec2(-3,  1), ivec2(-2,  1), ivec2(-1,  1), ivec2(0,  1), ivec2(1,  1), ivec2(2,  1), ivec2(3,  1),
    ivec2(-3,  2), ivec2(-2,  2), ivec2(-1,  2), ivec2(0,  2), ivec2(1,  2), ivec2(2,  2), ivec2(3,  2),
    ivec2(-3,  3), ivec2(-2,  3), ivec2(-1,  3), ivec2(0,  3), ivec2(1,  3), ivec2(2,  3), ivec2(3,  3),
};

void main() {
    const ivec2 ipos = ivec2(gl_GlobalInvocationID.xy);

    float historyLength = imageLoad(historyLengthImage, ipos).x;

    vec4 sumData = imageLoad(dataInputImage, ipos);

    if (historyLength < 4.0) {
        vec4 normalDepth = imageLoad(normalsDepthImage, ipos); // xyz = normal, w = depth
        uint material = imageLoad(materialImage, ipos).x;

        float luminance = luminanceFromRGB(sumData.rgb);

        vec2 sumMoments = vec2(luminance, luminance * luminance);
        float sumWeights = 1.0;

        // 7x7 bilateral filter (except center)
        for (uint i = 0; i < 48; i++) {
            const ivec2 p = ipos + COORDS[i];

            vec3 cColor = imageLoad(dataInputImage, p).rgb;
            vec4 cNormalDepth = imageLoad(normalsDepthImage, p);
            uint cMaterial = imageLoad(materialImage, p).x;

            float cLuminance = luminanceFromRGB(cColor);

            float w = computeWeight(normalDepth, cNormalDepth, luminance, cLuminance, material, cMaterial);

            sumWeights += w;

            sumMoments += vec2(cLuminance, cLuminance * cLuminance) * w; // weight squared?
            sumData.rgb += cColor * w;
        }

        sumData.rgb /= sumWeights;
        sumMoments /= sumWeights;

        float variance = sumMoments.y - sumMoments.x * sumMoments.x;
        variance *= 4.0 / historyLength;

        sumData.a = variance;
    }

    imageStore(dataOutputImage, ivec2(gl_GlobalInvocationID.xy), sumData);
}