#version 460

#include "compute_includes.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform DenoisePushConstants {
    uint stage;
} pc;

layout(binding = 0, rgba16f) uniform restrict readonly image2D dataInputImage;
layout(binding = 1, rgba16f) uniform restrict writeonly image2D dataOutputImage;
layout(binding = 2, rgba32f) uniform restrict readonly image2D normalsDepthImage; // TODO: 16f ?
layout(binding = 3, r8ui) uniform restrict readonly uimage2D materialImage;

// corner of an a trous WAVELET
const float WAVELET[3][3] = {
    { 0.140625, 0.09375, 0.0234375 },
    { 0.09375, 0.125, 0.015625 },
    { 0.0234375, 0.015625, 0.00390625 },
};

const ivec2 COORDS[24] = {
    ivec2(-2, -2), ivec2(-1, -2), ivec2(0, -2), ivec2(1, -2), ivec2(2, -2),
    ivec2(-2, -1), ivec2(-1, -1), ivec2(0, -1), ivec2(1, -1), ivec2(2, -1),
    ivec2(-2,  0), ivec2(-1,  0),               ivec2(1,  0), ivec2(2,  0),
    ivec2(-2,  1), ivec2(-1,  1), ivec2(0,  1), ivec2(1,  1), ivec2(2,  1),
    ivec2(-2,  2), ivec2(-1,  2), ivec2(0,  2), ivec2(1,  2), ivec2(2,  2),
};

float varianceGaussian() {
    float sum = 0.0;

    const float kernel[2][2] = {
        { 1.0 / 4.0, 1.0 / 8.0  },
        { 1.0 / 8.0, 1.0 / 16.0 }
    };

    const ivec2 ipos = ivec2(gl_GlobalInvocationID.xy);

    const int radius = 1;
    for (int yy = -radius; yy <= radius; yy++)
    {
        for (int xx = -radius; xx <= radius; xx++)
        {
            ivec2 p = ipos + ivec2(xx, yy);

            float k = kernel[abs(xx)][abs(yy)];

            sum += imageLoad(dataInputImage, p).a * k;
        }
    }

    return sum;
}

void main() {
    ivec2 ipos = ivec2(gl_GlobalInvocationID.xy);

    vec4 data = imageLoad(dataInputImage, ipos);
    data.rgb *= WAVELET[0][0];

    vec4 normalDepth = imageLoad(normalsDepthImage, ipos); // xyz = normal, w = depth
    uint material = imageLoad(materialImage, ipos).x;

    float weightSum = WAVELET[0][0];

    float variance = varianceGaussian();
    float luminance = luminanceFromRGB(data.rgb);

    int stride = 1 << pc.stage;

    for (uint i = 0; i < 24; i++) {
        const ivec2 c = COORDS[i];
        ivec2 p = ipos + c * stride;

        vec4 cData = imageLoad(dataInputImage, p);
        vec4 cNormalDepth = imageLoad(normalsDepthImage, p);
        uint cMaterial = imageLoad(materialImage, p).x;

        float cLuminance = luminanceFromRGB(cData.rgb);

        float w = computeWeight(normalDepth, cNormalDepth, luminance, cLuminance, material, cMaterial) * WAVELET[abs(c.x)][abs(c.y)];

        weightSum += w;

        data += cData * vec4(w.xxx, w * w);
    }

    data /= vec4(weightSum.xxx, weightSum * weightSum);

    imageStore(dataOutputImage, ivec2(gl_GlobalInvocationID.xy), data);
}