#version 460

#include "includes_general.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(constant_id = 0) const float RATIO_X = 1.0;
layout(constant_id = 1) const float RATIO_Y = 1.0;

const vec2 RATIO = vec2(RATIO_X, RATIO_Y);

layout(binding = 0) uniform restrict readonly RealTimeBuffer {
    vec4 rotation;
    vec3 position;
    ivec3 lightmapOrigin; // TODO: different origin per layer
    ivec4 deltaLightmapOrigins[LM_LAYERS];
} rt;

layout(binding = 1, rgba16) uniform restrict writeonly image2D colorImage;

layout(binding = 2) uniform sampler3D SDFImages[LM_LAYERS];

layout(binding = 3) buffer readonly RadianceBuffer {
    Radiance radiances[LM_LAYERS][RADIANCE_SIZE][RADIANCE_SIZE][RADIANCE_SIZE];
    Material materials[LM_LAYERS][RADIANCE_SIZE][RADIANCE_SIZE][RADIANCE_SIZE];
} cache;

const float NEAR_CLIPPING = 1.0;

// TODO: change to fragment shader if beneficial
// TODO: do lower res raymarch first, then higher res, etc.
// TODO: multisampling
void main() {
    const ivec2 VIEWPORT = ivec2(imageSize(colorImage).xy);
    const ivec2 IPOS = ivec2(gl_GlobalInvocationID.xy);

    // maps FragCoord to xy range [-1.0, 1.0]
    const vec2 NORM_COORD = RATIO * (IPOS * 2.0 / VIEWPORT - 1.0);
    const vec3 DIRECTION = normalize(vec3(NORM_COORD.x, 1.0, NORM_COORD.y));

    vec3 position = rt.position;
    vec4 rotation = rt.rotation;
    vec3 lmOrigin = rt.lightmapOrigin;

    vec3 dir = rotateWithQuat(rotation, DIRECTION);
    float totalDist = NEAR_CLIPPING;
    bool isHit = marchRay(SDFImages, position, dir, lmOrigin, 1e-3, 128, totalDist);

    if (!isHit) {
        vec3 color = vec3(dot(dir, vec3(0.0, 0.0, 1.0)) * 0.5 + 0.5);
        color = pow(color, vec3(5.0)) * 0.1;
        imageStore(colorImage, IPOS, vec4(color, 0.0));
        return;
    }

    ivec4 radIndex = radIndexAtPos(position);

    uint layer = lmLayerAtPos(position, lmOrigin);
    vec3 normal = calcNormalSDF(SDFImages[layer], posToLMTextureCoord(position, layer, lmOrigin));

    vec3 color = vec3(0.0); // TODO: handle light reflection in `radiance.glsl`
    float dotSum = 0.0;

    vec3 reflectance = cache.materials[radIndex.w][radIndex.x][radIndex.y][radIndex.z].reflectance;

    for (uint i = 0; i < 64; i++) {
        vec3 radDir = directFibonacciSphere(float(i));
        uvec2 packedRadiance = cache.radiances[radIndex.w][radIndex.x][radIndex.y][radIndex.z].packed[i];
        vec3 radiance = vec3(unpackHalf2x16(packedRadiance.x), unpackHalf2x16(packedRadiance.y).x);
        float d = clamp(dot(normal, -radDir) * dot(-dir, normal), 0.0, 1.0);
        dotSum += d;
        color += radiance * d;
    }
    color = color * reflectance / dotSum;

    imageStore(colorImage, IPOS, vec4(color, 0.0));
}
