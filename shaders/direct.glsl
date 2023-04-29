#version 460

#include "includes_general.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(constant_id = 0) const float RATIO_X = 1.0;
layout(constant_id = 1) const float RATIO_Y = 1.0;

const vec2 RATIO = vec2(RATIO_X, RATIO_Y);

layout(binding = 0) uniform restrict readonly RealTimeBuffer {
    vec4 rotation;
    vec4 previousRotation;
    vec3 position;
    vec3 previousPosition;
    ivec3 lightmapOrigin;
    ivec4 deltaLightmapOrigins[LM_COUNT];
} rt;

layout(binding = 1, rgba16) uniform restrict writeonly image2D colorImage;

layout(binding = 2) uniform sampler3D[LM_COUNT] lmInputColorTextures;

layout(binding = 3) uniform sampler3D SDFImages[LM_COUNT];

#include "includes_march_ray.glsl"

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
    bool isHit = marchRay(position, dir, lmOrigin, 1e-3, totalDist);

    if (!isHit) {
        imageStore(colorImage, IPOS, vec4(0.0)); // TODO: skybox
        return;
    }

    int lmLayer = lmLayerAtPos(position, lmOrigin);
    float mult = (1.0 / float(LM_SIZE)) / LM_UNIT_SIZES[lmLayer]; // TODO: consts
    vec3 texIdx = (position - lmOrigin) * mult + 0.5; // TODO: lmOrigin varying between layers

    vec4 color = texture(lmInputColorTextures[lmLayer], texIdx);
    color.rgb /= color.w;

    imageStore(colorImage, IPOS, vec4(color.rgb, 0.0));
}
