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
    ivec4 deltaLightmapOrigins[LIGHTMAP_COUNT];
    uint frame;
} rt;

layout(binding = 1) uniform restrict readonly GpuBVH {
    uint root;
    Bounds nodes[2 * MAX_OBJECTS];
} bvh;

layout(binding = 2, rgba16) uniform restrict writeonly image2D colorImage;

layout(binding = 3, rgba16) uniform restrict readonly image3D[RAYS_INDIRECT * LIGHTMAP_COUNT] lightmapImages;

layout(binding = 4, r32ui) uniform restrict uimage3D[LIGHTMAP_COUNT] lightmapUsedImages;

layout(binding = 5, r32ui) uniform restrict writeonly uimage3D[LIGHTMAP_COUNT] lightmapObjectHitImages;

layout(binding = 6, r32ui) uniform restrict readonly uimage3D[LIGHTMAP_COUNT] lightmapLevelImages;

#include "includes_trace_ray.glsl"

/// returns an index into a lightmap image in xyz, and the image index in w
ivec4 lightmapIndexAtPos(vec3 v) {
    const int HALF_IMAGE_SIZE = imageSize(lightmapImages[0]).x >> 1;
    const float INV_HALF_LM_SIZE = 1.0 / (float(HALF_IMAGE_SIZE) * LM_UNIT_SIZE);

    v -= rt.lightmapOrigin.xyz;
    uint lightmapNum = uint(log2(max(maximum(abs(v)) * INV_HALF_LM_SIZE, 0.5001)) + 1.5);
    float unitSize = (1 << lightmapNum) * LM_UNIT_SIZE;

    ivec3 index = ivec3(round(v / unitSize)) + HALF_IMAGE_SIZE;

    return ivec4(index, lightmapNum);
}

void main() {
    const uvec3 TOTAL_SIZE = gl_NumWorkGroups * gl_WorkGroupSize;
    const uint TOTAL_LENGTH = TOTAL_SIZE.x * TOTAL_SIZE.y; // only two dimensions are used
    const uint GLOBAL_INVOCATION_INDEX = TOTAL_SIZE.x * gl_GlobalInvocationID.y + gl_GlobalInvocationID.x; // only two dimensions are used

    const ivec2 VIEWPORT = ivec2(imageSize(colorImage).xy);
    const ivec2 IPOS = ivec2(gl_GlobalInvocationID.xy);

    // maps FragCoord to xy range [-1.0, 1.0]
    const vec2 NORM_COORD = RATIO * (IPOS * 2.0 / VIEWPORT - 1.0);
    const vec3 DIRECTION = normalize(vec3(NORM_COORD.x, 1.0, NORM_COORD.y));

    vec3 viewDir = rotateWithQuat(rt.rotation, DIRECTION);

    Ray ray = Ray(0, rt.position, viewDir, 0);

    traceRayWithBVH(ray);

    ivec4 lmIndex = lightmapIndexAtPos(ray.origin);

    bool outOfRange = lmIndex.w >= LIGHTMAP_COUNT;
    if (outOfRange) {
        imageStore(colorImage, IPOS, vec4(0.1)); // TODO: skybox
        return;
    }

    uint level = imageLoad(lightmapLevelImages[lmIndex.w], lmIndex.xyz).x;

    if (level == 0) {
        // TODO: separate function
        ivec3 chunk = ivec3(lmIndex.x / 32, lmIndex.yz);
        imageAtomicOr(lightmapUsedImages[lmIndex.w], chunk.xyz, 1 << (lmIndex.x % 32));

        imageStore(lightmapObjectHitImages[lmIndex.w], lmIndex.xyz, uvec4(ray.objectHit));
    }

    // TODO: bilinear color sampling
    vec3 color;
    if (gl_GlobalInvocationID.x < 1920 / 2) { // DEBUG
        color = level > 0 ? imageLoad(lightmapImages[LIGHTMAP_COUNT * (level - 1) + lmIndex.w], lmIndex.xyz).rgb : vec3(0.0); // NONDEBUG
    } else {
        // DEBUG
        color = vec3(level) / RAYS_INDIRECT;
    }
    imageStore(colorImage, IPOS, vec4(color, 0.0));
}
