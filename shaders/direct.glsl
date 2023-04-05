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

layout(binding = 3, rgba16) uniform restrict readonly image3D[RAYS_INDIRECT * LIGHTMAP_COUNT] lightmapImages; // TODO: CPU side - only grab the first LIGHTMAP_COUNT images

layout(binding = 4, rg32ui) uniform restrict uimage3D[LIGHTMAP_COUNT] lightmapSyncImages;

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
        imageStore(colorImage, IPOS, vec4(0.0));
        return;
    }

    uint sync = imageLoad(lightmapSyncImages[lmIndex.w], lmIndex.xyz).x;
    uint level = sync & BITS_LEVEL;
    bool isUnused = (sync & BIT_USED) == 0;
    bool isUnfinished = level != RAYS_INDIRECT;

    if (isUnused && isUnfinished) {
        imageStore(lightmapSyncImages[lmIndex.w], lmIndex.xyz, uvec4(sync | BIT_USED, ray.objectHit, uvec2(0)));
    }

    vec3 color = level > 0 ? imageLoad(lightmapImages[LIGHTMAP_COUNT * (level - 1) + lmIndex.w], lmIndex.xyz).rgb : vec3(0.0);
    //vec3 color = vec3(!isUnused); // DEBUG
    imageStore(colorImage, IPOS, vec4(color, 0.0));
}
