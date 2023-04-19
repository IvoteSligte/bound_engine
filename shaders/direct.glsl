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
    uint frame;
} rt;

layout(binding = 1) uniform restrict readonly GpuBVH {
    uint root;
    Bounds nodes[2 * MAX_OBJECTS];
} bvh;

layout(binding = 2, rgba16) uniform restrict writeonly image2D colorImage;

layout(binding = 3, rgba16) uniform restrict readonly image3D[LM_COUNT] lmInputColorImages;

#include "includes_trace_ray.glsl"

// TODO: change to fragment shader
void main() {
    const ivec2 VIEWPORT = ivec2(imageSize(colorImage).xy);
    const ivec2 IPOS = ivec2(gl_GlobalInvocationID.xy);

    // maps FragCoord to xy range [-1.0, 1.0]
    const vec2 NORM_COORD = RATIO * (IPOS * 2.0 / VIEWPORT - 1.0);
    const vec3 DIRECTION = normalize(vec3(NORM_COORD.x, 1.0, NORM_COORD.y));

    vec3 position = rt.position;
    vec4 rotation = rt.rotation;
    vec3 lightmapOrigin = rt.lightmapOrigin;

    vec3 viewDir = rotateWithQuat(rotation, DIRECTION);

    RayResult result = traceRayWithBVH(position, viewDir);

    vec3 p = (viewDir * result.distanceToHit) + position;
    ivec4 lmIndex = lightmapIndexAtPos(p, lightmapOrigin);

    bool outOfRange = lmIndex.w >= LM_COUNT;
    if (outOfRange) {
        imageStore(colorImage, IPOS, vec4(0.0)); // TODO: skybox
        return;
    }

    // TODO: bilinear color sampling (texture)
    vec3 color = imageLoad(lmInputColorImages[lmIndex.w], lmIndex.xyz).rgb;

    imageStore(colorImage, IPOS, vec4(color, 0.0));
}
