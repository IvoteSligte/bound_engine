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

layout(binding = 4, r32ui) uniform restrict uimage3D[LIGHTMAP_COUNT] lightmapSyncImages;

layout(binding = 5) buffer restrict CurrBuffer {
    HitItem items[SUBBUFFER_COUNT][SUBBUFFER_LENGTH];
} currBuffer;

layout(binding = 6) buffer restrict CurrCounters {
    uint counters[SUBBUFFER_COUNT];
} currCounters;

#include "includes_trace_ray.glsl"

/// returns an index into a lightmap image in xyz, and the image index in w
ivec4 lightmapIndexAtPos(vec3 v) {
    const int HALF_IMAGE_SIZE = imageSize(lightmapImages[0]).x >> 1;
    const float BASE_UNIT_SIZE = 0.5; // TODO: adapt this into the rust code, currently a base unit size of 1 is used there
    const float INV_HALF_LM_SIZE = 1.0 / (float(HALF_IMAGE_SIZE) * BASE_UNIT_SIZE);

    v -= rt.lightmapOrigin.xyz;
    uint lightmapNum = uint(log2(max(maximum(abs(v)) * INV_HALF_LM_SIZE, 0.5001)) + 1.0);
    float unitSize = (1 << lightmapNum) * BASE_UNIT_SIZE;

    ivec3 index = ivec3(floor(v / unitSize)) + HALF_IMAGE_SIZE;

    return ivec4(index, lightmapNum);
}

void main() {
    const uvec3 TOTAL_SIZE = gl_NumWorkGroups * gl_WorkGroupSize;
    const uint TOTAL_LENGTH = TOTAL_SIZE.x * TOTAL_SIZE.y; // only two dimensions are used
    const uint GLOBAL_INVOCATION_INDEX = TOTAL_SIZE.x * gl_GlobalInvocationID.y + gl_GlobalInvocationID.x; // only two dimensions are used

    // FIXME: the binding is not recognised when buffers are only written to and not read from
    HitItem useless = currBuffer.items[0][0];

    const ivec2 VIEWPORT = ivec2(imageSize(colorImage).xy);
    const ivec2 IPOS = ivec2(gl_GlobalInvocationID.xy);

    // maps FragCoord to xy range [-1.0, 1.0]
    const vec2 NORM_COORD = RATIO * (IPOS * 2.0 / VIEWPORT - 1.0);
    const vec3 DIRECTION = normalize(vec3(NORM_COORD.x, 1.0, NORM_COORD.y));

    vec3 viewDir = rotateWithQuat(rt.rotation, DIRECTION);

    Ray ray = Ray(0, rt.position, viewDir);

    vec3 hitObjPosition;
    traceRayWithBVH(ray, hitObjPosition);

    ivec4 lmIndex = lightmapIndexAtPos(ray.origin);

    uint sync = imageAtomicOr(lightmapSyncImages[lmIndex.w], lmIndex.xyz, BIT_USED);

    bool debug_set_used = false; // DEBUG

    bool isUnused = (sync & BIT_USED) == 0;
    if (isUnused) {
        const uint INVOCATIONS_PER_COUNTER = TOTAL_LENGTH / SUBBUFFER_COUNT;
        const uint COUNTER_INDEX = GLOBAL_INVOCATION_INDEX / INVOCATIONS_PER_COUNTER;

        uint bufIdx = atomicAdd(currCounters.counters[COUNTER_INDEX], 1);
        if (bufIdx < SUBBUFFER_LENGTH) {
            currBuffer.items[COUNTER_INDEX][bufIdx] = HitItem(ray.origin, ray.objectHit);
            debug_set_used = true;
        } else {
            imageStore(lightmapSyncImages[lmIndex.w], lmIndex.xyz, uvec4(sync));
        }
    }

    uint level = sync & BITS_LEVEL;
    if (level > 0) {
        //vec3 color = imageLoad(lightmapImages[LIGHTMAP_COUNT * (level - 1) + lmIndex.w], lmIndex.xyz).rgb;
        vec3 color = imageLoad(lightmapImages[lmIndex.w], lmIndex.xyz).rgb; // DEBUG
        imageStore(colorImage, IPOS, vec4(color, 0.0));
    }
}
