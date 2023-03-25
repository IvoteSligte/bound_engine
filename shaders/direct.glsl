#version 460

#include "includes_general.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

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

// TODO: specialization constants?
layout(binding = 2) uniform restrict readonly ConstantBuffer {
    vec2 ratio; // window height / width * fov
} cs;

layout(binding = 3, rgba16) uniform restrict writeonly image2D colorImage;

layout(binding = 4, rgba8) uniform restrict readonly image3D[LIGHTMAP_COUNT] lightmapMidImages;

layout(binding = 5, r32ui) uniform restrict uimage3D[LIGHTMAP_COUNT] lightmapSyncImages;

layout(binding = 6) buffer restrict MidBuffer {
    HitItem items[MID_SUBBUFFER_COUNT][MID_SUBBUFFER_LENGTH];
} midBuffer;

layout(binding = 7) buffer restrict MidCounters {
    uint counters[MID_SUBBUFFER_COUNT];
} midCounters;

layout(binding = 8) buffer restrict LastBuffer {
    HitItem items[LAST_SUBBUFFER_COUNT][LAST_SUBBUFFER_LENGTH];
} lastBuffer;

layout(binding = 9) buffer restrict LastCounters {
    uint counters[LAST_SUBBUFFER_COUNT];
} lastCounters;

#include "includes_trace_ray.glsl"

/// returns an index into a lightmap image in xyz, and the image index in w
ivec4 lightmapIndexAtPos(vec3 v) {
    const int HALF_IMAGE_SIZE = imageSize(lightmapMidImages[0]).x >> 1;
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
    HitItem useless = midBuffer.items[0][0];
    HitItem useless2 = lastBuffer.items[0][0];

    const ivec2 viewport = ivec2(imageSize(colorImage).xy);
    const ivec2 ipos = ivec2(gl_GlobalInvocationID.xy);

    // maps FragCoord to xy range [-1.0, 1.0]
    vec2 normCoord = ipos * (2.0 / viewport) - 1.0;
    // maps normCoord to a different range (e.g. for FOV and non-square windows)
    normCoord *= cs.ratio;

    vec3 viewDir = rotateWithQuat(rt.rotation, normalize(vec3(normCoord.x, 1.0, normCoord.y)));

    Ray ray = Ray(0, rt.position, viewDir);

    vec3 hitObjPosition;
    traceRayWithBVH(ray, hitObjPosition);

    ivec4 lmIndex = lightmapIndexAtPos(ray.origin);

    uint syncValue = imageAtomicOr(lightmapSyncImages[lmIndex.w], lmIndex.xyz, (1 | 2));
    if ((syncValue & 1) == 0) {
        const uint INVOCATIONS_PER_COUNTER = TOTAL_LENGTH / MID_SUBBUFFER_COUNT;
        const uint COUNTER_INDEX = GLOBAL_INVOCATION_INDEX / INVOCATIONS_PER_COUNTER;

        uint bufIdx = atomicAdd(midCounters.counters[COUNTER_INDEX], 1);
        if (bufIdx < MID_SUBBUFFER_LENGTH) {
            midBuffer.items[COUNTER_INDEX][bufIdx] = HitItem(ray.origin, ray.objectHit);
        }
    } else {
        vec3 color = imageLoad(lightmapMidImages[lmIndex.w], lmIndex.xyz).rgb;
        imageStore(colorImage, ipos, vec4(color, 0.0));
    }

    if ((syncValue & 2) == 0) {
        const uint INVOCATIONS_PER_COUNTER = TOTAL_LENGTH / LAST_SUBBUFFER_COUNT;
        const uint COUNTER_INDEX = GLOBAL_INVOCATION_INDEX / INVOCATIONS_PER_COUNTER;

        uint bufIdx = atomicAdd(lastCounters.counters[COUNTER_INDEX], 1);
        if (bufIdx < LAST_SUBBUFFER_LENGTH) {
            lastBuffer.items[COUNTER_INDEX][bufIdx] = HitItem(ray.origin, ray.objectHit);
        }
    }
}
