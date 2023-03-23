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

layout(binding = 3, rgba8) uniform restrict writeonly image2D colorImage;

layout(binding = 4, rgba8) uniform restrict readonly image3D[LIGHTMAP_COUNT] lightmapFinalImages;

layout(binding = 5) buffer restrict IndirectFinalBuffer {
    HitItem items[INDIRECT_FINAL_COUNTER_COUNT][INDIRECT_FINAL_ITEMS_PER_COUNTER];
} indirectFinalBuffer;

layout(binding = 6) buffer restrict IndirectFinalCounters {
    uint counters[INDIRECT_FINAL_COUNTER_COUNT];
} indirectFinalCounters;

#include "includes_trace_ray.glsl"

/// returns an index into a lightmap image in xyz, and the image index in w
ivec4 lightmapIndexAtPos(vec3 v) {
    const int HALF_IMAGE_SIZE = imageSize(lightmapFinalImages[0]).x >> 1;
    const float BASE_UNIT_SIZE = 0.5; // TODO: adapt this into the rust code, currently a base unit size of 1 is used there
    const float INV_HALF_LM_SIZE = 1.0 / (float(HALF_IMAGE_SIZE) * BASE_UNIT_SIZE);

    v -= rt.lightmapOrigin.xyz;
    uint lightmapNum = uint(log2(max(maximum(abs(v)) * INV_HALF_LM_SIZE, 0.5001)) + 1.0);
    float unitSize = (1 << lightmapNum) * BASE_UNIT_SIZE;

    ivec3 index = ivec3(floor(v / unitSize)) + HALF_IMAGE_SIZE;

    return ivec4(index, lightmapNum);
}

void main() {
    // FIXME: the binding is not recognised when 'indirectFinalBuffer' is only written to and not read from
    HitItem useless = indirectFinalBuffer.items[0][0];

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

    ivec4 index = lightmapIndexAtPos(ray.origin);
    vec4 data = imageLoad(lightmapFinalImages[index.w], index.xyz);

    // FIXME: only 1/INDIRECT_FINAL_COUNTER_COUNT of the image gets rendered
    if (data.w == 0.0) {
        const uvec3 TOTAL_SIZE = gl_NumWorkGroups * gl_WorkGroupSize;
        const uint TOTAL_LENGTH = TOTAL_SIZE.x * TOTAL_SIZE.y; // only two dimensions are used

        const uint INVOCATIONS_PER_COUNTER = TOTAL_LENGTH / INDIRECT_FINAL_COUNTER_COUNT;
        const uint GLOBAL_INVOCATION_INDEX = TOTAL_SIZE.x * gl_GlobalInvocationID.y + gl_GlobalInvocationID.x; // only two dimensions are used

        const uint COUNTER_INDEX = GLOBAL_INVOCATION_INDEX / INVOCATIONS_PER_COUNTER;

        uint bufIdx = atomicAdd(indirectFinalCounters.counters[COUNTER_INDEX], 1);
        if (bufIdx < INDIRECT_FINAL_ITEMS_PER_COUNTER) {
            indirectFinalBuffer.items[COUNTER_INDEX][bufIdx] = HitItem(ray.origin, ray.objectHit);
        }
    }

    imageStore(colorImage, ipos, vec4(data.rgb, 0.0));
}
