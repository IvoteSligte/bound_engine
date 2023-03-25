#version 460

#include "includes_general.glsl"

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

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

layout(binding = 2) uniform restrict readonly MutableData {
    Material mats[MAX_MATERIALS];
} buf;

layout(binding = 3, rgba8) uniform restrict writeonly image3D[LIGHTMAP_COUNT] lightmapMidImages;

layout(binding = 4, rgba8) uniform restrict readonly image3D[LIGHTMAP_COUNT] lightmapLastImages;

layout(binding = 5, r32ui) uniform restrict uimage3D[LIGHTMAP_COUNT] lightmapSyncImages;

layout(binding = 6) buffer restrict readonly MidBuffer {
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
    // FIXME: the binding is not recognised when 'lastBuffer' is only written to and not read from
    HitItem useless = lastBuffer.items[0][0];

    const uvec4 BAKED_SEEDS = uvec4(
        gl_GlobalInvocationID.x * 43217 + 128 * 1,
        gl_GlobalInvocationID.x * 73574 + 128 * 57,
        gl_GlobalInvocationID.x * 57895 + 128 * 6,
        gl_GlobalInvocationID.x * 11581 + 128 * 11
    );

    uvec4 seeds = BAKED_SEEDS;

    const uint COUNTER_INDEX_MID = gl_GlobalInvocationID.x / MID_SUBBUFFER_LENGTH;
    const uint BUFFER_INDEX = gl_GlobalInvocationID.x % MID_SUBBUFFER_LENGTH;

    // if buffer slot is empty
    if (BUFFER_INDEX >= midCounters.counters[COUNTER_INDEX_MID]) {
        return;
    }

    HitItem hitItem = midBuffer.items[COUNTER_INDEX_MID][BUFFER_INDEX];
    ivec4 lmIndex = lightmapIndexAtPos(hitItem.position);

    vec3 normal = normalize(hitItem.position - bvh.nodes[hitItem.objectHit].position);

    vec3 color = vec3(0.0);
    bool isValid = true;

    for (uint r = 0; r < MID_SAMPLES; r++) {
        vec3 randDir = randomDirection(normal, seeds);
        Ray ray = Ray(0, hitItem.position, randDir);

        vec3 hitObjPosition;
        traceRayWithBVH(ray, hitObjPosition);

        ivec4 lmIndex = lightmapIndexAtPos(ray.origin);

        uint syncValue = imageAtomicOr(lightmapSyncImages[lmIndex.w], lmIndex.xyz, 2);
        // there is a sample
        if ((syncValue & 4) != 0) { // FIXME: (syncValue & (4 | 2)) != 0 gives a better result (but not correct either)
            color += imageLoad(lightmapLastImages[lmIndex.w], lmIndex.xyz).rgb;
            continue;
        }

        // there is no sample (4)
        isValid = false;

        // no sample (4), and not being calculated (2)
        if ((syncValue & (2 | 4)) == 0) {
            const uint TOTAL_LENGTH = gl_NumWorkGroups.x * gl_WorkGroupSize.y;
            const uint INVOCATIONS_PER_COUNTER = TOTAL_LENGTH / LAST_SUBBUFFER_COUNT;
            const uint COUNTER_INDEX_LAST = gl_GlobalInvocationID.x / LAST_SUBBUFFER_LENGTH;

            uint bufIdx = atomicAdd(lastCounters.counters[COUNTER_INDEX_LAST], 1);
            if (bufIdx < LAST_SUBBUFFER_LENGTH) {
                lastBuffer.items[COUNTER_INDEX_LAST][bufIdx] = HitItem(ray.origin, ray.objectHit);
            } else {
                imageAtomicAnd(lightmapSyncImages[lmIndex.w], lmIndex.xyz, 1);
                break;
            }
        }
    }

    if (isValid) {
        Material material = buf.mats[hitItem.objectHit];
        color = (color * (1.0 / MID_SAMPLES)) * material.reflectance + material.emittance;

        imageStore(lightmapMidImages[lmIndex.w], lmIndex.xyz, vec4(color, 0.0));
    } else {
        // set bit 1 to false, indicating there are no 'final' samples
        imageAtomicAnd(lightmapSyncImages[lmIndex.w], lmIndex.xyz, 2 | 4);
    }
}
