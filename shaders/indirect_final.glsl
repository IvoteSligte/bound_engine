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

layout(binding = 3, rgba8) uniform restrict writeonly image3D[LIGHTMAP_COUNT] lightmapFinalImages;

layout(binding = 4, rgba8) uniform restrict readonly image3D[LIGHTMAP_COUNT] lightmapTrueImages;

layout(binding = 5) buffer restrict readonly IndirectFinalBuffer {
    HitItem items[INDIRECT_FINAL_COUNTER_COUNT][INDIRECT_FINAL_ITEMS_PER_COUNTER];
} indirectFinalBuffer;

layout(binding = 6) buffer restrict IndirectFinalCounters {
    uint counters[INDIRECT_FINAL_COUNTER_COUNT];
} indirectFinalCounters;

layout(binding = 7) buffer restrict IndirectTrueBuffer {
    HitItem items[INDIRECT_TRUE_COUNTER_COUNT][INDIRECT_TRUE_ITEMS_PER_COUNTER];
} indirectTrueBuffer;

layout(binding = 8) buffer restrict IndirectTrueCounters {
    uint counters[INDIRECT_TRUE_COUNTER_COUNT];
} indirectTrueCounters;

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
    // FIXME: the binding is not recognised when 'indirectTrueBuffer' is only written to and not read from
    HitItem useless = indirectTrueBuffer.items[0][0];

    const uvec4 BAKED_SEEDS = uvec4(
        gl_GlobalInvocationID.x * 43217 + 128 * 1,
        gl_GlobalInvocationID.x * 73574 + 128 * 57,
        gl_GlobalInvocationID.x * 57895 + 128 * 6,
        gl_GlobalInvocationID.x * 11581 + 128 * 11
    );

    uvec4 seeds = BAKED_SEEDS;

    const uint COUNTER_INDEX_FINAL = gl_GlobalInvocationID.x / INDIRECT_FINAL_ITEMS_PER_COUNTER;
    const uint BUFFER_INDEX = gl_GlobalInvocationID.x % INDIRECT_FINAL_ITEMS_PER_COUNTER;

    // if buffer slot is empty
    if (BUFFER_INDEX >= indirectFinalCounters.counters[COUNTER_INDEX_FINAL]) {
        return;
    }

    HitItem hitItem = indirectFinalBuffer.items[COUNTER_INDEX_FINAL][BUFFER_INDEX];

    vec3 normal = normalize(hitItem.position - bvh.nodes[hitItem.objectHit].position);

    vec3 color = vec3(0.0);
    bool isValid = true;

    for (uint r = 0; r < INDIRECT_FINAL_SAMPLES; r++) {
        vec3 randDir = randomDirection(normal, seeds);
        Ray ray = Ray(0, hitItem.position, randDir);

        vec3 hitObjPosition;
        traceRayWithBVH(ray, hitObjPosition);

        ivec4 index = lightmapIndexAtPos(ray.origin);
        vec4 data = imageLoad(lightmapTrueImages[index.w], index.xyz);

        if (data.w == 0.0) {
            const uint COUNTER_INDEX_TRUE = gl_GlobalInvocationID.x / INDIRECT_TRUE_ITEMS_PER_COUNTER;

            uint bufIdx = atomicAdd(indirectTrueCounters.counters[COUNTER_INDEX_TRUE], 1);
            if (bufIdx < INDIRECT_TRUE_ITEMS_PER_COUNTER) {
                indirectTrueBuffer.items[COUNTER_INDEX_TRUE][bufIdx] = HitItem(ray.origin, ray.objectHit);
            }

            isValid = false;
        } else {
            color += data.rgb;
        }
    }

    if (isValid) {
        Material material = buf.mats[hitItem.objectHit];
        color = (color * (1.0 / INDIRECT_FINAL_SAMPLES)) * material.reflectance + material.emittance;
        
        ivec4 index = lightmapIndexAtPos(hitItem.position);

        imageStore(lightmapFinalImages[index.w], index.xyz, vec4(color, 1.0));
    }
}
