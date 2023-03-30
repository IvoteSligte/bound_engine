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

layout(binding = 3, rgba8) uniform restrict image3D[RAYS_INDIRECT * LIGHTMAP_COUNT] lightmapImages;

layout(binding = 4, r32ui) uniform restrict uimage3D[LIGHTMAP_COUNT] lightmapSyncImages;

layout(binding = 5) buffer restrict readonly CurrBuffer {
    HitItem items[SUBBUFFER_COUNT][SUBBUFFER_LENGTH];
} currBuffer;

layout(binding = 6) buffer restrict CurrCounters {
    uint counters[SUBBUFFER_COUNT];
} currCounters;

layout(binding = 7) buffer restrict NextBuffer {
    HitItem items[SUBBUFFER_COUNT][SUBBUFFER_LENGTH];
} nextBuffer;

layout(binding = 8) buffer restrict NextCounters {
    uint counters[SUBBUFFER_COUNT];
} nextCounters;

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
    // FIXME: binding is invalid when the buffer is not read from
    HitItem useless = nextBuffer.items[0][0];

    const uvec4 BAKED_SEEDS = uvec4(
        gl_GlobalInvocationID.x * 432178 + 128 * 1,
        gl_GlobalInvocationID.x * 735741 + 128 * 57,
        gl_GlobalInvocationID.x * 578956 + 128 * 6,
        gl_GlobalInvocationID.x * 115817 + 128 * 11
    );

    uvec4 seeds = BAKED_SEEDS;

    const uint COUNTER_INDEX = gl_GlobalInvocationID.x / SUBBUFFER_LENGTH;
    const uint BUFFER_INDEX = gl_GlobalInvocationID.x % SUBBUFFER_LENGTH;

    // if buffer slot is empty
    if (BUFFER_INDEX >= currCounters.counters[COUNTER_INDEX]) {
        return;
    }

    HitItem hitItem = currBuffer.items[COUNTER_INDEX][BUFFER_INDEX];
    ivec4 lmIndex = lightmapIndexAtPos(hitItem.position);

    vec3 normal = normalize(hitItem.position - bvh.nodes[hitItem.objectHit].position);
    
    vec3 color = vec3(0.0);
    uint sync = imageLoad(lightmapSyncImages[lmIndex.w], lmIndex.xyz).x;
    uint level = sync & NOT_BIT_CALC;

    for (uint r = 0; r < SAMPLES; r++) {
        vec3 randDir = randomDirection(normal, seeds);
        Ray ray = Ray(0, hitItem.position, randDir);

        vec3 hitObjPosition;
        traceRayWithBVH(ray, hitObjPosition);

        ivec4 lmIndexSample = lightmapIndexAtPos(ray.origin);
        uint syncSample = imageAtomicOr(lightmapSyncImages[lmIndexSample.w], lmIndexSample.xyz, BIT_CALC);

        uint levelSample = syncSample & NOT_BIT_CALC;
        bool isUnused = (syncSample & BIT_CALC) == 0;

        if (levelSample < level) {
            if (isUnused) {
                const uint COUNTER_INDEX = gl_GlobalInvocationID.x / SUBBUFFER_LENGTH;

                uint bufIdx = atomicAdd(nextCounters.counters[COUNTER_INDEX], 1);
                if (bufIdx < SUBBUFFER_LENGTH) {
                    nextBuffer.items[COUNTER_INDEX][bufIdx] = HitItem(ray.origin, ray.objectHit);
                } else {
                    imageStore(lightmapSyncImages[lmIndexSample.w], lmIndexSample.xyz, uvec4(syncSample));
                }
            }

            imageStore(lightmapSyncImages[lmIndex.w], lmIndex.xyz, uvec4(level));
            return;
        }

        if (isUnused) {
            imageStore(lightmapSyncImages[lmIndexSample.w], lmIndexSample.xyz, uvec4(syncSample));
        }

        // TODO: improve level 0 stuff, its behaviour is different so moving it to a different shader might be useful
        // otherwise, create an image and fill it with emission at that point
        if (level == 0) {
            Material material = buf.mats[ray.objectHit];
            color += material.emittance;
        } else {
            color += imageLoad(lightmapImages[LIGHTMAP_COUNT * (level - 1) + lmIndexSample.w], lmIndexSample.xyz).rgb;
        }
    }

    Material material = buf.mats[hitItem.objectHit];
    color = color * (material.reflectance * (1.0 / SAMPLES)) + material.emittance;

    imageStore(lightmapImages[LIGHTMAP_COUNT * level + lmIndex.w], lmIndex.xyz, vec4(color, 0.0));

    uint storeValue = ((level + 1) == RAYS_INDIRECT) ? ((level + 1) | BIT_CALC) : (level + 1);
    imageStore(lightmapSyncImages[lmIndex.w], lmIndex.xyz, uvec4(storeValue));
}
