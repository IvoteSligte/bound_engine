#version 460

#include "includes_general.glsl"

layout(local_size_x = SAMPLES, local_size_y = 1, local_size_z = 1) in;

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

layout(binding = 3, rgba16) uniform restrict image3D[RAYS_INDIRECT * LIGHTMAP_COUNT] lightmapImages;

layout(binding = 4, r32ui) uniform restrict uimage3D[LIGHTMAP_COUNT] lightmapSyncImages;

layout(binding = 5) buffer restrict CurrBuffer {
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

layout(binding = 9) uniform restrict readonly BlueNoise {
    vec4 items[SAMPLES];
} bn;

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

struct Sample {
    vec3 color;
    bool isValid;
};

shared Sample SharedSamples[SAMPLES];

void main() {
    // FIXME: binding is invalid when the buffer is not read from
    HitItem useless = nextBuffer.items[0][0];

    const uint COUNTER_INDEX = gl_WorkGroupID.x / SUBBUFFER_LENGTH;
    const uint BUFFER_INDEX = gl_WorkGroupID.x % SUBBUFFER_LENGTH;

    // if buffer slot is empty
    if (BUFFER_INDEX >= currCounters.counters[COUNTER_INDEX]) {
        return;
    }

    HitItem hitItem = currBuffer.items[COUNTER_INDEX][BUFFER_INDEX];
    ivec4 lmIndex = lightmapIndexAtPos(hitItem.position);

    vec3 normal = normalize(hitItem.position - bvh.nodes[hitItem.objectHit].position);

    uint sync = imageLoad(lightmapSyncImages[lmIndex.w], lmIndex.xyz).x;
    uint level = sync & BITS_LEVEL;

    vec3 randDir = normalize(normal + bn.items[gl_LocalInvocationID.x].xyz);
    Ray ray = Ray(0, hitItem.position, randDir, 0);

    traceRayWithBVH(ray); // bottleneck

    ivec4 lmIndexSample = lightmapIndexAtPos(ray.origin);

    vec3 color = vec3(0.0);
    bool isValid = true;

    bool outOfRange = lmIndexSample.w >= LIGHTMAP_COUNT;
    if (!outOfRange) {
        uint syncSample = imageAtomicOr(lightmapSyncImages[lmIndexSample.w], lmIndexSample.xyz, BIT_USED);

        uint levelSample = syncSample & BITS_LEVEL;
        bool isUnused = (syncSample & BIT_USED) == 0;

        // TODO: if (levelSample < level && oneOrMoreSamplesGatheredAreInvalid) -> add to buffer
        // TODO: else if (levelSample == level && allSamplesGatheredAreValid) -> add to buffer
        if (isUnused) {
            if (levelSample >= level) {
                imageStore(lightmapSyncImages[lmIndexSample.w], lmIndexSample.xyz, uvec4(syncSample));
            } else {
                const uint COUNTER_INDEX = gl_WorkGroupID.x / SUBBUFFER_LENGTH;

                uint bufIdx = atomicAdd(nextCounters.counters[COUNTER_INDEX], 1);
                if (bufIdx < SUBBUFFER_LENGTH) {
                    nextBuffer.items[COUNTER_INDEX][bufIdx] = HitItem(ray.origin, ray.objectHit, ray.materialHit);
                } else {
                    imageStore(lightmapSyncImages[lmIndexSample.w], lmIndexSample.xyz, uvec4(syncSample));
                }
            }
        }

        if (levelSample >= level) {
            // TODO: improve level 0 stuff, its behaviour is different so moving it to a different shader might be useful
            // otherwise, create an image and fill it with emission at points
            if (level == 0) {
                Material material = buf.mats[ray.materialHit];
                color = material.emittance;
            } else {
                color = imageLoad(lightmapImages[LIGHTMAP_COUNT * (level - 1) + lmIndexSample.w], lmIndexSample.xyz).rgb;
            }
        }

        isValid = levelSample >= level;
    }

    SharedSamples[gl_LocalInvocationID.x] = Sample(color, isValid);

    barrier();

    if (gl_LocalInvocationID.x < 64) {
        for (uint i = 1; i < SAMPLES / 64; i++) {
            Sample samp = SharedSamples[i * 64 + gl_LocalInvocationID.x];
            color += samp.color;
            isValid = isValid && samp.isValid;
        }
        SharedSamples[gl_LocalInvocationID.x] = Sample(color, isValid);
    }

    barrier();

    if (gl_LocalInvocationID.x == 0) {
        for (uint i = 1; i < 64; i++) {
            Sample samp = SharedSamples[i];
            color += samp.color;
            isValid = isValid && samp.isValid;
        }

        if (isValid) {
            Material material = buf.mats[hitItem.materialHit];
            color = color * (material.reflectance * (1.0 / SAMPLES)) + material.emittance;

            imageStore(lightmapImages[LIGHTMAP_COUNT * level + lmIndex.w], lmIndex.xyz, vec4(color, 0.0));

            // TODO: merge all ray bounces into one lightmap, then remove BIT_USED here for infinite bounces
            uint storeValue = ((level + 1) == RAYS_INDIRECT) ? ((level + 1) | BIT_USED) : (level + 1);
            imageStore(lightmapSyncImages[lmIndex.w], lmIndex.xyz, uvec4(storeValue));
        } else {
            uint storeValue = level;
            imageStore(lightmapSyncImages[lmIndex.w], lmIndex.xyz, uvec4(storeValue));
        }
    }
}
