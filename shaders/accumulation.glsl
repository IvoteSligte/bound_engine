#version 460

#include "includes_general.glsl"

layout(local_size_x = LM_SAMPLES, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint OFFSET_USED = 0; // FIXME: CPU side

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

layout(binding = 2) uniform restrict readonly MutableData {
    Material mats[MAX_MATERIALS];
} buf;

layout(binding = 3, rgba16) uniform restrict image3D[LM_RAYS * LM_COUNT] lightmapImages;

// TODO: possibly merge binding 4, 5, 6 into a buffer

layout(binding = 4, r32ui) uniform restrict uimage3D[LM_COUNT] lightmapUsedImages; // TODO: maybe merge into one uimage3D

layout(binding = 5, r32ui) uniform restrict uimage3D[LM_COUNT] lightmapObjectHitImages;

layout(binding = 6, r32ui) uniform restrict uimage3D[LM_COUNT] lightmapLevelImages;

layout(binding = 7) uniform restrict readonly BlueNoise {
    vec4 items[LM_SAMPLES];
} bn;

#include "includes_trace_ray.glsl"

/// returns an index into a lightmap image in xyz, and the image index in w
ivec4 lightmapIndexAtPos(vec3 v) {
    const int HALF_LM_SIZE = LM_SIZE / 2;
    const float INV_HALF_LM_SIZE = 1.0 / (float(HALF_LM_SIZE) * LM_UNIT_SIZE);

    v -= rt.lightmapOrigin.xyz;
    uint lightmapNum = uint(log2(max(maximum(abs(v)) * INV_HALF_LM_SIZE, 0.500001)) + 1.0);

    ivec3 index = ivec3(floor(v / LM_UNIT_SIZES[lightmapNum])) + HALF_LM_SIZE;

    return ivec4(index, lightmapNum);
}

vec3 posAtLightmapIndex(ivec4 lmIndex) {
    const int HALF_LM_SIZE = LM_SIZE / 2;

    vec3 v = (lmIndex.xyz - (HALF_LM_SIZE + 0.5)) * LM_UNIT_SIZES[lmIndex.w] + rt.lightmapOrigin.xyz;

    return v;
}

struct Sample {
    vec3 color;
    bool isValid;
};

shared Sample SharedSamples[LM_SAMPLES];

void main() {
    const uint LIGHTMAP_LAYER = gl_WorkGroupID.x / (LM_SIZE / 32);
    const ivec3 LIGHTMAP_CHUNK = ivec3(gl_WorkGroupID.x % (LM_SIZE / 32), gl_WorkGroupID.yz); // TODO: do not dispatch for ignored chunks ([2, inf) layers in the middle)

    uint used = imageLoad(lightmapUsedImages[LIGHTMAP_LAYER], LIGHTMAP_CHUNK).x;
    const uint MASK = ALL_ONES << OFFSET_USED;

    uvec2 lsb = findLSB(uvec2(used & MASK, used)); // prioritizes unexplored lightmap voxels
    uint target = lsb.x == -1 ? lsb.y : lsb.x;

    ivec3 lmIndex = ivec3((32 * LIGHTMAP_CHUNK.x) + target, gl_WorkGroupID.yz);

    if (target == -1) {
        return;
    }

    uint nodeHitIndex = imageLoad(lightmapObjectHitImages[LIGHTMAP_LAYER], lmIndex.xyz).x;
    Bounds nodeHit = bvh.nodes[nodeHitIndex];
    vec3 point = posAtLightmapIndex(ivec4(lmIndex.xyz, LIGHTMAP_LAYER));
    vec3 normal = normalize(point - nodeHit.position);

    vec3 hitPoint = normal * nodeHit.radius + nodeHit.position;

    uint level = imageLoad(lightmapLevelImages[LIGHTMAP_LAYER], lmIndex.xyz).x;

    vec3 randDir = normalize(normal + bn.items[gl_LocalInvocationID.x].xyz);
    Ray ray = Ray(0, hitPoint, randDir, 0);

    traceRayWithBVH(ray); // bottleneck

    Sample samp = Sample(vec3(0.0), true);

    ivec4 lmIndexSample = lightmapIndexAtPos(ray.origin);
    uint levelSample = imageLoad(lightmapLevelImages[lmIndexSample.w], lmIndexSample.xyz).x;

    bool sampleLevelIsEnough = levelSample >= level;
    bool inRange = lmIndexSample.w < LM_COUNT;

    if (inRange) {
        if (sampleLevelIsEnough) {
            // TODO: improve level 0 stuff, its behaviour is different so moving it to a different shader might be useful.
            // otherwise, create an image and fill it with emission at points
            if (level == 0) {
                samp.color = buf.mats[ray.materialHit].emittance;
            } else {
                samp.color = imageLoad(lightmapImages[LM_COUNT * (level - 1) + lmIndexSample.w], lmIndexSample.xyz).rgb;
            }
        } else if (levelSample == 0) {
            // TODO: separate function
            ivec3 chunkSample = ivec3(lmIndexSample.x / 32, lmIndexSample.yz);
            uint usedSample = imageAtomicOr(lightmapUsedImages[lmIndexSample.w], chunkSample.xyz, 1 << (lmIndexSample.x % 32));

            if ((usedSample & (1 << (lmIndexSample.x % 32))) == 0) {
                imageStore(lightmapObjectHitImages[lmIndexSample.w], lmIndexSample.xyz, uvec4(ray.objectHit));
            }
        }
    }

    samp.isValid = !inRange || sampleLevelIsEnough;
    SharedSamples[gl_LocalInvocationID.x] = samp;

    barrier();

    if (gl_LocalInvocationID.x < 64) {
        for (uint i = 1; i < LM_SAMPLES / 64; i++) {
            Sample samp2 = SharedSamples[i * 64 + gl_LocalInvocationID.x];
            samp.color += samp2.color;
            samp.isValid = samp.isValid && samp2.isValid;
        }
        SharedSamples[gl_LocalInvocationID.x] = samp;
    }

    barrier();

    if (gl_LocalInvocationID.x == 0) {
        for (uint i = 1; i < 64; i++) {
            Sample samp2 = SharedSamples[i];
            samp.color += samp2.color;
            samp.isValid = samp.isValid && samp2.isValid;
        }

        if (samp.isValid) {
            Material material = buf.mats[nodeHit.material];
            samp.color = samp.color * (material.reflectance * (1.0 / LM_SAMPLES)) + material.emittance;

            imageStore(lightmapImages[LM_COUNT * level + LIGHTMAP_LAYER], lmIndex.xyz, vec4(samp.color, 0.0));

            // FIXME: (general) level keeps going up past LM_RAYS
            // FIXME: (general) level does not reach LM_RAYS
            imageStore(lightmapLevelImages[LIGHTMAP_LAYER], lmIndex.xyz, uvec4(level + 1));

            if (level + 1 == LM_RAYS) {
                // clear `target` bit
                imageAtomicAnd(lightmapUsedImages[LIGHTMAP_LAYER], LIGHTMAP_CHUNK, ALL_ONES ^ target);
            }
        }
    }
}
