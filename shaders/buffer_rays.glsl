#version 460

#include "includes_general.glsl"

layout(local_size_x = SAMPLES, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const int LIGHTMAP_INDEX = 0; // TODO: command buffers/pipelines with this

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

layout(binding = 4, rg32ui) uniform restrict uimage3D[LIGHTMAP_COUNT] lightmapSyncImages;

// TODO: correct bindings
layout(binding = 5) uniform restrict readonly BlueNoise {
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

vec3 posAtLightmapIndex(ivec4 lmIndex) {
    const int HALF_IMAGE_SIZE = imageSize(lightmapImages[0]).x >> 1;

    float unitSize = (1 << lmIndex.w) * LM_UNIT_SIZE;
    vec3 v = (lmIndex.xyz - HALF_IMAGE_SIZE) * unitSize;

    return v;
}

shared vec3 SharedColors[SAMPLES];
shared bool SharedIsValids[SAMPLES];

void main() {
    const ivec4 LM_INDEX = ivec4(gl_WorkGroupID.xyz, LIGHTMAP_INDEX);

    uvec2 misc = imageLoad(lightmapSyncImages[LM_INDEX.w], LM_INDEX.xyz).xy;
    uint sync = misc.x;
    uint objectHit = misc.y;
    bool isUnused = (sync & BIT_USED) == 0;
    
    if (isUnused) {
        return; // TODO: check if this does not mess up anything parallelised
    }

    uint level = sync & BITS_LEVEL;

    vec3 position = posAtLightmapIndex(LM_INDEX);

    vec3 normal = normalize(position - bvh.nodes[objectHit].position);

    vec3 randDir = normalize(normal + bn.items[gl_LocalInvocationID.x].xyz);
    Ray ray = Ray(0, position, randDir, 0);

    traceRayWithBVH(ray);

    ivec4 lmIndexSample = lightmapIndexAtPos(ray.origin);

    bool outOfRange = lmIndexSample.w >= LIGHTMAP_COUNT;
    if (outOfRange) {
        SharedColors[gl_LocalInvocationID.x] = vec3(0.0);
        SharedIsValids[gl_LocalInvocationID.x] = true;
    } else {
        uint syncSample = imageLoad(lightmapSyncImages[lmIndexSample.w], lmIndexSample.xyz).x;
        uint levelSample = syncSample & BITS_LEVEL;

        if (levelSample < level) {
            bool isUnused = (syncSample & BIT_USED) == 0;
            if (isUnused) {
                imageStore(lightmapSyncImages[lmIndexSample.w], lmIndexSample.xyz, uvec4(syncSample | BIT_USED, ray.objectHit, uvec2(0)));
            }

            SharedColors[gl_LocalInvocationID.x] = vec3(0.0);
            SharedIsValids[gl_LocalInvocationID.x] = false;
        } else {
            // TODO: improve level 0 stuff, its behaviour is different so moving it to a different shader might be useful
            // otherwise, create an image and fill it with emission at points
            if (level == 0) {
                Material material = buf.mats[ray.materialHit];
                SharedColors[gl_LocalInvocationID.x] = material.emittance;
            } else {
                SharedColors[gl_LocalInvocationID.x] = imageLoad(lightmapImages[LIGHTMAP_COUNT * (level - 1) + lmIndexSample.w], lmIndexSample.xyz).rgb;
            }

            SharedIsValids[gl_LocalInvocationID.x] = true;
        }
    }

    barrier();

    if (gl_LocalInvocationID.x < 64) {
        vec3 color = vec3(0.0);
        bool isValid = true;
        for (uint i = 0; i < SAMPLES / 64; i++) {
            color += SharedColors[i * 64 + gl_LocalInvocationID.x];
            isValid = isValid && SharedIsValids[i * 64 + gl_LocalInvocationID.x];
        }
        SharedColors[gl_LocalInvocationID.x] = color;
        SharedIsValids[gl_LocalInvocationID.x] = isValid;
    }

    barrier();

    if (gl_LocalInvocationID.x == 0) {
        vec3 color = vec3(0.0);
        bool isValid = true;
        for (uint i = 0; i < 64; i++) {
            color += SharedColors[i];
            isValid = isValid && SharedIsValids[i];
        }

        if (isValid) {
            Material material = buf.mats[bvh.nodes[objectHit].material]; // TODO: optimize by storing materialHit
            color = color * (material.reflectance * (1.0 / SAMPLES)) + material.emittance;

            imageStore(lightmapImages[LIGHTMAP_COUNT * level + LM_INDEX.w], LM_INDEX.xyz, vec4(color, 0.0));

            imageStore(lightmapSyncImages[LM_INDEX.w], LM_INDEX.xyz, uvec4(level + 1, 0, uvec2(0)));
        } else {
            imageStore(lightmapSyncImages[LM_INDEX.w], LM_INDEX.xyz, uvec4(level, 0, uvec2(0)));
        }
    }
}
