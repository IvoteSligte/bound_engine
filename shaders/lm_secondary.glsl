#version 460

#include "includes_general.glsl"

layout(local_size_x = LM_SAMPLES / 4, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) uniform restrict readonly RealTimeBuffer {
    vec4 rotation;
    vec4 previousRotation;
    vec3 position;
    vec3 previousPosition;
    ivec3 lightmapOrigin;
    ivec4 deltaLightmapOrigins[LM_COUNT];
    uint frame;
} rt;

layout(binding = 1) uniform restrict readonly ObjectBuffer {
    Object objects[MAX_OBJECTS];
} objBuffer;

layout(binding = 2) uniform restrict readonly MutableData {
    Material mats[MAX_MATERIALS];
} buf;

layout(binding = 3, rgba16) uniform restrict readonly image3D[LM_COUNT] lmInputColorImages;

layout(binding = 4, rgba16) uniform restrict writeonly image3D[LM_COUNT] lmOutputColorImages;

layout(binding = 5) buffer restrict readonly LMBuffer {
    Voxel voxels[LM_SIZE * LM_SIZE * LM_SIZE * LM_COUNT];
} lmBuffer;

layout(binding = 6) uniform restrict readonly NoiseBuffer {
    vec4 dirs[gl_WorkGroupSize.x][4];
} noise;

#include "includes_trace_ray.glsl"

shared SharedStruct SharedData;
shared vec3 SharedColors[gl_WorkGroupSize.x];

RayResult[4] traceRay4(vec3 origin, mat4x3 dirs) {
    RayResult results[4] = RayResult[4](
        RayResult(FLT_MAX, 0, 0),
        RayResult(FLT_MAX, 0, 0),
        RayResult(FLT_MAX, 0, 0),
        RayResult(FLT_MAX, 0, 0)
    );

    for (uint i = 0; i < MAX_OBJECTS; i++) {
        Object obj = objBuffer.objects[i]; // TODO: shared memory

        float[4] dists = distanceToObject4(origin, dirs, obj);

        for (uint j = 0; j < 4; i++) {
            if (dists[j] > EPSILON && dists[j] < results[j].distanceToHit) {
                // is a leaf, store data
                results[j] = RayResult(dists[i], i, obj.material);
            }
        }
    }

    return results;
}

void main() {
    if (gl_LocalInvocationID.x == 0) {
        SharedData = SharedStruct(lmBuffer.voxels[gl_WorkGroupID.x], rt.lightmapOrigin);
    }
    barrier();
    
    SharedStruct sData = SharedData;
    Voxel voxel = sData.voxel;

    vec4 randDirs[4] = noise.dirs[gl_LocalInvocationID.x]; // FIXME: figure out how to pass a mat4x3 directly to the shader

    mat4x3 dirMat = mat4x3(
        normalize(voxel.normal + randDirs[0].xyz),
        normalize(voxel.normal + randDirs[1].xyz),
        normalize(voxel.normal + randDirs[2].xyz),
        normalize(voxel.normal + randDirs[3].xyz)
    );

    RayResult results[4] = traceRay4(voxel.hitPoint, dirMat); // bottleneck

    vec3 color = vec3(0.0);
    for (uint i = 0; i < 4; i++) {
        vec3 p = (dirMat[i] * results[i].distanceToHit) + voxel.hitPoint;
        ivec4 lmIndexSample = lightmapIndexAtPos(p, sData.lightmapOrigin);
        color += imageLoad(lmInputColorImages[lmIndexSample.w], lmIndexSample.xyz).rgb; // TODO: texture access for smoother results
    }

    SharedColors[gl_LocalInvocationID.x] = color;

    barrier();

    if (gl_LocalInvocationID.x < 64) {
        for (uint i = 1; i < gl_WorkGroupSize.x / 64; i++) {
            color += SharedColors[i * 64 + gl_LocalInvocationID.x];
        }
        SharedColors[gl_LocalInvocationID.x] = color;
    }

    barrier();

    if (gl_LocalInvocationID.x == 0) {
        for (uint i = 1; i < 64; i++) {
            color += SharedColors[i];
        }

        Material material = buf.mats[voxel.materialHit];
        color = color * (material.reflectance * (1.0 / LM_SAMPLES) + material.emittance);

        imageStore(lmOutputColorImages[voxel.lmIndex.w], voxel.lmIndex.xyz, vec4(color, 0.0));
    }
}
