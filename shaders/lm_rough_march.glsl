#version 460

#include "includes_general.glsl"

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) uniform restrict readonly RealTimeBuffer {
    vec4 rotation;
    vec4 previousRotation;
    vec3 position;
    vec3 previousPosition;
    ivec3 lightmapOrigin;
    uint lightmapBufferOffset;
    ivec4 deltaLightmapOrigins[LM_COUNT];
} rt;

layout(binding = 1) buffer restrict readonly LMBufferSlice {
    Voxel voxels[LM_VOXELS_PER_FRAME];
} lmBuffer;

layout(binding = 2) buffer restrict LMMarchBuffer {
    float dists[LM_VOXELS_PER_FRAME][64];
} lmMarchBuffer;

layout(binding = 3) uniform restrict readonly NoiseBuffer {
    vec4 dirs[LM_SAMPLES];
    vec4 midDirs[64];
} noise;

layout(binding = 4) uniform sampler3D SDFImages[LM_COUNT];

#include "includes_march_ray.glsl"

shared SharedStruct SharedData;
shared vec3 SharedColors[gl_WorkGroupSize.x];

void main() {
    if (gl_LocalInvocationID.x == 0) {
        SharedData = SharedStruct(lmBuffer.voxels[rt.lightmapBufferOffset + gl_WorkGroupID.x], rt.lightmapOrigin);
    }
    barrier();
    
    SharedStruct sData = SharedData;
    Voxel voxel = sData.voxel;
    ivec4 lmIndex = ivec4(unpackBytesUint(voxel.lmIndex));

    vec4 randDir = noise.midDirs[gl_LocalInvocationID.x];
    vec3 dir = normalize(voxel.normal + randDir.xyz);

    float totalDist = 2.0 * LM_UNIT_SIZES[lmIndex.w];
    vec3 position = voxel.position;
    bool isHit = marchRay(position, dir, sData.lightmapOrigin, MAX_RAY_RADIUS, 16, totalDist); // bottleneck

    lmMarchBuffer.dists[gl_WorkGroupID.x][gl_LocalInvocationID.x] = totalDist;
}
