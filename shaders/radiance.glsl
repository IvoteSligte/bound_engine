#version 460

#extension GL_EXT_shader_atomic_float : enable

#include "includes_general.glsl"
#include "sh_rotation.glsl"

layout(constant_id = 0) const int CHECKERBOARD_OFFSET = 0;

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(binding = 0) buffer RadianceBuffer {
    Material materials[LM_LAYERS][RADIANCE_SIZE][RADIANCE_SIZE][RADIANCE_SIZE];
} cache;

layout(binding = 1, rgba16f) uniform image3D[LM_LAYERS * 4] radianceImages;

vec3[4] loadSHCoefs(ivec3 index, int layer) {
    vec3[4] coefs;
    for (int i = 0; i < 4; i++) {
        coefs[i] = imageLoad(radianceImages[i * LM_LAYERS + layer], index).rgb;
    }
    return coefs;
}

void storeSHCoefs(ivec3 index, int layer, vec3[4] coefs) {
    for (int i = 0; i < 4; i++) {
        imageStore(radianceImages[i * LM_LAYERS + layer], index, vec4(coefs[i], 0.0));
    }
}

void madAssign(inout vec3[4] dst, float multiplier, vec3[4] additive) {
    for (int i = 0; i < 4; i++) {
        dst[i] += additive[i] * multiplier;
    }
}

void main() {
    const int LAYER = int(gl_GlobalInvocationID.x / RADIANCE_SIZE);
    // index in layer
    const ivec3 IIL = ivec3(
        gl_GlobalInvocationID.x % RADIANCE_SIZE,
        gl_GlobalInvocationID.y * 2 + (gl_GlobalInvocationID.x + gl_GlobalInvocationID.z + CHECKERBOARD_OFFSET) % 2, // checkerboard transform
        gl_GlobalInvocationID.z);

    // stores coefs as array of RGB channels
    vec3[4] coefs = vec3[](vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0));

    // -X
    if (IIL.x != 0) {
        vec3[4] rCoefs = loadSHCoefs(ivec3(IIL.x-1, IIL.yz), LAYER);
        madAssign(coefs, SH_cosLobe_C0, rCoefs);
        coefs[3] += SH_cosLobe_C1 * rCoefs[0];
    }
    // +X
    if (IIL.x != RADIANCE_SIZE - 1) {
        vec3[4] rCoefs = loadSHCoefs(ivec3(IIL.x+1, IIL.yz), LAYER);
        madAssign(coefs, SH_cosLobe_C0, rCoefs);
        coefs[3] -= SH_cosLobe_C1 * rCoefs[0];
    }
    // -Y
    if (IIL.y != 0) {
        vec3[4] rCoefs = loadSHCoefs(ivec3(IIL.x, IIL.y-1, IIL.z), LAYER);
        madAssign(coefs, SH_cosLobe_C0, rCoefs);
        coefs[1] += SH_cosLobe_C1 * rCoefs[0];
    }
    // +Y
    if (IIL.y != RADIANCE_SIZE - 1) {
        vec3[4] rCoefs = loadSHCoefs(ivec3(IIL.x, IIL.y+1, IIL.z), LAYER);
        madAssign(coefs, SH_cosLobe_C0, rCoefs);
        coefs[1] -= SH_cosLobe_C1 * rCoefs[0];
    }
    if (IIL.z != 0) {
        vec3[4] rCoefs = loadSHCoefs(ivec3(IIL.xy, IIL.z-1), LAYER);
        madAssign(coefs, SH_cosLobe_C0, rCoefs);
        coefs[2] -= SH_cosLobe_C1 * rCoefs[0];
    }
    if (IIL.z != RADIANCE_SIZE - 1) {
        vec3[4] rCoefs = loadSHCoefs(ivec3(IIL.xy, IIL.z+1), LAYER);
        madAssign(coefs, SH_cosLobe_C0, rCoefs);
        coefs[2] += SH_cosLobe_C1 * rCoefs[0];
    }

    // TODO: cosLobe diffuse reflection stuff
    vec3 normal = vec3(0.0, 0.0, 1.0); // TODO: get (average?) normal in voxel from scene
    vec4 cosLobe = dirToCosineLobe(normal);

    coefs[0] += cache.materials[LAYER][IIL.x][IIL.y][IIL.z].emittance;

    // FIXME: minimal BASE_FALLOFF value where the radiance doesn't diverge is dependent on either the number of tiles in a layer or the tile size
    const float BASE_FALLOFF = 0.185;
    float distFallOff = radUnitSizeLayer(0) / radUnitSizeLayer(LAYER);
    for (int i = 0; i < 4; i++) {
        coefs[i] *= BASE_FALLOFF * distFallOff;
    }

    storeSHCoefs(IIL, LAYER, coefs);
}
