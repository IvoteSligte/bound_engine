#version 460

#extension GL_EXT_shader_atomic_float : enable

#include "includes_general.glsl"
#include "sh_rotation.glsl"

layout(constant_id = 0) const int CHECKERBOARD_OFFSET = 0;

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(binding = 0) buffer RadianceBuffer {
    Radiance radiances[LM_LAYERS][RADIANCE_SIZE][RADIANCE_SIZE][RADIANCE_SIZE];
    Material materials[LM_LAYERS][RADIANCE_SIZE][RADIANCE_SIZE][RADIANCE_SIZE];
} cache;

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
        vec3[4] rCoefs = unpackSHCoefs(cache.radiances[LAYER][IIL.x-1][IIL.y][IIL.z].sh);
        for (int i = 0; i < 4; i++) {
            coefs[i] += SH_cosLobe_C0 * rCoefs[i];
        }
        coefs[3] += SH_cosLobe_C1 * rCoefs[0];
    }
    // +X
    if (IIL.x != RADIANCE_SIZE - 1) {
        vec3[4] rCoefs = unpackSHCoefs(cache.radiances[LAYER][IIL.x+1][IIL.y][IIL.z].sh);
        for (int i = 0; i < 4; i++) {
            coefs[i] += SH_cosLobe_C0 * rCoefs[i];
        }
        coefs[3] -= SH_cosLobe_C1 * rCoefs[0];
    }
    // -Y
    if (IIL.y != 0) {
        vec3[4] rCoefs = unpackSHCoefs(cache.radiances[LAYER][IIL.x][IIL.y-1][IIL.z].sh);
        for (int i = 0; i < 4; i++) {
            coefs[i] += SH_cosLobe_C0 * rCoefs[i];
        }
        coefs[1] += SH_cosLobe_C1 * rCoefs[0];
    }
    // +Y
    if (IIL.y != RADIANCE_SIZE - 1) {
        vec3[4] rCoefs = unpackSHCoefs(cache.radiances[LAYER][IIL.x][IIL.y+1][IIL.z].sh);
        for (int i = 0; i < 4; i++) {
            coefs[i] += SH_cosLobe_C0 * rCoefs[i];
        }
        coefs[1] -= SH_cosLobe_C1 * rCoefs[0];
    }
    if (IIL.z != 0) {
        vec3[4] rCoefs = unpackSHCoefs(cache.radiances[LAYER][IIL.x][IIL.y][IIL.z-1].sh);
        for (int i = 0; i < 4; i++) {
            coefs[i] += SH_cosLobe_C0 * rCoefs[i];
        }
        coefs[2] -= SH_cosLobe_C1 * rCoefs[0];
    }
    if (IIL.z != RADIANCE_SIZE - 1) {
        vec3[4] rCoefs = unpackSHCoefs(cache.radiances[LAYER][IIL.x][IIL.y][IIL.z+1].sh);
        for (int i = 0; i < 4; i++) {
            coefs[i] += SH_cosLobe_C0 * rCoefs[i];
        }
        coefs[2] += SH_cosLobe_C1 * rCoefs[0];
    }

    coefs[0] += cache.materials[LAYER][IIL.x][IIL.y][IIL.z].emittance;

    vec3 normal = vec3(0.0, 0.0, 1.0); // TODO: get (average?) normal in voxel from scene

    vec4 cosLobe = dirToCosineLobe(normal);

    vec4[3] tempCoefs = vec4[](
        vec4(coefs[0].x, coefs[1].x, coefs[2].x, coefs[3].x),
        vec4(coefs[0].y, coefs[1].y, coefs[2].y, coefs[3].y),
        vec4(coefs[0].z, coefs[1].z, coefs[2].z, coefs[3].z)
        // cosLobe * dot(cosLobe, vec4(coefs[0].x, coefs[1].x, coefs[2].x, coefs[3].x)),
        // cosLobe * dot(cosLobe, vec4(coefs[0].y, coefs[1].y, coefs[2].y, coefs[3].y)),
        // cosLobe * dot(cosLobe, vec4(coefs[0].z, coefs[1].z, coefs[2].z, coefs[3].z))
    );
    coefs = vec3[](
        vec3(tempCoefs[0].x, tempCoefs[1].x, tempCoefs[2].x),
        vec3(tempCoefs[0].y, tempCoefs[1].y, tempCoefs[2].y),
        vec3(tempCoefs[0].z, tempCoefs[1].z, tempCoefs[2].z),
        vec3(tempCoefs[0].w, tempCoefs[1].w, tempCoefs[2].w)
    );

    for (int i = 0; i < 4; i++) {
        // FIXME: minimal BASE_FALLOFF value where the radiance doesn't diverge is dependent on either the number of tiles in a layer or the tile size
        const float BASE_FALLOFF = 0.185;
        float distFallOff = radUnitSizeLayer(0) / radUnitSizeLayer(LAYER);
        coefs[i] *= BASE_FALLOFF * distFallOff;
    }
    cache.radiances[LAYER][IIL.x][IIL.y][IIL.z].sh = packSHCoefs(coefs);
}
