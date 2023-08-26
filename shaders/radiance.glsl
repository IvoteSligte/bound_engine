#version 460

#extension GL_EXT_shader_atomic_float : enable

#include "includes_general.glsl"
#include "sh_rotation.glsl"

layout(constant_id = 0) const int OFFSET_X = 0;
layout(constant_id = 1) const int OFFSET_Y = 0;
layout(constant_id = 2) const int OFFSET_Z = 0;

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(binding = 0) buffer RadianceBuffer {
    PackedVoxel voxels[LM_LAYERS][RADIANCE_SIZE][RADIANCE_SIZE][RADIANCE_SIZE];
} cache;

layout(binding = 1, rgba16f) uniform image3D radianceImages[LM_LAYERS * SH_CS];

vec3[SH_CS] loadSHCoefs(ivec3 index, int layer) {
    vec3[SH_CS] coefs;
    for (int i = 0; i < SH_CS; i++) {
        coefs[i] = imageLoad(radianceImages[layer * LM_LAYERS + i], index).rgb;
    }
    return coefs;
}

void storeSHCoefs(ivec3 index, int layer, vec3[SH_CS] coefs) {
    for (int i = 0; i < SH_CS; i++) {
        imageStore(radianceImages[layer * LM_LAYERS + i], index, vec4(coefs[i], 0.0));
    }
}

void mulAssign(inout vec3[SH_CS] dst, float multiplier) {
    for (int i = 0; i < SH_CS; i++) {
        dst[i] *= multiplier;
    }
}

void madAssign(inout vec3[SH_CS] dst, float multiplier, vec3[SH_CS] additive) {
    for (int i = 0; i < SH_CS; i++) {
        dst[i] += additive[i] * multiplier;
    }
}

vec3 dot_coefs(vec4 a, vec3[SH_CS] b) {
    vec3 result = a[0] * b[0];
    for (int i = 1; i < SH_CS; i++) {
        result += a[i] * b[i];
    }
    return result;
}

void main() {
    const int LAYER = int(gl_GlobalInvocationID.x / RADIANCE_SIZE);
    // index in layer
    const ivec3 IIL = ivec3(
        (gl_GlobalInvocationID.x * 2 + OFFSET_X) % RADIANCE_SIZE,
        gl_GlobalInvocationID.y * 2 + OFFSET_Y,
        gl_GlobalInvocationID.z * 2 + OFFSET_Z);

    // stores coefs as array of RGB channels
    vec3[SH_CS] coefs = vec3[](vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0));
    vec3[SH_CS] tCoefs;

    const float BASE_FALLOFF = 2.0 / 3.0;
    float layer_falloff = pow(0.95, LAYER);
    float normalizer = SH_norm_C0 * BASE_FALLOFF * layer_falloff;

    // sparse second order SH * second order SH multiplication cropped to a second order SH
    // -X
    tCoefs = loadSHCoefs(ivec3(IIL.x-1, IIL.yz), LAYER);
    madAssign(coefs, SH_cosLobe_C0 * normalizer, tCoefs);
    coefs[3] += -SH_cosLobe_C1 * normalizer * tCoefs[0];
    // +X
    tCoefs = loadSHCoefs(ivec3(IIL.x+1, IIL.yz), LAYER);
    madAssign(coefs, SH_cosLobe_C0 * normalizer, tCoefs);
    coefs[3] += SH_cosLobe_C1 * normalizer * tCoefs[0];
    // -Y
    tCoefs = loadSHCoefs(ivec3(IIL.x, IIL.y-1, IIL.z), LAYER);
    madAssign(coefs, SH_cosLobe_C0 * normalizer, tCoefs);
    coefs[1] += -SH_cosLobe_C1 * normalizer * tCoefs[0];
    // +Y
    tCoefs = loadSHCoefs(ivec3(IIL.x, IIL.y+1, IIL.z), LAYER);
    madAssign(coefs, SH_cosLobe_C0 * normalizer, tCoefs);
    coefs[1] += SH_cosLobe_C1 * normalizer * tCoefs[0];
    // -Z
    tCoefs = loadSHCoefs(ivec3(IIL.xy, IIL.z-1), LAYER);
    madAssign(coefs, SH_cosLobe_C0 * normalizer, tCoefs);
    coefs[2] += -SH_cosLobe_C1 * normalizer * tCoefs[0];
    // +Z
    tCoefs = loadSHCoefs(ivec3(IIL.xy, IIL.z+1), LAYER);
    madAssign(coefs, SH_cosLobe_C0 * normalizer, tCoefs);
    coefs[2] += SH_cosLobe_C1 * normalizer * tCoefs[0];

    Voxel voxel = unpackVoxel(cache.voxels[LAYER][IIL.x][IIL.y][IIL.z]);

    // TODO: maybe allow multiple normals per voxel
    if (voxel.intersections > 0.0) {
        vec4 cosLobe = dirToCosineLobe(voxel.normal);

        if (voxel.intersections > 1.0 && (dot(ceil(abs(voxel.normal)), vec3(1.0)) > 1.0)) {
            vec3 sig = round(voxel.normal);

            tCoefs = loadSHCoefs(IIL + ivec3(sig), LAYER);
            madAssign(coefs, SH_cosLobe_C0 * SH_norm_C0, tCoefs);
            coefs[3] += cosLobe.x * SH_norm_C0 * tCoefs[0]; // X
            coefs[1] += cosLobe.y * SH_norm_C0 * tCoefs[0]; // Y
            coefs[2] += cosLobe.z * SH_norm_C0 * tCoefs[0]; // Z

            mulAssign(coefs, 6.0 / 7.0);
        }

        vec3 s = voxel.reflectance * max(vec3(0.0), dot_coefs(cosLobe, coefs));
        coefs[0] = s *  cosLobe[0];
        coefs[1] = s * -cosLobe[1]; // opposite direction
        coefs[2] = s * -cosLobe[2];
        coefs[3] = s * -cosLobe[3];
    }

    coefs[0] += voxel.emittance;

    storeSHCoefs(IIL, LAYER, coefs);
}

// TODO: use atomics for diagonal reads and change the rendering pattern to checkerboard rendering
