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

// sparse second order SH * second order SH multiplication cropped to a second order SH
// Von Neumann neighborhood (direct neighbors)
void propagateVonNeumann(inout vec3[SH_CS] coefs, ivec3 iil, int layer, float normalizer) {
    vec3[SH_CS] tCoefs;
    // -X
    tCoefs = loadSHCoefs(ivec3(iil.x-1, iil.yz), layer);
    madAssign(coefs, SH_cosLobe_C0 * normalizer, tCoefs);
    coefs[3] += -SH_cosLobe_C1 * normalizer * tCoefs[0];
    // +X
    tCoefs = loadSHCoefs(ivec3(iil.x+1, iil.yz), layer);
    madAssign(coefs, SH_cosLobe_C0 * normalizer, tCoefs);
    coefs[3] += SH_cosLobe_C1 * normalizer * tCoefs[0];
    // -Y
    tCoefs = loadSHCoefs(ivec3(iil.x, iil.y-1, iil.z), layer);
    madAssign(coefs, SH_cosLobe_C0 * normalizer, tCoefs);
    coefs[1] += -SH_cosLobe_C1 * normalizer * tCoefs[0];
    // +Y
    tCoefs = loadSHCoefs(ivec3(iil.x, iil.y+1, iil.z), layer);
    madAssign(coefs, SH_cosLobe_C0 * normalizer, tCoefs);
    coefs[1] += SH_cosLobe_C1 * normalizer * tCoefs[0];
    // -Z
    tCoefs = loadSHCoefs(ivec3(iil.xy, iil.z-1), layer);
    madAssign(coefs, SH_cosLobe_C0 * normalizer, tCoefs);
    coefs[2] += -SH_cosLobe_C1 * normalizer * tCoefs[0];
    // +Z
    tCoefs = loadSHCoefs(ivec3(iil.xy, iil.z+1), layer);
    madAssign(coefs, SH_cosLobe_C0 * normalizer, tCoefs);
    coefs[2] += SH_cosLobe_C1 * normalizer * tCoefs[0];
}

void propagateEdges(inout vec3[SH_CS] coefs, ivec3 iil, int layer, float normalizer) {
    const float INV_LENGTH = 1.0 / 1.414213562; // 1.0 / sqrt(2.0)
    float weight = normalizer * INV_LENGTH;
    vec3[SH_CS] tCoefs;

    // -X, -Y
    tCoefs = loadSHCoefs(iil + ivec3(-1, -1,  0), layer);
    madAssign(coefs, SH_cosLobe_C0 * weight, tCoefs);
    coefs[3] += -SH_cosLobe_C1 * weight * tCoefs[0];
    coefs[1] += -SH_cosLobe_C1 * weight * tCoefs[0];

    // -X, +Y
    tCoefs = loadSHCoefs(iil + ivec3(-1,  1,  0), layer);
    madAssign(coefs, SH_cosLobe_C0 * weight, tCoefs);
    coefs[3] += -SH_cosLobe_C1 * weight * tCoefs[0];
    coefs[1] +=  SH_cosLobe_C1 * weight * tCoefs[0];

    // +X, -Y
    tCoefs = loadSHCoefs(iil + ivec3( 1, -1,  0), layer);
    madAssign(coefs, SH_cosLobe_C0 * weight, tCoefs);
    coefs[3] +=  SH_cosLobe_C1 * weight * tCoefs[0];
    coefs[1] += -SH_cosLobe_C1 * weight * tCoefs[0];

    // +X, +Y
    tCoefs = loadSHCoefs(iil + ivec3( 1,  1,  0), layer);
    madAssign(coefs, SH_cosLobe_C0 * weight, tCoefs);
    coefs[3] +=  SH_cosLobe_C1 * weight * tCoefs[0];
    coefs[1] +=  SH_cosLobe_C1 * weight * tCoefs[0];

    // -X, -Z
    tCoefs = loadSHCoefs(iil + ivec3(-1,  0, -1), layer);
    madAssign(coefs, SH_cosLobe_C0 * weight, tCoefs);
    coefs[3] += -SH_cosLobe_C1 * weight * tCoefs[0];
    coefs[2] += -SH_cosLobe_C1 * weight * tCoefs[0];

    // -X, +Z
    tCoefs = loadSHCoefs(iil + ivec3(-1,  0,  1), layer);
    madAssign(coefs, SH_cosLobe_C0 * weight, tCoefs);
    coefs[3] += -SH_cosLobe_C1 * weight * tCoefs[0];
    coefs[2] +=  SH_cosLobe_C1 * weight * tCoefs[0];

    // +X, -Z
    tCoefs = loadSHCoefs(iil + ivec3( 1,  0, -1), layer);
    madAssign(coefs, SH_cosLobe_C0 * weight, tCoefs);
    coefs[3] +=  SH_cosLobe_C1 * weight * tCoefs[0];
    coefs[2] += -SH_cosLobe_C1 * weight * tCoefs[0];

    // +X, +Z
    tCoefs = loadSHCoefs(iil + ivec3( 1,  0,  1), layer);
    madAssign(coefs, SH_cosLobe_C0 * weight, tCoefs);
    coefs[3] +=  SH_cosLobe_C1 * weight * tCoefs[0];
    coefs[2] +=  SH_cosLobe_C1 * weight * tCoefs[0];

    // -Y, -Z
    tCoefs = loadSHCoefs(iil + ivec3( 0, -1, -1), layer);
    madAssign(coefs, SH_cosLobe_C0 * weight, tCoefs);
    coefs[1] += -SH_cosLobe_C1 * weight * tCoefs[0];
    coefs[2] += -SH_cosLobe_C1 * weight * tCoefs[0];

    // -Y, +Z
    tCoefs = loadSHCoefs(iil + ivec3( 0, -1,  1), layer);
    madAssign(coefs, SH_cosLobe_C0 * weight, tCoefs);
    coefs[1] += -SH_cosLobe_C1 * weight * tCoefs[0];
    coefs[2] +=  SH_cosLobe_C1 * weight * tCoefs[0];

    // +Y, -Z
    tCoefs = loadSHCoefs(iil + ivec3( 0,  1, -1), layer);
    madAssign(coefs, SH_cosLobe_C0 * weight, tCoefs);
    coefs[1] +=  SH_cosLobe_C1 * weight * tCoefs[0];
    coefs[2] += -SH_cosLobe_C1 * weight * tCoefs[0];

    // +Y, +Z
    tCoefs = loadSHCoefs(iil + ivec3( 0,  1,  1), layer);
    madAssign(coefs, SH_cosLobe_C0 * weight, tCoefs);
    coefs[1] +=  SH_cosLobe_C1 * weight * tCoefs[0];
    coefs[2] +=  SH_cosLobe_C1 * weight * tCoefs[0];
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

    const float BASE_FALLOFF = 0.2765; //0.178;
    float layer_falloff = pow(0.95, LAYER);
    float normalizer = SH_norm_C0 * BASE_FALLOFF * layer_falloff;

    propagateVonNeumann(coefs, IIL, LAYER, normalizer);
    propagateEdges(coefs, IIL, LAYER, normalizer);

    Voxel voxel = unpackVoxel(cache.voxels[LAYER][IIL.x][IIL.y][IIL.z]);

    // TODO: get a surface cache to handle diffuse reflections
    if (voxel.intersections > 0.0) {
        vec4 cosLobe = dirToCosineLobe(voxel.normal);
        vec3 s = voxel.reflectance * max(vec3(0.0), dot_coefs(cosLobe, coefs));
        coefs[0] = s *  cosLobe[0];
        coefs[1] = s * -cosLobe[1]; // opposite direction
        coefs[2] = s * -cosLobe[2];
        coefs[3] = s * -cosLobe[3];
    }

    coefs[0] += voxel.emittance;

    storeSHCoefs(IIL, LAYER, coefs);
}

// TODO: render in chunks based on workgroup sizes, a 4x4x4 block can then be used instead of the current awkward pattern
// improves memory reads
