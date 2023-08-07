const uint ALL_ONES = 4294967295;

const float LM_UNIT_SIZE = 0.5; // TODO: sync this with the rust code (defines)

#define FLT_MAX 3.402823466e+38

#define EPSILON 1e-5

#define SH_cosLobe_C0 0.886226925 // sqrt(pi)/2
#define SH_cosLobe_C1 1.02332671 // sqrt(pi/3)

struct Material {
    vec3 reflectance;
    vec3 emittance;
};

struct Object {
    vec3 position;
    float radius;
    uint material;
};

struct Radiance {
    uvec2 sh[4]; // INFO: may need to be packed into `uvec4`s, 16 byte array elements might be more performant
};

vec3 rotateWithQuat(vec4 q, vec3 v) {
    vec3 t = q.w * v + cross(q.xyz, v);
    return 2.0 * cross(q.xyz, t) + v;
}

float maximum(vec3 v) {
    return max(max(v.x, v.y), v.z);
}

uvec4 unpackBytesUint(uint bytes) {
    return uvec4(
         bytes        & 255, // [0, 8)
        (bytes >>  8) & 255, // [8, 16)
        (bytes >> 16) & 255, // [16, 24)
        (bytes >> 24) & 255  // [24, 32)
    );
}

// packs 4 bytes into a uint, assumes the inputs are in the range [0, 255]
uint packBytesUint(uvec4 bytes) {
    return uint(
        bytes.x | (bytes.y << 8) | (bytes.z << 16) | (bytes.w << 24)
    );
}

int lmLayerAtPos(vec3 v, vec3 lmOrigin) {
    const float INV_HALF_LM_SIZE = 1.0 / (float(LM_SIZE / 2) * LM_UNIT_SIZE);
    return int(log2(max(maximum(abs(v - lmOrigin)) * INV_HALF_LM_SIZE, 0.5)) + 1.001);
}

float lmUnitSizeLayer(uint lmLayer) {
    return float(1 << lmLayer) * LM_UNIT_SIZE;
}

float lmMultsLayer(uint lmLayer) {
    return (1.0 / float(LM_SIZE)) / lmUnitSizeLayer(lmLayer);
}

float lmSizeLayer(uint lmLayer) {
    return float(LM_SIZE) * lmUnitSizeLayer(lmLayer);
}

float lmHalfSizeLayer(uint lmLayer) {
    return (float(LM_SIZE) * 0.5) * lmUnitSizeLayer(lmLayer);
}

/// returns an index into a lightmap image in xyz, and the image index in w
// TODO: lmOrigin per layer
ivec4 lmIndexAtPos(vec3 pos, vec3 lmOrigin) {
    uint lmLayer = lmLayerAtPos(pos, lmOrigin);
    ivec3 index = ivec3(floor((pos - lmOrigin) / lmUnitSizeLayer(lmLayer))) + LM_SIZE / 2;

    return ivec4(index, lmLayer);
}

// TODO: lmOrigin per layer
vec3 posAtLmIndex(ivec4 lmIndex, vec3 lmOrigin) {
    return (vec3(lmIndex.xyz - LM_SIZE / 2) + 0.5) * lmUnitSizeLayer(lmIndex.w) + lmOrigin;
}

float radUnitSizeLayer(uint lmLayer) {
    return lmUnitSizeLayer(lmLayer) * float(LM_SIZE / RADIANCE_SIZE);
}

vec3 posAtRadIndex(ivec4 radIndex) { // FIXME: scale with LM_SIZE
    return (vec3(radIndex.xyz - RADIANCE_SIZE / 2) + 0.5) * radUnitSizeLayer(radIndex.w); // TODO: movable origin
}

vec3 posToLMTextureCoord(vec3 pos, uint lmLayer, vec3 lmOrigin) {
    return ((pos - lmOrigin) / lmSizeLayer(lmLayer)) + 0.5;
}

ivec4 radIndexAtPos(vec3 pos) {
    ivec4 lmIndex = lmIndexAtPos(pos, vec3(0.0));
    lmIndex.xyz /= (LM_SIZE / RADIANCE_SIZE);
    return ivec4(lmIndex); // TODO: movable origin
}

bool marchRay(sampler3D[LM_LAYERS] SDFImages, inout vec3 pos, vec3 dir, vec3 sdfOrigin, float threshold, uint samples, inout float totalDist) {
    float dist = totalDist;

    vec3 dPos = dir * threshold;

    uint layer = lmLayerAtPos(pos, sdfOrigin);

    for (uint i = 0; i < samples; i++) {
        pos += dir * dist;

        vec3 idx = pos - sdfOrigin; // TODO: sdfOrigin varying between layers
        bool outOfLayer = maximum(abs(idx)) > lmHalfSizeLayer(layer);
        bool coneTooBig = threshold * totalDist > 0.5 * lmUnitSizeLayer(layer);
        bool increaseLayer = outOfLayer || coneTooBig;
        layer = increaseLayer ? layer + 1 : layer;

        float mult = lmMultsLayer(layer);

        vec3 texIdx = (pos - sdfOrigin) * mult + 0.5; // TODO: sdfOrigin varying between layers
        dist = texture(SDFImages[layer], clamp(texIdx, 0.0, 1.0)).x;
        totalDist += dist;

        if (dist <= threshold * totalDist) { // hit or out of bounds
            return layer < LM_LAYERS;
        }
    }

    return false;
}

// Originally sourced from https://www.shadertoy.com/view/ldfSWs
vec3 calcNormalSDF(sampler3D sdf, vec3 pos, float eps) {
    const vec3 v1 = vec3( 1.0,-1.0,-1.0);
    const vec3 v2 = vec3(-1.0,-1.0, 1.0);
    const vec3 v3 = vec3(-1.0, 1.0,-1.0);
    const vec3 v4 = vec3( 1.0, 1.0, 1.0);

    return normalize( v1 * texture(sdf, pos + v1*eps).x +
                    v2 * texture(sdf, pos + v2*eps).x +
                    v3 * texture(sdf, pos + v3*eps).x +
                    v4 * texture(sdf, pos + v4*eps).x );
}

vec3 calcNormalSDF(sampler3D sdf, vec3 pos) {
    return calcNormalSDF(sdf, pos, 0.002);
}

vec3 evaluateRGBSphericalHarmonics(vec3 dir, vec3[4] coefs) {
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;

    vec3 s = vec3(0.0);

    s += coefs[0] * 0.28209479;

    s += coefs[1] * 0.48860251 * y;
    s += coefs[2] * 0.48860251 * z;
    s += coefs[3] * 0.48860251 * x;

    // s += coefs[4] * 1.09254843 * x * y;
    // s += coefs[5] * 1.09254843 * y * z;
    // s += coefs[6] * 0.31539156 * (3 * z * z - 1);
    // s += coefs[7] * 1.09254843 * x * z;
    // s += coefs[8] * 0.54627421 * (x * x - y * y);

    return s;
}

vec3 unpackSHCoef(uvec2 smallCoef) {
    return vec3(unpackHalf2x16(smallCoef.x), unpackHalf2x16(smallCoef.y).x);
}

vec3[4] unpackSHCoefs(uvec2[4] smallCoefs) {
    vec3[4] coefs;
    for (int i = 0; i < 4; i++) {
        coefs[i] = unpackSHCoef(smallCoefs[i]);
    }
    return coefs;
}

uvec2 packSHCoef(vec3 coef) {
    return uvec2(packHalf2x16(coef.rg), packHalf2x16(vec2(coef.b, 0.0)));
}

// credit to https://ericpolman.com/2016/06/28/light-propagation-volumes/
vec4 dirToCosineLobe(vec3 dir) {
    //dir = normalize(dir);
    return vec4(SH_cosLobe_C0, -SH_cosLobe_C1 * dir.y, SH_cosLobe_C1 * dir.z, -SH_cosLobe_C1 * dir.x);
}
