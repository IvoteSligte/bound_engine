const uint ALL_ONES = 4294967295;

const float LM_UNIT_SIZE = 0.5; // TODO: sync this with the rust code (defines)

const float LM_UNIT_SIZES[LM_COUNT] = float[](
    float(1 << 0) * LM_UNIT_SIZE,
    float(1 << 1) * LM_UNIT_SIZE,
    float(1 << 2) * LM_UNIT_SIZE,
    float(1 << 3) * LM_UNIT_SIZE,
    float(1 << 4) * LM_UNIT_SIZE,
    float(1 << 5) * LM_UNIT_SIZE
);

#define FLT_MAX 3.402823466e+38

#define EPSILON 1e-5

struct Material {
    vec3 reflectance;
    vec3 emittance;
};

struct Object {
    vec3 position;
    float radius;
    uint material;
};

struct Ray {
    uint objectHit;
    vec3 origin;
    vec3 direction;
    uint materialHit;
};

struct HitItem {
    vec3 position;
    uint objectHit;
    uint materialHit;
};

struct LmPoint {
    float sampleCount;
    vec3 position;
    uint material;
    vec3 normal;
    vec3 color;
};

struct SharedStruct {
    LmPoint point;
    vec3 lightmapOrigin;
};

vec3 rotateWithQuat(vec4 q, vec3 v) {
    vec3 t = q.w * v + cross(q.xyz, v);
    return 2.0 * cross(q.xyz, t) + v;
}

float maximum(vec3 v) {
    return max(max(v.x, v.y), v.z);
}

const int HALF_LM_SIZE = LM_SIZE / 2;
const float INV_HALF_LM_SIZE = 1.0 / (float(HALF_LM_SIZE) * LM_UNIT_SIZE);

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
    return int(log2(max(maximum(abs(v - lmOrigin)) * INV_HALF_LM_SIZE, 0.5)) + 1.001);
}

/// returns an index into a lightmap image in xyz, and the image index in w
// TODO: lmOrigin per layer
ivec4 lmIndexAtPos(vec3 v, vec3 lmOrigin) {
    uint lmLayer = lmLayerAtPos(v, lmOrigin);
    ivec3 index = ivec3(floor((v - lmOrigin) / LM_UNIT_SIZES[lmLayer])) + HALF_LM_SIZE;

    return ivec4(index, lmLayer);
}

// TODO: lmOrigin per layer
vec3 posAtLightmapIndex(ivec4 lmIndex, vec3 lmOrigin) {
    return (vec3(lmIndex.xyz - HALF_LM_SIZE) + 0.5) * LM_UNIT_SIZES[lmIndex.w] + lmOrigin;
}

const float MULTS[LM_COUNT] = float[](
    (1.0 / float(LM_SIZE)) / LM_UNIT_SIZES[0],
    (1.0 / float(LM_SIZE)) / LM_UNIT_SIZES[1],
    (1.0 / float(LM_SIZE)) / LM_UNIT_SIZES[2],
    (1.0 / float(LM_SIZE)) / LM_UNIT_SIZES[3],
    (1.0 / float(LM_SIZE)) / LM_UNIT_SIZES[4],
    (1.0 / float(LM_SIZE)) / LM_UNIT_SIZES[5]
);

const float LAYER_COMPS[LM_COUNT] = float[](
    float(LM_SIZE) * LM_UNIT_SIZES[0] * 0.5,
    float(LM_SIZE) * LM_UNIT_SIZES[1] * 0.5,
    float(LM_SIZE) * LM_UNIT_SIZES[2] * 0.5,
    float(LM_SIZE) * LM_UNIT_SIZES[3] * 0.5,
    float(LM_SIZE) * LM_UNIT_SIZES[4] * 0.5,
    float(LM_SIZE) * LM_UNIT_SIZES[5] * 0.5
);
