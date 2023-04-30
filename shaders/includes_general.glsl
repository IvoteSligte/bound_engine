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

struct Voxel {
    ivec4 lmIndex; // TODO: compress
    uint material;
    vec3 position;
    vec3 normal;
};

struct SharedStruct {
    Voxel voxel;
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
