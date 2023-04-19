const uint MAX_OBJECTS = 256;
const uint MAX_MATERIALS = 256;

const uint ALL_ONES = 4294967295;

const float LM_UNIT_SIZE = 0.5; // TODO: adapt this into the rust code, currently a base unit size of 1 is used there

const float LM_UNIT_SIZES[LM_COUNT] = float[](
    (1 << 0) * LM_UNIT_SIZE,
    (1 << 1) * LM_UNIT_SIZE,
    (1 << 2) * LM_UNIT_SIZE,
    (1 << 3) * LM_UNIT_SIZE,
    (1 << 4) * LM_UNIT_SIZE,
    (1 << 5) * LM_UNIT_SIZE
);

#define FLT_MAX 3.402823466e+38

#define EPSILON 1e-5

struct Material {
    vec3 reflectance;
    vec3 emittance;
};

/// node of a bvh
struct Bounds {
    vec3 position;
    float radiusSquared;
    uint child;
    uint next;
    uint material;
    float radius; // TODO: implement
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
    ivec4 lmIndex;
    uint materialHit;
    vec3 hitPoint;
    vec3 normal;
};

struct SharedStruct {
    Voxel voxel;
    vec3 lightmapOrigin;
};

struct RayResult {
    float distanceToHit;
    uint objectHit;
    uint materialHit;
};

struct Directions {
    mat4x3 directions;
    vec3 averageDirection;
    float radius;
};

struct CPUDirections {
    vec4 directions[4];
    vec3 averageDirection;
    float radius;
};

struct SharedGpuBVH {
    uint root;
    Bounds nodes[2 * MAX_OBJECTS];
};

vec3 rotateWithQuat(vec4 q, vec3 v) {
    vec3 t = q.w * v + cross(q.xyz, v);
    return 2.0 * cross(q.xyz, t) + v;
}

float maximum(vec3 v) {
    return max(max(v.x, v.y), v.z);
}

/// returns an index into a lightmap image in xyz, and the image index in w
ivec4 lightmapIndexAtPos(vec3 v, vec3 lmOrigin) {
    const int HALF_LM_SIZE = LM_SIZE / 2;
    const float INV_HALF_LM_SIZE = 1.0 / (float(HALF_LM_SIZE) * LM_UNIT_SIZE);

    v -= lmOrigin;
    uint lightmapNum = uint(log2(max(maximum(abs(v)) * INV_HALF_LM_SIZE, 0.500001)) + 1.0);

    ivec3 index = ivec3(floor(v / LM_UNIT_SIZES[lightmapNum])) + HALF_LM_SIZE;

    return ivec4(index, lightmapNum);
}

vec3 posAtLightmapIndex(ivec4 lmIndex, vec3 lmOrigin) {
    const int HALF_LM_SIZE = LM_SIZE / 2;

    vec3 v = (lmIndex.xyz - (HALF_LM_SIZE - 0.5)) * LM_UNIT_SIZES[lmIndex.w] + lmOrigin;

    return v;
}
