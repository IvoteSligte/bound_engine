#include "includes_random.glsl"

const uint MAX_OBJECTS = 16;
const uint MAX_MATERIALS = 8;

const uint SUBBUFFER_COUNT = 256;
const uint SUBBUFFER_LENGTH = ITEM_COUNT / SUBBUFFER_COUNT; // even division

const uint ALL_ONES = 4294967295;
const uint BIT_USED = 1 << 30; // bit 31
const uint BITS_SAMPLES = 262080; // bits [7, 18]
const uint BITS_LEVEL = 63; // bits [0, 6]

const float LM_UNIT_SIZE = 0.5; // TODO: adapt this into the rust code, currently a base unit size of 1 is used there

#define FLT_MAX 3.402823466e+38

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

vec3 rotateWithQuat(vec4 q, vec3 v) {
    vec3 t = q.w * v + cross(q.xyz, v);
    return 2.0 * cross(q.xyz, t) + v;
}

vec4 quatTowardsNormalFromUp(vec3 n) {
    vec4 q = vec4(-n.y, n.x, 0.0, n.z + 1.0);
    return normalize(q);
    // The following should be equivalent to normalize(q): q * inversesqrt((2.0 * n.z) + 2.0 + very_small_value)
    // using the identity length(n) == 1 but does not produce the same results for unknown reasons
}

/// generates a cosine-distributed random direction relative to the normal
vec3 randomDirectionCos(vec3 normal, inout uvec4 seeds) {
    const float OFFSET_1_CORRECTION = 3.14159265 * 2.0 * HYBRID_TAUS_NORMALIZER;
    const float OFFSET_2_CORRECTION = 3.14159265 * 0.5 * HYBRID_TAUS_NORMALIZER;

    float r1 = OFFSET_1_CORRECTION * HybridTausUnnormalized(seeds);
    float r2 = OFFSET_2_CORRECTION * HybridTausUnnormalized(seeds);

    vec3 rand;
    rand.xy = vec2(cos(r1), sin(r1)) * cos(r2);
    rand.z = sin(r2);

    vec4 quat = quatTowardsNormalFromUp(normal);
    return rotateWithQuat(quat, rand);
}

float maximum(vec3 v) {
    return max(max(v.x, v.y), v.z);
}