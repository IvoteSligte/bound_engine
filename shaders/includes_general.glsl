#include "includes_random.glsl"

const uint MAX_OBJECTS = 16;
const uint MAX_MATERIALS = 8;
const uint RAYS_INDIRECT = 4;

const uint LIGHTMAP_COUNT = 6;

const uint MID_SAMPLES = 1024;
const uint LAST_SAMPLES = 1024;

const uint MID_SUBBUFFER_COUNT = 256;
const uint LAST_SUBBUFFER_COUNT = 256;

const uint MID_SUBBUFFER_LENGTH = MID_COUNT / MID_SUBBUFFER_COUNT; // even division
const uint LAST_SUBBUFFER_LENGTH = LAST_COUNT / LAST_SUBBUFFER_COUNT; // even division

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
    uint leaf;
};

struct Ray {
    uint objectHit;
    vec3 origin;
    vec3 direction;
};

struct HitItem {
    vec3 position;
    uint objectHit;
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
vec3 randomDirection(vec3 normal, inout uvec4 seeds) {
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