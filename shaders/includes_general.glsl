#include "includes_random.glsl"

const uint MAX_OBJECTS = 16;
const uint MAX_MATERIALS = 8;
const uint RAYS_INDIRECT = 4;

const uint LIGHTMAP_COUNT = 6;

const uint INDIRECT_FINAL_SAMPLES = 256;
const uint INDIRECT_TRUE_SAMPLES = 256;

const uint INDIRECT_FINAL_COUNTER_COUNT = 256;
const uint INDIRECT_TRUE_COUNTER_COUNT = 256;

const uint INDIRECT_FINAL_ITEMS_PER_COUNTER = INDIRECT_FINAL_COUNT / INDIRECT_FINAL_COUNTER_COUNT; // even division
const uint INDIRECT_TRUE_ITEMS_PER_COUNTER = INDIRECT_TRUE_COUNT / INDIRECT_TRUE_COUNTER_COUNT; // even division

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

float distanceToObject(Ray ray, Bounds bnd, out bool is_inside) {
    vec3 v = bnd.position - ray.origin;
    vec2 m = v * mat2x3(ray.direction, v); // two dot products calculated using one matrix multiplication
    is_inside = m.y < bnd.radiusSquared;
    float d = dot(vec3((m.x * m.x), -m.y, bnd.radiusSquared), vec3(1.0));
    return d < 0.0 ? 0.0 : m.x - sqrt(d);
}

float maximum(vec3 v) {
    return max(max(v.x, v.y), v.z);
}