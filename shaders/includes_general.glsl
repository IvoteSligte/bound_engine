const uint MAX_OBJECTS = 256;
const uint MAX_MATERIALS = 256;

const uint ALL_ONES = 4294967295;
const uint BIT_USED = 1 << 30; // bit 31
const uint BITS_LEVEL = 63; // bits [0, 6]

const float LM_UNIT_SIZE = 0.5; // TODO: adapt this into the rust code, currently a base unit size of 1 is used there

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
};

struct Ray {
    uint objectHit;
    vec3 origin;
    vec3 direction;
    uint materialHit;
};

vec3 rotateWithQuat(vec4 q, vec3 v) {
    vec3 t = q.w * v + cross(q.xyz, v);
    return 2.0 * cross(q.xyz, t) + v;
}

float maximum(vec3 v) {
    return max(max(v.x, v.y), v.z);
}