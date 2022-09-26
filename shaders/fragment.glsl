#version 460

#include "random.glsl"

layout(location = 0) out vec4 fragColor;

layout(push_constant) uniform PushConstantData {
    vec4 rot;
    vec3 pos;
    float time;
} pc;

const uint MAX_OBJECTS = 8;
const uint MAX_SAMPLES = 256;

struct Material {
    vec3 reflectance;
    vec3 emittance;
};

struct Object {
    vec3 pos;
    float size;
};

struct Ray {
    vec3 origin; // origin
    vec3 direction; // direction
    vec3 normalOfObject; // normalOfObject at origin (or vec3(-0.0))
    vec3 color; // color left after previous absorptions
    float distanceToObject;
    uint objectHit;
};

layout(binding = 0) uniform readonly MutableData {
    uint matCount;
    uint objCount;
    Material mats[MAX_OBJECTS];
    Object objs[MAX_OBJECTS];
} buf;

layout(binding = 1) uniform readonly ConstantBuffer {
    vec2 view; // window size
    vec2 ratio; // window height / width
} cs;

vec3 rotate(vec4 q, vec3 v) {
    vec3 t = cross(q.xyz, v) + q.w * v;
    return v + 2.0 * cross(q.xyz, t);
}

// quasi random number generator
vec3 randomUnitVectorOnHemisphere(vec3 n, vec3 seed) {
    const float m = 1432324.329543;

    vec3 v = mod(seed * m, 2.0) - 1.0;

    return -normalize(faceforward(v, v, n));
}

// calculates the distance between a ray (Ray) and a sphere (Object)
float distanceToObject(Ray ray, Object obj) {
    vec3 v = obj.pos - ray.origin;
    float a = dot(ray.direction, v); // distance to plane through obj.pos perpendicular to ray.direction
    return a - sqrt(obj.size * obj.size - dot(v, v) + a * a);
}

void initRay(inout Ray ray) {
    ray.distanceToObject = 1e20;
    ray.objectHit = MAX_OBJECTS;
}

void traceRay(inout Ray ray) {
    for (uint i = 0; i < buf.objCount; i++) {
        float d = distanceToObject(ray, buf.objs[i]);

        if (d > 0.0 && d < ray.distanceToObject) {
            ray.distanceToObject = d;
            ray.objectHit = i;
        }
    }
}

void updateRay(inout Ray ray) {
    ray.origin += ray.direction * ray.distanceToObject;
    ray.normalOfObject = normalize(ray.origin - buf.objs[ray.objectHit].pos);
    ray.direction = randomUnitVectorOnHemisphere(ray.normalOfObject, ray.origin);
}

void shade(inout Ray ray) {
    Material material = buf.mats[ray.objectHit];

    float cos_theta = dot(ray.direction, ray.normalOfObject);
    vec3 BRDF = material.reflectance;

    fragColor.xyz += ray.color * material.emittance;
    ray.color *= cos_theta * BRDF * 2.0;
}

void main() {
    // maps FragCoord to xy range [-1.0, 1.0]
    vec2 normCoord = gl_FragCoord.xy * 2.0 / cs.view - 1.0;
    // maps normCoord to a different range (e.g. for FOV and non-square windows)
    normCoord *= cs.ratio;

    fragColor.xyz = vec3(0.0);
    vec3 viewDir = normalize(rotate(pc.rot, vec3(normCoord.x, 1.0, normCoord.y)));

    Ray ray = Ray(pc.pos, viewDir, vec3(0.0), vec3(1.0), 0.0, 0);

    initRay(ray);
    traceRay(ray);
    if (ray.objectHit == MAX_OBJECTS) {
        return;
    }
    updateRay(ray);
    shade(ray);

    for (uint r = 0; r < MAX_SAMPLES; r++) {
        initRay(ray);
        traceRay(ray);
        if (ray.objectHit == MAX_OBJECTS) {
            ray.direction = randomUnitVectorOnHemisphere(ray.normalOfObject, ray.origin + r);
            continue;
        }
        updateRay(ray);
        shade(ray);
    }
}