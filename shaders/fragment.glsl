#version 460

#include "random.glsl"

#define PI 3.1415926538

layout(location = 0) out vec4 fragColor;

layout(push_constant) uniform PushConstantData {
    vec4 rot;
    vec3 pos;
    float time;
} pc;

const uint MAX_MATERIALS = 8;
const uint MAX_OBJECTS = 8;

const uint MAX_DEPTH = 256;

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
    vec3 dir; // direction
    vec3 normal; // normal at origin (or vec3(-0.0))
    uint depth;
    vec3 color; // color left after previous absorptions
};

struct Index {
    uint i;
};

layout(binding = 0) uniform readonly MutableData {
    uint matCount;
    uint objCount;
    Material mats[MAX_MATERIALS];
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

vec3 randomUnitVectorOnHemisphere(vec3 n, vec3 seed) {
    vec3 v = random(seed);
    return normalize(v) * sign(dot(n, v));
}

// calculates the distance between a ray (Ray) and a sphere (Object)
float distanceToObject(Ray ray, Object obj) {
    vec3 v = obj.pos - ray.origin;
    float a = dot(ray.dir, v); // distance to plane through obj.pos perpendicular to ray.dir
    return a - sqrt(obj.size * obj.size - dot(v, v) + a * a);
}

// cone tracing
void distanceCombined(Ray ray, Object obj, out vec3 point, out float dist, out float perpDist) {
    vec3 v = obj.pos - ray.origin;
    float a = dot(ray.dir, v); // distance to plane through obj.pos perpendicular to ray.dir
    float b = dot(v, v) - a * a;
    float c = sqrt(b);
    perpDist = c - obj.size; // perpendicular distance
    dist = a - sqrt(obj.size * obj.size - min(b, obj.size * obj.size)); // distance to point on sphere

    vec3 d = obj.pos + normalize(ray.dir * a - v) * min(c, obj.size);
    point = ray.origin + normalize(d - ray.origin) * dist;
}

void main() {
    // maps FragCoord to xy range [-1.0, 1.0]
    vec2 normCoord = gl_FragCoord.xy * 2.0 / cs.view - 1.0;
    // maps normCoord to a different range (e.g. for FOV and non-square windows)
    normCoord *= cs.ratio;

    fragColor.xyz = vec3(0.0);
    vec3 viewDir = normalize(rotate(pc.rot, vec3(normCoord.x, 1.0, normCoord.y)));

    vec3 seed = vec3(normCoord.xy / normCoord.yx, normCoord.x * normCoord.y);

    Ray ray = Ray(pc.pos, viewDir, vec3(-0.0), 0, vec3(1.0));

    for (uint r = 0; r < MAX_DEPTH; r++) {
        seed += vec3(1.3, 2.4, 3.1);

        uint index = MAX_OBJECTS;
        float dist = 1e20;

        for (uint i = 0; i < buf.objCount; i++) {
            float d = distanceToObject(ray, buf.objs[i]);

            if (d > 0.0 && d < dist) {
                dist = d;
                index = i;
            }
        }
        
        if (index == MAX_OBJECTS) {
            if (ray.normal.x != -0.0) {
                ray.dir = randomUnitVectorOnHemisphere(ray.normal, seed);
                ray.depth += 1;
                continue;
            } else {
                return;
            }
        }

        Material material = buf.mats[index];

        Ray newRay;
        newRay.origin = ray.origin + ray.dir * dist;
        newRay.normal = normalize(newRay.origin - buf.objs[index].pos);
        newRay.dir = randomUnitVectorOnHemisphere(newRay.normal, seed);
        newRay.depth = ray.depth + 1;

        float cos_theta = dot(newRay.dir, newRay.normal);
        vec3 BRDF = material.reflectance;

        fragColor.xyz += ray.color * material.emittance / (r + 1);
        newRay.color = ray.color * cos_theta * BRDF * 2.0;

        ray = newRay;
    }
}