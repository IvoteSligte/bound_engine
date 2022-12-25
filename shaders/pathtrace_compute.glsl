#version 460

#include "compute_includes.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

const uint RAYS_PER_SAMPLE = 4;

struct Bounds { // node of a binary tree
    vec3 center;
    float radiusSquared;
    uint child;
    uint next;
    uint leaf;
};

struct Ray {
    vec3 origin; // origin
    vec3 direction; // direction
    vec3 normalOfObject; // normalOfObject at origin (or vec3(-0.0))
    vec3 color; // color left after previous absorptions
    float distanceToObject;
    uint nodeHit;
};

layout(binding = 0) uniform restrict readonly RealTimeBuffer {
    vec4 rotation;
    vec3 position;
    float time;
    vec4 previousRotation;
    vec4 inversePreviousRotation; // inverse previous rotation
    vec3 previousPosition; // previous position
} rt;

layout(binding = 1) uniform restrict readonly BoundingVolumeHierarchy {
    uint head;
    Bounds nodes[2 * MAX_OBJECTS];
} bvh;

layout(binding = 2) uniform restrict readonly MutableData {
    Material mats[MAX_MATERIALS];
} buf;

// TODO: specialization constants?
layout(binding = 3) uniform restrict readonly ConstantBuffer {
    vec2 ratio; // window height / width * fov
} cs;

layout(binding = 4) uniform sampler1D blueNoiseTexture;

layout(binding = 5, rgba16f) uniform restrict image2D dataOutputImage;

// pseudo random number generator
// https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
float rand(vec2 co){
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

vec3 rotate(vec4 q, vec3 v) {
    vec3 t = cross(q.xyz, v) + q.w * v;
    return v + 2.0 * cross(q.xyz, t);
}

// generates a cosine-distributed random direction relative to the normal
// the given normal does not need to be normalized
vec3 randomDirection(vec3 normal, uint index) {
    const float PI = 3.14159265;
    const float PI_2 = PI * 0.5;

    const float imageS = textureSize(blueNoiseTexture, 0).x;
    vec2 offset = texture(blueNoiseTexture, index / imageS).xy * vec2(PI, PI_2);

    // atan(a / b) gives a different, incorrent number than atan(a, b)
    // vec2(longitude, latitude) or vec2(phi, theta)
    vec2 t = vec2(atan(normal.y, normal.x), atan(normal.z, length(normal.xy))) + offset.xy;

    vec2 s = sin(t);
    vec2 c = cos(t);

    return vec3(c.x*c.y, s.x*c.y, s.y);
}

float distanceToObject(in Ray ray, in Bounds bnd, out bool is_inside) {
    vec3 v = bnd.center - ray.origin;
    float a = dot(ray.direction, v);
    float b = dot(v, v);
    is_inside = b < bnd.radiusSquared;
    // if (a < 0.0) { return 0.0; }
    float d = bnd.radiusSquared + a * a - b;
    if (d < 0.0) { return 0.0; }
    return a - sqrt(d);
}

void traceRayWithBVH(inout Ray ray) {
    ray.distanceToObject = 1e20;
    ray.nodeHit = 0;

    uint curr_idx = bvh.head;

    while (curr_idx != 0) {
        Bounds curr = bvh.nodes[curr_idx];

        bool is_inside;
        float d = distanceToObject(ray, curr, is_inside);
        bool is_hit = d > 0.0 && d < ray.distanceToObject;
        
        // not a leaf, move to child
        if (curr.leaf == 0 && (is_inside || is_hit)) {
            curr_idx = curr.child;
            continue;
        }
        
        if (is_hit) {
            // is a leaf, store data
            ray.distanceToObject = d;
            ray.nodeHit = curr_idx;
        }

        // move to next node
        curr_idx = curr.next;
    }
}

void updateRay(inout Ray ray, uint dirIdx) {
    ray.origin += ray.direction * ray.distanceToObject;
    ray.normalOfObject = ray.origin - bvh.nodes[ray.nodeHit].center;
    ray.direction = randomDirection(ray.normalOfObject, dirIdx);
}

void shade(inout Ray ray, inout vec3 color) {
    Material material = buf.mats[bvh.nodes[ray.nodeHit].leaf];

    color += ray.color * material.emittance;
    // rays are fired according to brdf, negating the need to calculate it here
    ray.color *= material.reflectance;
}

void main() {
    const ivec2 viewport = ivec2(imageSize(dataOutputImage).xy);
    const ivec2 ipos = ivec2(gl_GlobalInvocationID.xy);

    // maps FragCoord to xy range [-1.0, 1.0]
    vec2 normCoord = gl_GlobalInvocationID.xy * 2.0 / viewport - 1.0;
    // maps normCoord to a different range (e.g. for FOV and non-square windows)
    normCoord *= cs.ratio;

    vec4 data = vec4(0.0);
    vec3 viewDir = rotate(rt.rotation, normalize(vec3(normCoord.x, 1.0, normCoord.y)));

    Ray ray = Ray(rt.position, viewDir, vec3(0.0), vec3(1.0), 0.0, 0);

    uint dirIdx = uint(fract(sin(dot(gl_GlobalInvocationID.xy, vec2(12.9898, 78.233)) + gl_GlobalInvocationID.z + rt.time)) * 43758.5453);

    traceRayWithBVH(ray);
    updateRay(ray, dirIdx);
    shade(ray, data.rgb);

    Ray rayDirect = ray;

    for (uint r = 0; r < RAYS_PER_SAMPLE; r++) {
        dirIdx += 1;
        traceRayWithBVH(ray);
        updateRay(ray, dirIdx);
        shade(ray, data.rgb);
    }

    vec4 loaded = imageLoad(dataOutputImage, ipos);
    imageStore(dataOutputImage, ipos, vec4(loaded.rgb + data.rgb / 8.0, 0.0));
}
