#version 460

#include "compute_includes.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

const uint MAX_OBJECTS = 16;
const uint MAX_MATERIALS = 8;
const uint RAYS_PER_SAMPLE = 6;

struct Material {
    vec3 reflectance;
    vec3 emittance;
};

struct Bounds { // node of a bvh
    vec3 position;
    float radiusSquared;
    uint child;
    uint next;
    uint leaf;
};

struct Ray {
    vec3 origin;
    vec3 direction;
    vec3 normalOfObject;
    vec3 colorLeft;
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

layout(binding = 1) uniform restrict readonly GpuBVH {
    uint root;
    Bounds nodes[2 * MAX_OBJECTS];
} bvh;

layout(binding = 2) uniform restrict readonly MutableData {
    Material mats[MAX_MATERIALS];
} buf;

// TODO: specialization constants?
layout(binding = 3) uniform restrict readonly ConstantBuffer {
    vec2 ratio; // window height / width * fov
} cs;

layout(binding = 4, rgba16f) uniform restrict image2D dataOutputImage;

vec3 rotate(vec4 q, vec3 v) {
    vec3 t = q.w * v + cross(q.xyz, v);
    return 2.0 * cross(q.xyz, t) + v;
}

// FIXME: weird angles (visible with red light)
// FIXME: inverse cosine distribution ???
// generates a cosine-distributed random direction relative to the normal
// the given normal does not need to be normalized
vec3 randomDirection(vec3 normal, inout uvec4 seeds) {
    const float OFFSET_2_CORRECTION = 3.14159265 * 2.0 * HYBRID_TAUS_NORMALIZER;
    // angle offsets
    float o1 = HybridTaus(seeds);
    float o2 = OFFSET_2_CORRECTION * HybridTausUnnormalized(seeds);

    vec3 v = vec3(o1, normal.xz);
    vec3 sq = sqrt((-v * v) + 1.0);

    // longitude
    vec2 phi = vec2(normal.x, sq.y) * mat2x2(vec2(o1, sq.x), vec2(sq.x, -o1));
    // latitude
    vec2 theta = vec2(sq.z, normal.z) * mat2x2(vec2(sin(o2), cos(o2)), vec2(cos(o2), -sin(o2)));

    return vec3(phi.yx*theta.y, theta.x);
}

float distanceToObject(Ray ray, Bounds bnd, out bool is_inside) {
    vec3 v = bnd.position - ray.origin;
    vec2 m = v * mat2x3(ray.direction, v); // two dot products calculated using one matrix multiplication
    is_inside = m.y < bnd.radiusSquared;
    float d = dot(vec3((m.x * m.x), -m.y, bnd.radiusSquared), vec3(1.0));
    return d < 0.0 ? 0.0 : m.x - sqrt(d);
}

void traceRayWithBVH(inout Ray ray) {
    ray.distanceToObject = 1e20;
    ray.nodeHit = 0;

    uint curr_idx = bvh.root;

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

void updateRay(inout Ray ray, inout uvec4 seeds) {
    ray.origin = (ray.direction * ray.distanceToObject) + ray.origin;
    ray.normalOfObject = normalize(ray.origin - bvh.nodes[ray.nodeHit].position);
    ray.direction = randomDirection(ray.normalOfObject, seeds);
}

void shade(inout Ray ray, inout vec3 color) {
    Material material = buf.mats[bvh.nodes[ray.nodeHit].leaf];

    color = (ray.colorLeft * material.emittance) + color;
    // rays are fired according to brdf, negating the need to calculate it here
    ray.colorLeft *= material.reflectance;
}

void main() {
    const ivec2 viewport = ivec2(imageSize(dataOutputImage).xy);
    const ivec2 ipos = ivec2(gl_GlobalInvocationID.xy);

    // maps FragCoord to xy range [-1.0, 1.0]
    vec2 normCoord = ipos * (2.0 / viewport) - 1.0;
    // maps normCoord to a different range (e.g. for FOV and non-square windows)
    normCoord *= cs.ratio;

    vec4 data = vec4(0.0);
    vec3 viewDir = rotate(rt.rotation, normalize(vec3(normCoord.x, 1.0, normCoord.y)));

    // FIXME: weird edges in rendering caused by a change in angle (lighting in said edges changes based on position)
    Ray ray = Ray(rt.position, viewDir, vec3(0.0), vec3(1.0), 0.0, 0);

    const uvec4 baked_seeds = uvec4(
        gl_GlobalInvocationID.x * gl_GlobalInvocationID.y * 43217 + gl_GlobalInvocationID.z * 10 + 128 * 1,
        gl_GlobalInvocationID.x * gl_GlobalInvocationID.y * 73574 + gl_GlobalInvocationID.z * 5 + 128 * 57,
        gl_GlobalInvocationID.x * gl_GlobalInvocationID.y * 57895 + gl_GlobalInvocationID.z * 71 + 128 * 6,
        gl_GlobalInvocationID.x * gl_GlobalInvocationID.y * 11581 + gl_GlobalInvocationID.z * 12 + 128 * 11
    );

    uvec4 seeds = baked_seeds;

    traceRayWithBVH(ray);
    updateRay(ray, seeds);
    shade(ray, data.rgb);

    for (uint r = 0; r < RAYS_PER_SAMPLE; r++) {
        traceRayWithBVH(ray);
        updateRay(ray, seeds);
        shade(ray, data.rgb);
    }

    const float ALPHA = 1.0 / 32.0;
    vec4 loaded = imageLoad(dataOutputImage, ipos);
    imageStore(dataOutputImage, ipos, vec4(data.rgb * ALPHA + loaded.rgb, 0.0));
}
