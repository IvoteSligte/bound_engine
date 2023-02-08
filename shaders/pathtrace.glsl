#version 460

#include "compute_includes.glsl"

layout(local_size_x = 1, local_size_y = 1, local_size_z = 32) in;

const uint MAX_OBJECTS = 16;
const uint MAX_MATERIALS = 8;
const uint INDIRECT_RAYS_PER_SAMPLE = 3;

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

struct RayColorInfo {
    vec3 colorLeft;
    uint material;
};

struct Ray {
    RayColorInfo colorInfo;
    vec3 origin;
    vec3 direction;
    vec3 normalOfObject;
    float distanceToObject;
    uint nodeHit;
};

layout(binding = 0) uniform restrict readonly RealTimeBuffer {
    vec4 rotation;
    vec4 previousRotation;
    vec3 position;
    vec3 previousPosition;
    ivec3 lightmapOrigin;
    ivec3 deltaLightmapOrigin;
    uint frame;
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

layout(binding = 4, rgba16f) uniform restrict image2D colorImage;

layout(binding = 5, rgba16f) uniform restrict image3D lightmapImage;

vec3 rotate(vec4 q, vec3 v) {
    vec3 t = q.w * v + cross(q.xyz, v);
    return 2.0 * cross(q.xyz, t) + v;
}

/// generates a cosine-distributed random direction relative to the normal
// vec3 randomDirection(vec3 normal, inout uvec4 seeds) {
//     const float OFFSET_2_CORRECTION = 3.14159265 * 2.0 * HYBRID_TAUS_NORMALIZER;

//     float r1 = HybridTaus(seeds); // longitude offset
//     float r2 = OFFSET_2_CORRECTION * HybridTausUnnormalized(seeds); // latitude offset

//     vec3 v = vec3(r1, normal.xz);
//     vec3 sq = sqrt((-v * v) + 1.0);

//     vec2 phi = vec2(normal.x, sq.y) * mat2x2(vec2(sq.x, r1), vec2(r1, sq.x));
//     vec2 theta = vec2(sq.z, normal.z) * mat2x2(vec2(sin(r2), cos(r2)), vec2(cos(r2), -sin(r2))); // FIXME: weird angles (visible with red light)

//     return vec3(phi.yx*theta.y, theta.x);
// }

vec3 randomDirection(vec3 normal, inout uvec4 seeds) {
    const float OFFSET_2_CORRECTION = 3.14159265 * 2.0 * HYBRID_TAUS_NORMALIZER;

    float r1 = HybridTaus(seeds); // longitude offset
    float r2 = OFFSET_2_CORRECTION * HybridTausUnnormalized(seeds); // latitude offset

    vec2 a = vec2(r2, r1) + asin(normal.xz);

    vec2 s = sin(a);
    vec2 c = cos(a);

    return vec3(c.x*c.y, s.x*c.y, s.y);
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
    ray.colorInfo.material = 0;

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
            ray.colorInfo.material = curr.leaf;
        }

        // move to next node
        curr_idx = curr.next;
    }
}

void updateRay(inout Ray ray, inout uvec4 seeds) {
    ray.origin = (ray.direction * ray.distanceToObject) + ray.origin;
    ray.normalOfObject = normalize(ray.origin - bvh.nodes[ray.nodeHit].position); // TODO: move bvh.nodes[ray.nodeHit].position to ray data
    ray.direction = randomDirection(ray.normalOfObject, seeds);
}

vec3 shadeDirect(inout RayColorInfo rayColor) {
    Material material = buf.mats[rayColor.material];
    vec3 colorChange = rayColor.colorLeft * material.emittance / gl_WorkGroupSize.z;
    rayColor.colorLeft *= material.reflectance;
    return colorChange;
}

vec3 shadeIndirect(inout RayColorInfo rayColor, inout vec3 lightmapColor) {
    Material material = buf.mats[rayColor.material];
    lightmapColor = mix(lightmapColor, material.emittance / gl_WorkGroupSize.z, 0.01);
    vec3 colorChange = rayColor.colorLeft * lightmapColor;
    rayColor.colorLeft *= material.reflectance;
    return colorChange;
}

ivec3 lightmapIndices(vec3 v) {
    const ivec3 IMAGE_OFFSET = imageSize(lightmapImage) >> 1;
    return ivec3(v) - rt.lightmapOrigin + IMAGE_OFFSET;
}

shared vec3 SharedColors[gl_WorkGroupSize.z];

shared Ray SharedDirectRay;

void main() {
    const ivec2 viewport = ivec2(imageSize(colorImage).xy);
    const ivec2 ipos = ivec2(gl_GlobalInvocationID.xy);

    const uvec4 baked_seeds = uvec4(
        gl_GlobalInvocationID.x * gl_GlobalInvocationID.y * 43217 + gl_GlobalInvocationID.z * 10 + 128 * 1,
        gl_GlobalInvocationID.x * gl_GlobalInvocationID.y * 73574 + gl_GlobalInvocationID.z * 5 + 128 * 57,
        gl_GlobalInvocationID.x * gl_GlobalInvocationID.y * 57895 + gl_GlobalInvocationID.z * 71 + 128 * 6,
        gl_GlobalInvocationID.x * gl_GlobalInvocationID.y * 11581 + gl_GlobalInvocationID.z * 12 + 128 * 11
    );

    uvec4 seeds = baked_seeds + rt.frame * 65423;

    vec3 color = vec3(0.0);

    // maps FragCoord to xy range [-1.0, 1.0]
    vec2 normCoord = ipos * (2.0 / viewport) - 1.0;
    // maps normCoord to a different range (e.g. for FOV and non-square windows)
    normCoord *= cs.ratio;

    vec3 viewDir = rotate(rt.rotation, normalize(vec3(normCoord.x, 1.0, normCoord.y)));

    // FIXME: weird edges in rendering caused by a change in angle (lighting in said edges changes based on position)
    Ray ray = Ray(RayColorInfo(vec3(1.0), 0), rt.position, viewDir, vec3(0.0), 0.0, 0);

    traceRayWithBVH(ray);
    updateRay(ray, seeds);

    color += shadeDirect(ray.colorInfo);

    for (uint r = 0; r < INDIRECT_RAYS_PER_SAMPLE; r++) {
        // position of previous hit
        ivec3 indices = lightmapIndices(ray.origin);

        traceRayWithBVH(ray);
        updateRay(ray, seeds);

        vec4 data = imageLoad(lightmapImage, indices); // TODO: multiply by the material reflectance
        vec3 colorChange = shadeIndirect(ray.colorInfo, data.rgb);
        imageStore(lightmapImage, indices, vec4(data.rgb, data.w + 1.0));
        color += colorChange;
    }

    color += imageLoad(lightmapImage, lightmapIndices(ray.origin)).rgb;

    SharedColors[gl_LocalInvocationID.z] = color;
    memoryBarrierShared();
    barrier();

    if (gl_LocalInvocationID.z == 0) {
        for (uint i = 1; i < gl_WorkGroupSize.z; i++) {
            color += SharedColors[i];
        }

        vec4 loaded = imageLoad(colorImage, ipos);
        imageStore(colorImage, ipos, vec4(mix(loaded.rgb, color, 0.4), 0.0));
    }
}
