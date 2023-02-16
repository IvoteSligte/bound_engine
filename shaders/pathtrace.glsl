#version 460

#include "includes.glsl"

layout(local_size_x = 1, local_size_y = 1, local_size_z = 32) in;

const uint MAX_OBJECTS = 16;
const uint MAX_MATERIALS = 8;
const uint RAYS = 3;

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
    uint material;
    vec3 origin;
    vec3 direction;
};

layout(binding = 0) uniform restrict readonly RealTimeBuffer {
    vec4 rotation;
    vec4 previousRotation;
    vec3 position;
    vec3 previousPosition;
    ivec4 lightmapOrigins[LIGHTMAP_CASCADES];
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

layout(binding = 4, rgba16f) uniform restrict writeonly image2D colorImage;

layout(binding = 5, rgba32f) uniform restrict image3D[LIGHTMAP_CASCADES] lightmapImages;

vec3 rotateWithQuat(vec4 q, vec3 v) {
    vec3 t = q.w * v + cross(q.xyz, v);
    return 2.0 * cross(q.xyz, t) + v;
}

vec4 quatTowardsNormalFromUp(vec3 n) {
    vec4 q = vec4(-n.y, n.x, 0.0, 1.0 + n.z);
    return normalize(q);
    // The following should be equivalent to normalize(q): q * inversesqrt((2.0 * n.z) + 2.0 + very_small_value)
    // using the identity length(n) == 1 but does not produce the same results for unknown reasons
}

/// generates a cosine-distributed random direction relative to the normal
vec3 randomDirection(vec3 normal, inout uvec4 seeds) {
    const float OFFSET_1_CORRECTION = 3.14159265 * 2.0 * HYBRID_TAUS_NORMALIZER;

    float r1 = OFFSET_1_CORRECTION * HybridTausUnnormalized(seeds);
    float r2 = HybridTaus(seeds);

    vec3 rand;
    rand.xy = vec2(cos(r1), sin(r1)) * cos(r2);
    rand.z = r2;

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

void traceRayWithBVH(inout Ray ray, inout uvec4 seeds) {
    ray.material = 0;
    float distanceToHit = 1e20;
    vec3 nodeHitPosition = vec3(0.0);
    uint nodeHit = 0;

    uint curr_idx = bvh.root;

    while (curr_idx != 0) {
        Bounds curr = bvh.nodes[curr_idx];

        bool is_inside;
        float d = distanceToObject(ray, curr, is_inside);
        bool is_hit = d > 0.0 && d < distanceToHit;

        // not a leaf, move to child
        if (curr.leaf == 0 && (is_inside || is_hit)) {
            curr_idx = curr.child;
            continue;
        }

        if (is_hit) {
            // is a leaf, store data
            distanceToHit = d;
            nodeHit = curr_idx;
            nodeHitPosition = curr.position;
            ray.material = curr.leaf;
        }

        // move to next node
        curr_idx = curr.next;
    }

    ray.origin = (ray.direction * distanceToHit) + ray.origin;
    vec3 normal = normalize(ray.origin - nodeHitPosition); // TODO: do not do this after the last ray, it's unnecessary
    ray.direction = randomDirection(normal, seeds); // TODO: do not do this after the last ray, it's unnecessary
}

float maximum(vec3 v) {
    return max(max(v.x, v.y), v.z);
}

/// returns an index into a lightmap image in xyz, and the image index in w
ivec4 lightmapIndexAtPos(vec3 v) {
    const int HALF_IMAGE_SIZE = imageSize(lightmapImages[0]).x >> 1;
    const float INV_HALF_IMAGE_SIZE = 1.0 / float(HALF_IMAGE_SIZE);

    uint lightmapNum = uint(log2(max(maximum(abs(v)) * INV_HALF_IMAGE_SIZE, 0.5001)) + 1.0);
    int unitSize = 1 << lightmapNum;

    ivec3 index = ivec3(floor(v)) / unitSize - rt.lightmapOrigins[lightmapNum].xyz + HALF_IMAGE_SIZE;

    return ivec4(index, lightmapNum);
}

shared vec3 SharedColors[gl_WorkGroupSize.z];

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

    // maps FragCoord to xy range [-1.0, 1.0]
    vec2 normCoord = ipos * (2.0 / viewport) - 1.0;
    // maps normCoord to a different range (e.g. for FOV and non-square windows)
    normCoord *= cs.ratio;

    vec3 viewDir = rotateWithQuat(rt.rotation, normalize(vec3(normCoord.x, 1.0, normCoord.y)));

    Ray ray = Ray(0, rt.position, viewDir);

    ivec4 indices[RAYS];
    Material materials[RAYS];
    vec4 loads[RAYS];

    vec3 outColor = vec3(0.0);
    vec3 colorLeft = vec3(1.0);
    for (uint r = 0; r < RAYS; r++) {
        traceRayWithBVH(ray, seeds);

        indices[r] = lightmapIndexAtPos(ray.origin);
        materials[r] = buf.mats[ray.material];
        loads[r] = imageLoad(lightmapImages[indices[r].w], indices[r].xyz);

        outColor += loads[r].rgb * colorLeft;
        colorLeft *= materials[r].reflectance;
    }

    for (uint i = 0; i < RAYS; i++) {
        vec3 outLMColor = materials[i].emittance;

        outLMColor += i != 0 ? loads[i-1].rgb * materials[i-1].reflectance : vec3(0.0);
        outLMColor += i != RAYS-1 ? loads[i+1].rgb * materials[i+1].reflectance : vec3(0.0);

        imageStore(lightmapImages[indices[i].w], indices[i].xyz, vec4(mix(loads[i].rgb, outLMColor * 0.3333, exp2(-16)), 0.0));
    }

    SharedColors[gl_LocalInvocationID.z] = outColor;
    memoryBarrierShared();
    barrier();

    if (gl_LocalInvocationID.z == 0) {
        for (uint i = 1; i < gl_WorkGroupSize.z; i++) {
            outColor += SharedColors[i];
        }

        imageStore(colorImage, ipos, vec4(outColor, 0.0));
    }
}
