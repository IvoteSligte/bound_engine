#version 460

#include "includes.glsl"

layout(local_size_x = 8, local_size_y = 4, local_size_z = 1) in;

const uint MAX_OBJECTS = 16;
const uint MAX_MATERIALS = 8;
const uint RAYS_INDIRECT = 4;

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
    ivec3 lightmapOrigin;
    ivec4 deltaLightmapOrigins[LIGHTMAP_COUNT];
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

layout(binding = 5, rgba32f) uniform restrict image3D[LIGHTMAP_COUNT] lightmapImages;

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

void traceRayWithBVH(inout Ray ray, out vec3 hitPosition, inout uvec4 seeds) {
    ray.material = 0;
    hitPosition = vec3(0.0);
    float distanceToHit = 1e20;
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
            hitPosition = curr.position;
            ray.material = curr.leaf;
        }

        // move to next node
        curr_idx = curr.next;
    }

    ray.origin = (ray.direction * distanceToHit) + ray.origin;
}

float maximum(vec3 v) {
    return max(max(v.x, v.y), v.z);
}

/// returns an index into a lightmap image in xyz, and the image index in w
ivec4 lightmapIndexAtPos(vec3 v) {
    const int HALF_IMAGE_SIZE = imageSize(lightmapImages[0]).x >> 1;
    const float BASE_UNIT_SIZE = 0.5; // TODO: adapt this into the rust code, currently a base unit size of 1 is used there
    const float INV_HALF_LM_SIZE = 1.0 / (float(HALF_IMAGE_SIZE) * BASE_UNIT_SIZE);

    v -= rt.lightmapOrigin.xyz;
    uint lightmapNum = uint(log2(max(maximum(abs(v)) * INV_HALF_LM_SIZE, 0.5001)) + 1.0);
    float unitSize = (1 << lightmapNum) * BASE_UNIT_SIZE;

    ivec3 index = ivec3(floor(v / unitSize)) + HALF_IMAGE_SIZE;

    return ivec4(index, lightmapNum);
}

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

    vec3 hitPosition;
    traceRayWithBVH(ray, hitPosition, seeds);
    vec3 normal = normalize(ray.origin - hitPosition);
    ray.direction = randomDirection(normal, seeds);

    Material material = buf.mats[ray.material];

    ivec4 index = lightmapIndexAtPos(ray.origin);
    vec4 data = imageLoad(lightmapImages[index.w], index.xyz);

    if (data.w < 1e4) {
        const uint MAX_RAYS_PARALLEL = 2 << RAYS_INDIRECT;

        data.rgb += material.emittance;
        Ray rays[MAX_RAYS_PARALLEL];
        rays[0] = ray;
        vec3 colorsLeft[MAX_RAYS_PARALLEL >> 1];
        colorsLeft[0] = material.reflectance;
        
        for (uint i = 0; i < RAYS_INDIRECT; i++) {
            for (uint j = 0; j < (2 << i); j++) {
                rays[j] = rays[j >> 1];

                vec3 hitPosition;
                traceRayWithBVH(rays[j], hitPosition, seeds);
                if (i < RAYS_INDIRECT - 1) {
                    vec3 normal = normalize(rays[j].origin - hitPosition);
                    rays[j].direction = randomDirection(normal, seeds);
                }

                ivec4 indices = lightmapIndexAtPos(rays[j].origin);
                vec4 loadData = imageLoad(lightmapImages[indices.w], indices.xyz); // FIXME:

                Material material = buf.mats[rays[j].material];

                data.rgb += (material.emittance + loadData.rgb) / (1.0 + 2.0 * loadData.w) * colorsLeft[j >> 1];
                if (i < RAYS_INDIRECT - 1) {
                    colorsLeft[j] *= material.reflectance;
                }
            }
        }

        data.w += 1.0;
        imageStore(lightmapImages[index.w], index.xyz, data);
    }

    if (gl_LocalInvocationID.z == 0) {
        imageStore(colorImage, ipos, vec4(data.rgb / data.w, 0.0));
    }
}
