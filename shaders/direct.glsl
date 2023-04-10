#version 460

// START GENERAL INCLUDES

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

struct HitItem {
    vec3 position;
    uint objectHit;
    uint materialHit;
};

vec3 rotateWithQuat(vec4 q, vec3 v) {
    vec3 t = q.w * v + cross(q.xyz, v);
    return 2.0 * cross(q.xyz, t) + v;
}

float maximum(vec3 v) {
    return max(max(v.x, v.y), v.z);
}

// END GENERAL INCLUDES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(binding = 0) uniform restrict RealTimeBuffer {
    vec4 rotation;
    vec3 position;
    ivec3 lightmapOrigin;
    ivec4 deltaLightmapOrigins[LIGHTMAP_COUNT];
} rt;

layout(binding = 1) uniform restrict GpuBVH {
    uint root;
    Bounds nodes[2 * MAX_OBJECTS];
} bvh;

layout(binding = 2, rgba16) uniform restrict writeonly image2D colorImage;

layout(binding = 3/*,rgba16*/) uniform restrict image3D[RAYS_INDIRECT * LIGHTMAP_COUNT] lightmapImages; // TODO: CPU side - only grab the first LIGHTMAP_COUNT images

layout(binding = 4) buffer restrict LightmapSync {
    uint items[LIGHTMAP_COUNT][LIGHTMAP_SIZE][LIGHTMAP_SIZE][LIGHTMAP_SIZE];
} lmSync;

layout(binding = 5) buffer restrict CurrBuffer {
    HitItem items[SUBBUFFER_COUNT][SUBBUFFER_LENGTH];
} currBuffer;

layout(binding = 6) buffer restrict CurrCounters {
    uint counters[SUBBUFFER_COUNT];
} currCounters;

layout(binding = 7) uniform restrict SpecConstants {
    vec2 ratio;
} sc;

// START TRACE RAY INCLUDES
float distanceToObject(Ray ray, Bounds bnd) {
    vec3 v = bnd.position - ray.origin;
    vec2 m = v * mat2x3(ray.direction, v); // two dot products calculated using one matrix multiplication
    float d = (m.x * m.x - m.y) + bnd.radiusSquared;
    return d < 0.0 ? 0.0 : m.x - sqrt(d);
}

bool hitsBounds(Ray ray, Bounds bnd) {
    vec3 v = bnd.position - ray.origin;
    vec2 m = v * mat2x3(ray.direction, v); // two dot products calculated using one matrix multiplication
    return m.y < bnd.radiusSquared || (((-m.x) * m.x + m.y) < bnd.radiusSquared && m.x > EPSILON);
}

void traceRayWithBVH(inout Ray ray) {
    ray.objectHit = 0;
    ray.materialHit = 0;
    float distanceToHit = FLT_MAX;
    uint nodeHit = 0;

    uint currIdx = bvh.root;

    while (currIdx != 0) {
        Bounds curr = bvh.nodes[currIdx];

        if (curr.material == 0) {
            currIdx = hitsBounds(ray, curr) ? curr.child : curr.next;
            continue;
        }

        float dist = distanceToObject(ray, curr);

        if (dist > EPSILON && dist < distanceToHit) {
            // is a leaf, store data
            distanceToHit = dist;
            nodeHit = currIdx;
            ray.objectHit = currIdx;
            ray.materialHit = curr.material;
        }

        // move to next node
        currIdx = curr.next;
    }

    ray.origin = (ray.direction * distanceToHit) + ray.origin;
}
// END TRACE RAY INCLUDES

/// returns an index into a lightmap image in xyz, and the image index in w
ivec4 lightmapIndexAtPos(vec3 v) {
    const int HALF_IMAGE_SIZE = imageSize(lightmapImages[0]).x >> 1;
    const float INV_HALF_LM_SIZE = 1.0 / (float(HALF_IMAGE_SIZE) * LM_UNIT_SIZE);

    v -= rt.lightmapOrigin.xyz;
    uint lightmapNum = uint(log2(max(maximum(abs(v)) * INV_HALF_LM_SIZE, 0.5001)) + 1.5);
    float unitSize = (1 << lightmapNum) * LM_UNIT_SIZE;

    ivec3 index = ivec3(round(v / unitSize)) + HALF_IMAGE_SIZE;

    return ivec4(index, lightmapNum);
}

void main() {
    const uvec3 TOTAL_SIZE = gl_NumWorkGroups * gl_WorkGroupSize;
    const uint TOTAL_LENGTH = TOTAL_SIZE.x * TOTAL_SIZE.y; // only two dimensions are used
    const uint GLOBAL_INVOCATION_INDEX = TOTAL_SIZE.x * gl_GlobalInvocationID.y + gl_GlobalInvocationID.x; // only two dimensions are used

    // FIXME: the binding is not recognised when buffers are only written to and not read from
    HitItem useless = currBuffer.items[0][0];

    const ivec2 VIEWPORT = ivec2(imageSize(colorImage).xy);
    const ivec2 IPOS = ivec2(gl_GlobalInvocationID.xy);

    // maps FragCoord to xy range [-1.0, 1.0]
    const vec2 NORM_COORD = sc.ratio * (IPOS * 2.0 / VIEWPORT - 1.0);
    const vec3 DIRECTION = normalize(vec3(NORM_COORD.x, 1.0, NORM_COORD.y));

    vec3 viewDir = rotateWithQuat(rt.rotation, DIRECTION);

    Ray ray = Ray(0, rt.position, viewDir, 0);

    traceRayWithBVH(ray);

    ivec4 lmIndex = lightmapIndexAtPos(ray.origin);

    bool outOfRange = lmIndex.w >= LIGHTMAP_COUNT;
    if (outOfRange) {
        imageStore(colorImage, IPOS, vec4(0.0));
        return;
    }

    uint sync = atomicOr(lmSync.items[lmIndex.w][lmIndex.x][lmIndex.y][lmIndex.z], BIT_USED);
    uint level = sync & BITS_LEVEL;
    bool isUnused = (sync & BIT_USED) == 0;

    if (isUnused) {
        const uint INVOCATIONS_PER_COUNTER = TOTAL_LENGTH / SUBBUFFER_COUNT;
        const uint COUNTER_INDEX = GLOBAL_INVOCATION_INDEX / INVOCATIONS_PER_COUNTER;

        uint bufIdx = atomicAdd(currCounters.counters[COUNTER_INDEX], 1);
        if (bufIdx < SUBBUFFER_LENGTH) {
            currBuffer.items[COUNTER_INDEX][bufIdx] = HitItem(ray.origin, ray.objectHit, ray.materialHit);
        } else {
            lmSync.items[lmIndex.w][lmIndex.x][lmIndex.y][lmIndex.z] = sync;
        }
    }

    vec3 color = level > 0 ? imageLoad(lightmapImages[LIGHTMAP_COUNT * (level - 1) + lmIndex.w], lmIndex.xyz).rgb : vec3(0.0);
    imageStore(colorImage, IPOS, vec4(color, 0.0));
}
