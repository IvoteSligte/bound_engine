#version 460

// START GENERAL INCLUDES // TODO: figure out a way to inline a file

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

layout(local_size_x = SAMPLES, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) uniform restrict readonly RealTimeBuffer {
    vec4 rotation;
    vec4 previousRotation;
    vec3 position;
    vec3 previousPosition;
    ivec3 lightmapOrigin;
    ivec4 deltaLightmapOrigins[LIGHTMAP_COUNT];
} rt;

layout(binding = 1) uniform restrict readonly GpuBVH {
    uint root;
    Bounds nodes[2 * MAX_OBJECTS];
} bvh;

layout(binding = 2) uniform restrict readonly MutableData {
    Material mats[MAX_MATERIALS];
} buf;

layout(binding = 3, rgba16) uniform restrict image3D[RAYS_INDIRECT * LIGHTMAP_COUNT] lightmapImages;

layout(binding = 4) buffer restrict LightmapSync {
    uint items[LIGHTMAP_COUNT][LIGHTMAP_SIZE][LIGHTMAP_SIZE][LIGHTMAP_SIZE];
} lmSync;

layout(binding = 5) buffer restrict readonly CurrBuffer {
    HitItem items[SUBBUFFER_COUNT][SUBBUFFER_LENGTH];
} currBuffer;

layout(binding = 6) buffer restrict CurrCounters {
    uint counters[SUBBUFFER_COUNT];
} currCounters;

layout(binding = 7) buffer restrict NextBuffer {
    HitItem items[SUBBUFFER_COUNT][SUBBUFFER_LENGTH];
} nextBuffer;

layout(binding = 8) buffer restrict NextCounters {
    uint counters[SUBBUFFER_COUNT];
} nextCounters;

layout(binding = 9) uniform restrict readonly BlueNoise {
    vec4 items[SAMPLES];
} bn;

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

shared vec3 SharedColors[SAMPLES];
shared bool SharedIsValids[SAMPLES];

void main() {
    // FIXME: binding is invalid when the buffer is not read from
    HitItem useless = nextBuffer.items[0][0];

    const uint COUNTER_INDEX = gl_WorkGroupID.x / SUBBUFFER_LENGTH;
    const uint BUFFER_INDEX = gl_WorkGroupID.x % SUBBUFFER_LENGTH;

    // if buffer slot is empty
    if (BUFFER_INDEX >= currCounters.counters[COUNTER_INDEX]) {
        return;
    }

    HitItem hitItem = currBuffer.items[COUNTER_INDEX][BUFFER_INDEX];
    ivec4 lmIndex = lightmapIndexAtPos(hitItem.position);

    vec3 normal = normalize(hitItem.position - bvh.nodes[hitItem.objectHit].position);

    uint sync = lmSync.items[lmIndex.w][lmIndex.x][lmIndex.y][lmIndex.z];
    uint level = sync & BITS_LEVEL;

    vec3 randDir = normalize(normal + bn.items[gl_LocalInvocationID.x].xyz);
    Ray ray = Ray(0, hitItem.position, randDir, 0);

    traceRayWithBVH(ray);

    ivec4 lmIndexSample = lightmapIndexAtPos(ray.origin);

    bool outOfRange = lmIndexSample.w >= LIGHTMAP_COUNT;
    if (outOfRange) {
        SharedColors[gl_LocalInvocationID.x] = vec3(0.0);
        SharedIsValids[gl_LocalInvocationID.x] = true;
    } else {
        uint syncSample = atomicOr(lmSync.items[lmIndexSample.w][lmIndexSample.x][lmIndexSample.y][lmIndexSample.z], BIT_USED);

        uint levelSample = syncSample & BITS_LEVEL;
        bool isUnused = (syncSample & BIT_USED) == 0;

        if (levelSample < level) {
            if (isUnused) {
                const uint COUNTER_INDEX = gl_WorkGroupID.x / SUBBUFFER_LENGTH;

                uint bufIdx = atomicAdd(nextCounters.counters[COUNTER_INDEX], 1);
                if (bufIdx < SUBBUFFER_LENGTH) {
                    nextBuffer.items[COUNTER_INDEX][bufIdx] = HitItem(ray.origin, ray.objectHit, ray.materialHit);
                } else {
                    lmSync.items[lmIndexSample.w][lmIndexSample.x][lmIndexSample.y][lmIndexSample.z] = syncSample;
                }
            }

            SharedColors[gl_LocalInvocationID.x] = vec3(0.0);
            SharedIsValids[gl_LocalInvocationID.x] = false;
        } else {
            if (isUnused) {
                lmSync.items[lmIndexSample.w][lmIndexSample.x][lmIndexSample.y][lmIndexSample.z] = syncSample;
            }

            // TODO: improve level 0 stuff, its behaviour is different so moving it to a different shader might be useful
            // otherwise, create an image and fill it with emission at points
            if (level == 0) {
                Material material = buf.mats[ray.materialHit];
                SharedColors[gl_LocalInvocationID.x] = material.emittance;
            } else {
                SharedColors[gl_LocalInvocationID.x] = imageLoad(lightmapImages[LIGHTMAP_COUNT * (level - 1) + lmIndexSample.w], lmIndexSample.xyz).rgb;
            }

            SharedIsValids[gl_LocalInvocationID.x] = true;
        }
    }

    barrier();

    if (gl_LocalInvocationID.x < 64) {
        vec3 color = vec3(0.0);
        bool isValid = true;
        for (uint i = 0; i < SAMPLES / 64; i++) {
            color += SharedColors[i * 64 + gl_LocalInvocationID.x];
            isValid = isValid && SharedIsValids[i * 64 + gl_LocalInvocationID.x];
        }
        SharedColors[gl_LocalInvocationID.x] = color;
        SharedIsValids[gl_LocalInvocationID.x] = isValid;
    }

    barrier();

    if (gl_LocalInvocationID.x == 0) {
        vec3 color = vec3(0.0);
        bool isValid = true;
        for (uint i = 0; i < 64; i++) {
            color += SharedColors[i];
            isValid = isValid && SharedIsValids[i];
        }

        if (isValid) {
            Material material = buf.mats[hitItem.materialHit];
            color = color * (material.reflectance * (1.0 / SAMPLES)) + material.emittance;

            imageStore(lightmapImages[LIGHTMAP_COUNT * level + lmIndex.w], lmIndex.xyz, vec4(color, 0.0));

            // TODO: merge all ray bounces into one lightmap, then remove BIT_USED here for infinite bounces
            uint storeValue = ((level + 1) == RAYS_INDIRECT) ? ((level + 1) | BIT_USED) : (level + 1);
            lmSync.items[lmIndex.w][lmIndex.x][lmIndex.y][lmIndex.z] = storeValue;
        } else {
            uint storeValue = level;
            lmSync.items[lmIndex.w][lmIndex.x][lmIndex.y][lmIndex.z] = storeValue;
        }
    }
}
