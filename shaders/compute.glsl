#version 460

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(push_constant) uniform PushConstantData {
    vec4 rot;
    vec3 pos;
    float time;
    vec4 ipRot; // inverse previous rotation
    vec3 pPos; // previous position
} pc;

const uint MAX_OBJECTS = 8;
const uint RAYS_PER_SAMPLE = 4;

struct Material {
    vec3 reflectance;
    vec3 emittance;
};

struct Object {
    vec3 pos;
    float sizeSquared;
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
    vec2 view; // window sizeSquared
    vec2 ratio; // window height / width
} cs;

layout(binding = 2, rgba8) uniform restrict writeonly image2D renderImage;
layout(binding = 3, rgba16f) uniform readonly image2D accumulatorImageRead; // TODO: sampler for this one
layout(binding = 4, rgba16f) uniform writeonly image2D accumulatorImageWrite;

vec3 rotate(vec4 q, vec3 v) {
    vec3 t = cross(q.xyz, v) + q.w * v;
    return v + 2.0 * cross(q.xyz, t);
}

// pseudo random number generator
// https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
float rand(vec2 co) {
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}
vec3 rand(vec3 s) {
    return vec3(rand(s.xy), rand(s.yz), rand(s.zx)) * 2.0 - 1.0;
}

// TODO: try precomputed blue noise
vec3 randomUnitVectorOnHemisphere(vec3 n, vec3 seed) {
    vec3 v = rand(seed);
    v = sqrt(-2.0 * log(abs(v))) * sign(v); // maps to normal distribution
    return faceforward(normalize(v), -v, n);
}

// calculates the distance from a ray (Ray) to a sphere (Object)
// returns a negative value when the sphere is not hit
float distanceToObject(Ray ray, Object obj) {
    vec3 v = obj.pos - ray.origin;
    float a = dot(ray.direction, v); // distance to plane through obj.pos perpendicular to ray.direction
    return a - sqrt(obj.sizeSquared - dot(v, v) + a * a);
}

void traceRay(inout Ray ray) {
    ray.distanceToObject = 1e20;

    // TODO: hardcode the number of intersection tests (3x speedup)
    for (uint i = 0; i < MAX_OBJECTS; i++) {
        float d = distanceToObject(ray, buf.objs[i]);

        if (d > 0.0 && d < ray.distanceToObject) {
            ray.distanceToObject = d;
            ray.objectHit = i;
        }
    }
}

void updateRay(inout Ray ray, vec3 seed) {
    ray.origin += ray.direction * ray.distanceToObject;
    ray.normalOfObject = normalize(ray.origin - buf.objs[ray.objectHit].pos);
    ray.direction = randomUnitVectorOnHemisphere(ray.normalOfObject, seed);
}

void shade(inout Ray ray, inout vec4 fragData) {
    Material material = buf.mats[ray.objectHit];
    
    // max() makes sure the image does not gradually get darker
    float cos_theta = max(dot(ray.direction, ray.normalOfObject), 0.0);

    vec3 BRDF = material.reflectance;

    fragData.rgb += ray.color * material.emittance;
    ray.color *= cos_theta * BRDF * 2.0;
}

// // TODO: move to a more suitable location
shared vec4 fragDataShared[gl_WorkGroupSize.x][gl_WorkGroupSize.y][gl_WorkGroupSize.z];

vec4 bilinearAccumulatorLoad(vec2 i) {
    ivec2 tl = ivec2(i);
    ivec2 br = tl + ivec2(1);
    vec2 f = i - tl;

    vec4 x0 = imageLoad(accumulatorImageRead, tl);
    vec4 x1 = imageLoad(accumulatorImageRead, ivec2(br.x, tl.y));

    vec4 y0 = imageLoad(accumulatorImageRead, ivec2(tl.x, br.y));
    vec4 y1 = imageLoad(accumulatorImageRead, br);

    return mix(mix(x0, x1, f.x), mix(y0, y1, f.x), f.y);
}

void main() {
    // maps FragCoord to xy range [-1.0, 1.0]
    vec2 normCoord = gl_GlobalInvocationID.xy * 2.0 / cs.view - 1.0;
    // maps normCoord to a different range (e.g. for FOV and non-square windows)
    normCoord *= cs.ratio;

    vec2 s = normCoord.xy * 7.9 + gl_LocalInvocationID.z + pc.time + pc.pos.xy + pc.rot.yz;
    normCoord += (vec2(rand(s), rand(-s)) - 0.5) / cs.view;

    vec4 fragData = vec4(0.0);
    vec3 viewDir = rotate(pc.rot, normalize(vec3(normCoord.x, 1.0, normCoord.y)));

    Ray ray = Ray(pc.pos, viewDir, vec3(0.0), vec3(1.0), 0.0, 0);

    vec3 seed = pc.time + viewDir;

    traceRay(ray);
    updateRay(ray, seed);
    shade(ray, fragData);

    Ray baseRay = ray;
    for (uint r = 0; r < RAYS_PER_SAMPLE; r++) {
        seed += 1.0;
        traceRay(ray);
        updateRay(ray, seed);
        shade(ray, fragData);
    }

    // TODO: move blurring to different shader
    // TODO: figure out a way to blur based on variance

    fragDataShared[gl_LocalInvocationID.x][gl_LocalInvocationID.y][gl_LocalInvocationID.z] = fragData;
    memoryBarrierShared();
    barrier();

    if (gl_LocalInvocationID.z == 0) {
        for (uint z = 1; z < gl_WorkGroupSize.z; z++) {
            fragData += fragDataShared[gl_LocalInvocationID.x][gl_LocalInvocationID.y][z];
        }
        fragData.rgb /= 255.0; // number of samples
        fragData /= gl_WorkGroupSize.z;

        vec3 p = normalize(baseRay.origin - pc.pPos);
        vec3 r = rotate(pc.ipRot, p);
        vec2 i = r.xz / r.y / cs.ratio; // [-1, 1]
        i = (i + 1.0) * cs.view / 2.0; // [0, cs.view]

        Ray pRay = Ray(pc.pPos, p, vec3(0.0), vec3(1.0), 0.0, 0);
        traceRay(pRay);

        if (pRay.objectHit == baseRay.objectHit && clamp(i, vec2(0.0), cs.view - 1) == i) { // TODO: fix yellow appearing on red object when sample is invalid
            vec4 accData = bilinearAccumulatorLoad(i);
            fragData.rgb *= uint(accData.w < 255.0);
            fragData += accData; // TODO: fix formats (rgb/bgr)
        }
        fragData.w = min(fragData.w + 1.0, 255.0);

        imageStore(accumulatorImageWrite, ivec2(gl_GlobalInvocationID.xy), fragData); // accData.w = sample count
        imageStore(renderImage, ivec2(gl_GlobalInvocationID.xy), vec4(fragData.bgr * (255.0 / fragData.w), 1.0));
    }
}

// TODO: add another compute shader with a sampler for the raw fragData output from this shader
// TODO: add parallel sampling
