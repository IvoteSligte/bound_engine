#version 460

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform PushConstantData {
    vec4 rot;
    vec3 pos;
    float time;
    vec4 ipRot; // inverse previous rotation
    vec3 pPos; // previous position
} pc;

const uint MAX_OBJECTS = 8;
const uint SAMPLES_PER_PIXEL = 1;
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
layout(binding = 3, rgba16f) uniform readonly image2D accumulatorImageRead;
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

void initRay(inout Ray ray) {
    ray.distanceToObject = 1e20;
    ray.objectHit = MAX_OBJECTS;
}

void traceRay(inout Ray ray) {
    // TODO: hardcode the number of intersection tests (3x speedup)
    for (uint i = 0; i < buf.objCount; i++) {
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

void shade(inout Ray ray, inout vec3 fragColor) {
    Material material = buf.mats[ray.objectHit];
    
    // max() makes sure the image does not gradually get darker
    float cos_theta = max(dot(ray.direction, ray.normalOfObject), 0.0);

    vec3 BRDF = material.reflectance;

    fragColor += ray.color * material.emittance;
    ray.color *= cos_theta * BRDF * 2.0;
}

// // TODO: move to a more suitable location
shared vec3 fragColorsShared[gl_WorkGroupSize.x][gl_WorkGroupSize.y];

// vec3 sample3x3(vec3 color, float factor) {
//     const ivec2 ce = ivec2(gl_LocalInvocationID.xy); // center
//     const ivec2 tl = max(ce - 1, ivec2(0)); // top left
//     const ivec2 br = min(ce + 1, ivec2(gl_WorkGroupSize.xy) - 1); // bottom right

//     vec3 colors[8];

//     vec3 average = vec3(0.0);

//     colors[0] = fragColorsShared[tl.x][tl.y]; // TODO: fix sample not set sampling
//     average += colors[0];
//     colors[1] = fragColorsShared[ce.x][tl.y];
//     average += colors[1];
//     colors[2] = fragColorsShared[br.x][tl.y];
//     average += colors[2];
//     colors[3] = fragColorsShared[tl.x][ce.y];
//     average += colors[3];
//     // center of 3x3 square is already in `color`
//     colors[4] = fragColorsShared[br.x][ce.y];
//     average += colors[4];
//     colors[5] = fragColorsShared[tl.x][br.y];
//     average += colors[5];
//     colors[6] = fragColorsShared[ce.x][br.y];
//     average += colors[6];
//     colors[7] = fragColorsShared[br.x][br.y];
//     average += colors[7];

//     average /= 8.0;

//     return mix(color, average, factor * 8.0 / 9.0);
// }

vec4 bilinearAccumulatorLoad(vec2 i) {
    ivec2 tl = ivec2(i);
    ivec2 br = min(tl + ivec2(1), ivec2(cs.view) - 1);
    vec2 f = i - tl;

    vec4 x0 = imageLoad(accumulatorImageRead, tl);
    vec4 x1 = imageLoad(accumulatorImageRead, ivec2(br.x, tl.y));

    vec4 y0 = imageLoad(accumulatorImageRead, ivec2(tl.x, br.y));
    vec4 y1 = imageLoad(accumulatorImageRead, br);

    return mix(mix(x0, x1, f.x), mix(y0, y1, f.x), f.y);
}

// TODO: blurring last render image based on sample count
// vec4 blur3x3AccumulatorLoad(vec2 i) {
//     ivec2 tl = ivec2(i - 1.5);
//     ivec2 br = ivec2(i + 1.5);
//     vec2 f = i - tl - 1.0;

    
// }

void main() {
    // maps FragCoord to xy range [-1.0, 1.0]
    vec2 normCoord = gl_GlobalInvocationID.xy * 2.0 / cs.view - 1.0;
    // maps normCoord to a different range (e.g. for FOV and non-square windows)
    normCoord *= cs.ratio;

    vec3 fragColor = vec3(0.0);
    vec4 accData = vec4(0.0);
    vec3 viewDir = rotate(pc.rot, normalize(vec3(normCoord.x, 1.0, normCoord.y)));

    Ray ray = Ray(pc.pos, viewDir, vec3(0.0), vec3(1.0), 0.0, 0);

    vec3 seed = pc.time + viewDir;

    initRay(ray);
    traceRay(ray);
    if (ray.objectHit != MAX_OBJECTS) {
        updateRay(ray, seed);
        shade(ray, fragColor);

        Ray baseRay = ray;
        for (uint s = 0; s < SAMPLES_PER_PIXEL; s++) { // TODO: set a good offset for the rays
            ray = baseRay;

            for (uint r = 0; r < RAYS_PER_SAMPLE; r++) {
                seed += 1.0;
                initRay(ray);
                traceRay(ray);
                if (ray.objectHit == MAX_OBJECTS) {
                    ray.direction = randomUnitVectorOnHemisphere(ray.normalOfObject, seed);
                    continue;
                }
                updateRay(ray, seed);
                shade(ray, fragColor);
            }
        }

        fragColor /= SAMPLES_PER_PIXEL;
        fragColor /= 255.0;

        vec3 p = normalize(baseRay.origin - pc.pPos);
        vec3 r = rotate(pc.ipRot, p);
        vec2 i = r.xz / r.y / cs.ratio; // [-1, 1]
        i = (i + 1.0) * cs.view / 2.0; // [0, cs.view]

        Ray pRay = Ray(pc.pPos, p, vec3(0.0), vec3(1.0), 1e20, MAX_OBJECTS);
        traceRay(pRay);

        if (pRay.objectHit == baseRay.objectHit) { // TODO: fix yellow appearing on red object when sample is invalid
            accData = bilinearAccumulatorLoad(i) * float(clamp(i, vec2(0.0), cs.view - 1) == i); // i < 0.0 gives undefined values
            fragColor *= uint(accData.w != 255.0);
            fragColor += accData.rgb; // TODO: fix formats (rgb/bgr)
        }
    }
    accData.w = min(accData.w + 1.0, 255.0);

    // TODO: move to different shader
    // // TODO: figure out a way to blur based on variance
    // fragColorsShared[gl_LocalInvocationID.x][gl_LocalInvocationID.y] = fragColor;
    // memoryBarrierShared();
    // barrier();
    // fragColor = sample3x3(fragColor, 1.0 - accData.w / 255.0);



    imageStore(accumulatorImageWrite, ivec2(gl_GlobalInvocationID.xy), vec4(fragColor, accData.w)); // accData.w = sample count
    imageStore(renderImage, ivec2(gl_GlobalInvocationID.xy), vec4(fragColor.bgr * (255.0 / accData.w), 1.0));
}

// TODO: add another compute shader with a sampler for the raw fragColor output from this shader
// TODO: add parallel sampling
