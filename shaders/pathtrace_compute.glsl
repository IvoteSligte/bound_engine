#version 460

#include "compute_includes.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform PathtracePushConstants {
    vec4 rot;
    vec3 pos;
    float time;
    vec4 pRot;
    vec4 ipRot; // inverse previous rotation
    vec3 pPos; // previous position
} pc;

const uint RAYS_PER_SAMPLE = 4;
const float GAMMA = RAYS_PER_SAMPLE * 0.25;

struct Ray {
    vec3 origin; // origin
    vec3 direction; // direction
    vec3 normalOfObject; // normalOfObject at origin (or vec3(-0.0))
    vec3 color; // color left after previous absorptions
    float distanceToObject;
    uint objectHit;
};

// TODO: specialization constants?
layout(binding = 0) uniform restrict readonly MutableData {
    Material mats[MAX_MATERIALS];
    Object objs[MAX_OBJECTS];
} buf;

// TODO: specialization constants?
layout(binding = 1) uniform restrict readonly ConstantBuffer {
    vec2 ratio; // window height / width * fov
} cs;

layout(binding = 2) uniform sampler2D accumulatorImage; // temporal accumulator image
layout(binding = 3, rgba16f) uniform restrict writeonly image2D dataOutputImage;
layout(binding = 4, rgba32f) uniform restrict writeonly image2D normalsDepthImage;
layout(binding = 5, r16f) uniform restrict image2D historyLengthImage;

layout(binding = 6) uniform sampler2D blueNoiseSampler;

// pseudo random number generator
// https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
float rand(vec2 co){
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

vec3 rotate(vec4 q, vec3 v) {
    vec3 t = cross(q.xyz, v) + q.w * v;
    return v + 2.0 * cross(q.xyz, t);
}

vec3 randomDirection(vec3 normal, float index) {
    const float PI = 3.14159265;
    const float PI_2 = PI * 0.5;

    const float imageW = textureSize(blueNoiseSampler, 0).x;

    vec2 offset = texture(blueNoiseSampler, vec2(index, index / imageW)).xy;
    offset *= vec2(PI_2, PI);

    // atan(a / b) gives a different, incorrent number than atan(a, b)
    // vec2(longitude, latitude) or vec2(phi, theta)
    vec2 t = vec2(atan(normal.y, normal.x), atan(normal.z, length(normal.xy))) + offset;

    vec2 s = sin(t);
    vec2 c = cos(t);

    return vec3(c.x*c.y, s.x*c.y, s.y);
}

// calculates the distance from a ray (Ray) to a sphere (Object)
// returns a negative value when the sphere is not hit
float distanceToObject(Ray ray, Object obj) {
    vec3 v = obj.pos - ray.origin;
    float a = dot(ray.direction, v); // distance to plane through obj.pos perpendicular to ray.direction
    return a - sqrt(obj.radiusSquared - dot(v, v) + a * a);
}

void traceRay(inout Ray ray) {
    ray.distanceToObject = 1e20;
    ray.objectHit = MAX_OBJECTS;

    // TODO: hardcode the number of intersection tests (3x speedup)
    for (uint i = 0; i < MAX_OBJECTS; i++) {
        float d = distanceToObject(ray, buf.objs[i]);

        if (d > 0.0 && d < ray.distanceToObject) {
            ray.distanceToObject = d;
            ray.objectHit = i;
        }
    }
}

void updateRay(inout Ray ray, float dirIdx) {
    ray.origin += ray.direction * ray.distanceToObject;
    ray.normalOfObject = ray.origin - buf.objs[ray.objectHit].pos;
    ray.direction = randomDirection(ray.normalOfObject, dirIdx);
}

void shade(inout Ray ray, inout vec4 data) {
    Material material = buf.mats[buf.objs[ray.objectHit].mat];

    data.rgb += ray.color * material.emittance;
    // rays are fired according to brdf, negating the need to calculate it here
    ray.color *= material.reflectance;
}

void main() {
    const ivec2 viewport = ivec2(imageSize(dataOutputImage).xy);

    // maps FragCoord to xy range [-1.0, 1.0]
    vec2 normCoord = gl_GlobalInvocationID.xy * 2.0 / viewport - 1.0;
    // maps normCoord to a different range (e.g. for FOV and non-square windows)
    normCoord *= cs.ratio;

    vec4 data = vec4(0.0);
    vec3 viewDir = rotate(pc.rot, normalize(vec3(normCoord.x, 1.0, normCoord.y)));

    Ray ray = Ray(pc.pos, viewDir, vec3(0.0), vec3(1.0), 0.0, 0);

    float dirIdx = pc.time * 32145.313 + sin(dot(gl_GlobalInvocationID.xy, vec2(12.9898, 78.233))) * 43758.5453;

    traceRay(ray);
    updateRay(ray, dirIdx);
    shade(ray, data);

    Ray rayDirect = ray;
    for (uint r = 0; r < RAYS_PER_SAMPLE; r++) {
        dirIdx += 1;
        traceRay(ray);
        updateRay(ray, dirIdx);
        shade(ray, data);
    }

    // screen space coordinate from global point
    vec3 p = normalize(rayDirect.origin - pc.pPos);
    vec3 r = rotate(pc.ipRot, p);
    vec2 i = r.xz / r.y / cs.ratio; // [-1, 1]
    i = (i + 1.0) * viewport * 0.5; // [0, viewport]

    Ray prevRayDirect = Ray(pc.pPos, p, vec3(0.0), vec3(1.0), 0.0, 0);
    traceRay(prevRayDirect);

    data.rgb /= GAMMA;

    float historyLength = 1.0;

    float moment2 = luminanceFromRGB(data.rgb);
    vec2 moment = vec2(moment2, moment2 * moment2);

    // TODO: improve acceptance criteria
    if (prevRayDirect.objectHit == rayDirect.objectHit && all(lessThan(i, viewport - 1.0)) && all(greaterThan(i, vec2(0.0)))) {
        vec2 ni = (i + 0.5) / viewport;

        historyLength = imageLoad(historyLengthImage, ivec2(gl_GlobalInvocationID.xy)).x;
        historyLength = min(32.0, historyLength + 1.0);

        vec4 tData = texture(accumulatorImage, ni);

        float moment1 = luminanceFromRGB(tData.rgb);
        moment = mix(vec2(moment1, moment1 * moment1), moment, 1.0 / historyLength);

        data.rgb = mix(tData.rgb, data.rgb, 1.0 / historyLength);
    }

    imageStore(historyLengthImage, ivec2(gl_GlobalInvocationID.xy), vec4(historyLength, vec3(0.0)));

    // variance
    data.a = max(moment.y - moment.x * moment.x, 0.0);

    imageStore(normalsDepthImage, ivec2(gl_GlobalInvocationID.xy), vec4(normalize(rayDirect.normalOfObject), rayDirect.distanceToObject));
    imageStore(dataOutputImage, ivec2(gl_GlobalInvocationID.xy), data);
}