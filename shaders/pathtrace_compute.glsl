#version 460

#include "compute_includes.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

const uint RAYS_PER_SAMPLE = 4;

struct Bounds { // node of a binary tree
    vec3 center;
    float radiusSquared;
    uint left;
    uint right;
    uint leaf;
    uint parent;
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
    Bounds volumes[2 * MAX_OBJECTS];
} bvh;

layout(binding = 2) uniform restrict readonly MutableData {
    Material mats[MAX_MATERIALS];
} buf;

// TODO: specialization constants?
layout(binding = 3) uniform restrict readonly ConstantBuffer {
    vec2 ratio; // window height / width * fov
} cs;

layout(binding = 4) uniform sampler2D blueNoiseTexture;

layout(binding = 5) uniform sampler2D accumulatorTexture; // temporal accumulator image
layout(binding = 6, rgba16f) uniform restrict writeonly image2D dataOutputImage;
layout(binding = 7) uniform sampler2D temporalMomentsTexture; // stores previous frame's moments
layout(binding = 8, rg32f) uniform restrict writeonly image2D momentsImage;

layout(set = 1, binding = 0, rgba32f) uniform restrict writeonly image2D normalsDepthImage;
layout(set = 1, binding = 1, r8ui) uniform restrict uimage2D historyLengthImage;

// pseudo random number generator
// https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
float rand(vec2 co){
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

vec3 rotate(vec4 q, vec3 v) {
    vec3 t = cross(q.xyz, v) + q.w * v;
    return v + 2.0 * cross(q.xyz, t);
}

// NOTE: dot(ray.direction, ray.normalOfObject) gives higher results when looking at surfaces pointing up/down
vec3 randomDirection(vec3 normal, float index) {
    const float PI = 3.14159265;
    const float PI_2 = PI * 0.5;

    const float imageW = textureSize(blueNoiseTexture, 0).x;

    vec2 offset = texture(blueNoiseTexture, vec2(index, index / imageW)).xy;
    offset *= vec2(PI, PI_2);

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
    float d = bnd.radiusSquared + a * a - b;
    is_inside = b < bnd.radiusSquared;
    if (d < 0.0) { return -1.0; }
    return a - sqrt(d);
}

void traceRayWithBVH(inout Ray ray) {
    ray.distanceToObject = 1e20;
    ray.nodeHit = 0;

    // TODO: add a cpu-side limit to the number of layers
    uint stack[32]; // max number of layers is 32 because of pre-set stack size
    stack[0] = bvh.head;
    int i = 0;

    while (i >= 0) {
        Bounds node = bvh.volumes[stack[i]];

        bool is_inside;
        float d = distanceToObject(ray, node, is_inside);
        bool is_hit = d > 0.0 && d < ray.distanceToObject;

        // not a leaf, move to left child
        if (node.leaf == 0 && (is_inside || is_hit)) {
            i += 1;
            stack[i] = node.left;
        }
        else {
            if (is_hit) {
                // is a leaf, store data
                ray.distanceToObject = d;
                ray.nodeHit = stack[i];
            }

            // move to next node (to the right)
            i -= 1;
            stack[i] = bvh.volumes[stack[i]].right;
        }
    }
}

void updateRay(inout Ray ray, float dirIdx) {
    ray.origin += ray.direction * ray.distanceToObject;
    ray.normalOfObject = ray.origin - bvh.volumes[ray.nodeHit].center;
    ray.direction = randomDirection(ray.normalOfObject, dirIdx);
}

void shade(inout Ray ray, inout vec4 data) {
    Material material = buf.mats[bvh.volumes[ray.nodeHit].leaf];

    data.rgb += ray.color * material.emittance;
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

    float dirIdx = rt.time * 32145.313 + fract(sin(dot(gl_GlobalInvocationID.xy, vec2(12.9898, 78.233)))) * 43758.5453;

    traceRayWithBVH(ray);
    updateRay(ray, dirIdx);
    shade(ray, data);

    Ray rayDirect = ray;

    for (uint r = 0; r < RAYS_PER_SAMPLE; r++) {
        dirIdx += 1.0;
        traceRayWithBVH(ray);
        updateRay(ray, dirIdx);
        shade(ray, data);
    }

    // screen space coordinate from global point
    vec3 p = normalize(rayDirect.origin - rt.previousPosition);
    vec3 r = rotate(rt.inversePreviousRotation, p);
    vec2 i = r.xz / r.y / cs.ratio; // [-1, 1]
    i = (i + 1.0) * viewport * 0.5; // [0, viewport]

    Ray prevRayDirect = Ray(rt.previousPosition, p, vec3(0.0), vec3(1.0), 0.0, 0);
    traceRayWithBVH(prevRayDirect);

    float historyLength = 1.0;

    float l = luminanceFromRGB(data.rgb);
    vec2 moments = vec2(l, l * l);

    // TODO: improve acceptance criteria
    if (prevRayDirect.nodeHit == rayDirect.nodeHit && all(lessThan(i, viewport - 1.0)) && all(greaterThan(i, vec2(0.0)))) {
        vec2 ni = (i + 0.5) / viewport;

        historyLength = imageLoad(historyLengthImage, ipos).x;
        historyLength = min(32.0, historyLength + 1.0);

        vec4 tData = texture(accumulatorTexture, ni);
        vec2 tMoments = texture(temporalMomentsTexture, ni).xy;

        moments = mix(tMoments, moments, 1.0 / historyLength);
        data.a = max(moments.y - moments.x * moments.x, 0.0);
        data.rgb = mix(tData.rgb, data.rgb, 1.0 / historyLength);
    }

    imageStore(historyLengthImage, ipos, uvec4(historyLength, uvec3(0)));
    imageStore(normalsDepthImage, ipos, vec4(normalize(rayDirect.normalOfObject), rayDirect.distanceToObject));
    imageStore(momentsImage, ipos, vec4(moments, vec2(0.0)));
    imageStore(dataOutputImage, ipos, data);
}
