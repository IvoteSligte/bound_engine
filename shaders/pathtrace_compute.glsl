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
    ivec2 view; // window size
    vec2 ratio; // window height / width * fov
} cs;

layout(binding = 2) uniform sampler2D accumulatorImageRead;
layout(binding = 3, rgba16f) uniform restrict writeonly image2D accumulatorImageWrite;
layout(binding = 4, rgba32f) uniform restrict writeonly image2D normalsDepthImage;
layout(binding = 5, r8ui) uniform restrict writeonly uimage2D materialImage;
layout(binding = 6) uniform sampler2D prevMomentImage; // TODO: momentImage output
layout(binding = 7, r16f) uniform restrict writeonly image2D curMomentImage;
layout(binding = 8) uniform sampler2D prevVarianceImage;
layout(binding = 9, r16f) uniform restrict writeonly image2D curVarianceImage;

layout(binding = 10) uniform sampler2D blueNoiseSampler;

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

void shade(inout Ray ray, inout vec4 fragData) {
    Material material = buf.mats[buf.objs[ray.objectHit].mat];

    // rays are fired to account for diffuse falloff
    vec3 BRDF = material.reflectance * 2.0;

    fragData.rgb += ray.color * material.emittance;
    ray.color *= BRDF;
}

// TODO: fix samples per pixel
void main() {
    // maps FragCoord to xy range [-1.0, 1.0]
    vec2 normCoord = gl_GlobalInvocationID.xy * 2.0 / cs.view - 1.0;
    // maps normCoord to a different range (e.g. for FOV and non-square windows)
    normCoord *= cs.ratio;

    vec4 fragData = vec4(0.0);
    vec3 viewDir = rotate(pc.rot, normalize(vec3(normCoord.x, 1.0, normCoord.y)));

    Ray ray = Ray(pc.pos, viewDir, vec3(0.0), vec3(1.0), 0.0, 0);

    // TODO: fix
    float dirIdx = pc.time + rand(viewDir.xz) * 1e2;

    traceRay(ray);
    updateRay(ray, dirIdx);
    shade(ray, fragData);

    Ray baseRay = ray;
    for (uint r = 0; r < RAYS_PER_SAMPLE; r++) {
        dirIdx += 1;
        traceRay(ray);
        updateRay(ray, dirIdx);
        shade(ray, fragData);
    }

    // screen space coordinate from global point
    vec3 p = normalize(baseRay.origin - pc.pPos);
    vec3 r = rotate(pc.ipRot, p);
    vec2 i = r.xz / r.y / cs.ratio; // [-1, 1]
    i = (i + 1.0) * cs.view * 0.5; // [0, cs.view]

    Ray pRay = Ray(pc.pPos, p, vec3(0.0), vec3(1.0), 0.0, 0);
    traceRay(pRay);

    if (pRay.objectHit == baseRay.objectHit && clamp(i, vec2(0.0), cs.view - 1.0) == i) {
        vec2 ni = (i + 0.5) / cs.view;

        vec2 moment = texture(prevMomentImage, ni).xy;
        imageStore(curMomentImage, ivec2(gl_GlobalInvocationID.xy), vec4(moment, vec2(0.0)));

        float variance = texture(prevVarianceImage, ni).x;
        imageStore(curVarianceImage, ivec2(gl_GlobalInvocationID.xy), vec4(variance));

        vec4 accData = texture(accumulatorImageRead, ni);
        fragData = mix(accData, fragData, 0.05);

        // TODO: use normals, albedo, material to determine temporal stability
        // accData /= distance(pc.pos, pc.pPos) + 1.0;
        // accData *= dot(rotate(pc.pRot, vec3(0.0, 1.0, 0.0)), rotate(pc.rot, vec3(0.0, 1.0, 0.0)));
    }

    imageStore(normalsDepthImage, ivec2(gl_GlobalInvocationID.xy), vec4(normalize(baseRay.normalOfObject), baseRay.distanceToObject));
    imageStore(materialImage, ivec2(gl_GlobalInvocationID.xy), uvec4(buf.objs[baseRay.objectHit].mat, uvec3(0)));

    imageStore(accumulatorImageWrite, ivec2(gl_GlobalInvocationID.xy), vec4(fragData.rgb, fragData.w + 1.0)); // accData.w = sample count
}
