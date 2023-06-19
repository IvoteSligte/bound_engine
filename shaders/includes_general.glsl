const uint ALL_ONES = 4294967295;

const float LM_UNIT_SIZE = 0.5; // TODO: sync this with the rust code (defines)

#define FLT_MAX 3.402823466e+38

#define EPSILON 1e-5

struct Material {
    vec3 reflectance;
    vec3 emittance;
};

struct Object {
    vec3 position;
    float radius;
    uint material;
};

struct Radiance {
    uint packed[64]; // INFO: may need to be packed into `uvec4`s, 16 byte array elements might be more performant
};

vec3 rotateWithQuat(vec4 q, vec3 v) {
    vec3 t = q.w * v + cross(q.xyz, v);
    return 2.0 * cross(q.xyz, t) + v;
}

float maximum(vec3 v) {
    return max(max(v.x, v.y), v.z);
}

const int HALF_LM_SIZE = LM_SIZE / 2;
const float INV_HALF_LM_SIZE = 1.0 / (float(HALF_LM_SIZE) * LM_UNIT_SIZE);

uvec4 unpackBytesUint(uint bytes) {
    return uvec4(
         bytes        & 255, // [0, 8)
        (bytes >>  8) & 255, // [8, 16)
        (bytes >> 16) & 255, // [16, 24)
        (bytes >> 24) & 255  // [24, 32)
    );
}

// packs 4 bytes into a uint, assumes the inputs are in the range [0, 255]
uint packBytesUint(uvec4 bytes) {
    return uint(
        bytes.x | (bytes.y << 8) | (bytes.z << 16) | (bytes.w << 24)
    );
}

int lmLayerAtPos(vec3 v, vec3 lmOrigin) {
    return int(log2(max(maximum(abs(v - lmOrigin)) * INV_HALF_LM_SIZE, 0.5)) + 1.001);
}

float lmUnitSizeLayer(uint lmLayer) {
    return float(1 << lmLayer) * LM_UNIT_SIZE;
}

float lmMultsLayer(uint lmLayer) {
    return (1.0 / float(LM_SIZE)) / lmUnitSizeLayer(lmLayer);
}

float lmCompsLayer(uint lmLayer) {
    return (float(LM_SIZE) * 0.5) * lmUnitSizeLayer(lmLayer);
}

/// returns an index into a lightmap image in xyz, and the image index in w
// TODO: lmOrigin per layer
ivec4 lmIndexAtPos(vec3 pos, vec3 lmOrigin) {
    uint lmLayer = lmLayerAtPos(pos, lmOrigin);
    ivec3 index = ivec3(floor((pos - lmOrigin) / lmUnitSizeLayer(lmLayer))) + HALF_LM_SIZE;

    return ivec4(index, lmLayer);
}

// TODO: lmOrigin per layer
vec3 posAtLmIndex(ivec4 lmIndex, vec3 lmOrigin) {
    return (vec3(lmIndex.xyz - HALF_LM_SIZE) + 0.5) * lmUnitSizeLayer(lmIndex.w) + lmOrigin;
}

ivec4 radIndexToLmIndex(uvec4 radIndex) {
    return ivec4(radIndex.xyz * (LM_SIZE / RADIANCE_SIZE), radIndex.w);
}

vec3 posAtRadIndex(uvec4 radIndex) { // FIXME: scale with LM_SIZE
    return posAtLmIndex(radIndexToLmIndex(radIndex), vec3(0.0)); // TODO: movable origin
}

float radUnitSizeLayer(uint lmLayer) {
    return lmUnitSizeLayer(lmLayer) * float(LM_SIZE / RADIANCE_SIZE);
}

uvec4 radIndexAtPos(vec3 pos) {
    return uvec4(lmIndexAtPos(pos, vec3(0.0))) / (LM_SIZE / RADIANCE_SIZE); // TODO: movable origin
}

bool marchRay(sampler3D[LM_LAYERS] SDFImages, inout vec3 pos, vec3 dir, vec3 sdfOrigin, float threshold, uint samples, inout float totalDist) {
    float dist = totalDist;

    vec3 dPos = dir * threshold;

    uint layer = lmLayerAtPos(pos, sdfOrigin);

    for (uint i = 0; i < samples; i++) {
        pos += dir * dist;

        vec3 idx = pos - sdfOrigin; // TODO: sdfOrigin varying between layers
        bool outOfLayer = maximum(abs(idx)) > lmCompsLayer(layer);
        bool coneTooBig = threshold * totalDist > 0.5 * lmUnitSizeLayer(layer);
        bool increaseLayer = outOfLayer || coneTooBig;
        layer = increaseLayer ? layer + 1 : layer;

        float mult = lmMultsLayer(layer);

        vec3 texIdx = (pos - sdfOrigin) * mult + 0.5; // TODO: sdfOrigin varying between layers
        dist = texture(SDFImages[layer], clamp(texIdx, 0.0, 1.0)).x;
        totalDist += dist;

        if (dist <= threshold * totalDist) { // hit or out of bounds
            return layer < LM_LAYERS;
        }
    }

    return false;
}

// Originally sourced from https://www.shadertoy.com/view/ldfSWs
vec3 calcNormalSDF(sampler3D sdf, vec3 pos, float eps) {
  const vec3 v1 = vec3( 1.0,-1.0,-1.0);
  const vec3 v2 = vec3(-1.0,-1.0, 1.0);
  const vec3 v3 = vec3(-1.0, 1.0,-1.0);
  const vec3 v4 = vec3( 1.0, 1.0, 1.0);

  return normalize( v1 * texture(sdf, pos + v1*eps).x +
                    v2 * texture(sdf, pos + v2*eps).x +
                    v3 * texture(sdf, pos + v3*eps).x +
                    v4 * texture(sdf, pos + v4*eps).x );
}

vec3 calcNormalSDF(sampler3D sdf, vec3 pos) {
  return calcNormalSDF(sdf, pos, 0.002);
}

// direct version of fibonacci_sphere algorithm (https://gist.github.com/Seanmatthews/a51ac697db1a4f58a6bca7996d75f68c)
// uses 64 points
vec3 directFibonacciSphere(float i) {
    const float GA = 2.39996322973; // golden angle

    float theta = GA * i;

    float z = mix((1.0 / 64) - 1.0, 1.0 - (1.0 / 64), i / (64.0 - 1.0));

    float radius = sqrt(1.0 - z * z);

    float y = radius * sin(theta);
    float x = radius * cos(theta);

    return vec3(x, y, z);
}

uint inverseFibonacciSphere(vec3 v) {
    const float PI = 3.1415927;
    const float GA = 2.3999632; // golden angle
    const float N = 64.0;
    
    float theta = atan(v.x, v.y);
    
    float z_index = (N - 1.0) - ((N - 1.0) / 2.0) * (v.z / (1.0 / N - 1.0) + 1.0);
    
    float z_theta = mod(GA * z_index, 2.0 * PI);
    
    float deviation = fract((theta - z_theta) / (2.0 * PI) + 0.5) - 0.5;
    
    float circles = (GA * N) / (2.0 * PI);
    
    float indices_circle = N / circles;
    
    float index_offset = deviation * indices_circle;
    
    return uint(z_index + index_offset);
}
