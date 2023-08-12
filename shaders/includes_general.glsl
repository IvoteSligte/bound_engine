#define LM_UNIT_SIZE 0.5 // TODO: sync this with the rust code (defines)

#define FLT_MAX 3.402823466e+38

#define EPSILON 1e-5

#define SH_cosLobe_C0 0.886226925 // sqrt(pi)/2
#define SH_cosLobe_C1 1.02332671 // sqrt(pi/3)
#define SH_cosLobe_C2 0.495415912 // sqrt(5*pi)/8

struct Material {
    vec3 reflectance;
    vec3 emittance;
};

struct Object {
    vec3 position;
    float radius;
    uint material;
};

struct PackedVoxel {
    uvec2 emittance;
    uint reflectance;
    uint normal;
};

struct Voxel {
    vec3 emittance;
    vec3 reflectance;
    vec3 normal;
};

PackedVoxel packVoxel(Voxel v) {
    return PackedVoxel(
        uvec2(packHalf2x16(v.emittance.xy), packHalf2x16(vec2(v.emittance.z, 0.0))),
        packUnorm4x8(vec4(v.reflectance, 0.0)),
        packSnorm4x8(vec4(v.normal, 0.0))
    );
}

Voxel unpackVoxel(PackedVoxel v) {
    return Voxel(
        vec3(unpackHalf2x16(v.emittance.x), unpackHalf2x16(v.emittance.y).x),
        unpackUnorm4x8(v.reflectance).rgb,
        unpackSnorm4x8(v.normal).xyz
    );
}

vec3 rotateWithQuat(vec4 q, vec3 v) {
    vec3 t = q.w * v + cross(q.xyz, v);
    return 2.0 * cross(q.xyz, t) + v;
}

float maximum(vec3 v) {
    return max(max(v.x, v.y), v.z);
}

int lmLayerAtPos(vec3 v, vec3 lmOrigin) {
    const float INV_HALF_LM_SIZE = 1.0 / (float(LM_SIZE / 2) * LM_UNIT_SIZE);
    return int(log2(max(maximum(abs(v - lmOrigin)) * INV_HALF_LM_SIZE, 0.5)) + 1.001);
}

float lmUnitSizeLayer(uint lmLayer) {
    return float(1 << lmLayer) * LM_UNIT_SIZE;
}

float lmMultsLayer(uint lmLayer) {
    return (1.0 / float(LM_SIZE)) / lmUnitSizeLayer(lmLayer);
}

float lmSizeLayer(uint lmLayer) {
    return float(LM_SIZE) * lmUnitSizeLayer(lmLayer);
}

float lmHalfSizeLayer(uint lmLayer) {
    return (float(LM_SIZE) * 0.5) * lmUnitSizeLayer(lmLayer);
}

/// returns an index into a lightmap image in xyz, and the image index in w
// TODO: origin per layer
ivec3 lmIndexAtPos(vec3 pos, vec3 origin, out int layer) {
    layer = lmLayerAtPos(pos, origin);
    return ivec3(floor((pos - origin) / lmUnitSizeLayer(layer))) + LM_SIZE / 2;
}

vec3 lmIndexAtPosF(vec3 pos, vec3 origin, out int layer) {
    layer = lmLayerAtPos(pos, origin);
    return (pos - origin) / float(lmUnitSizeLayer(layer)) + float(LM_SIZE / 2);
}

// TODO: origin per layer
vec3 posAtLmIndex(ivec3 index, int layer, vec3 origin) {
    return (vec3(index - LM_SIZE / 2) + 0.5) * lmUnitSizeLayer(layer) + origin;
}

float radUnitSizeLayer(uint layer) {
    return lmUnitSizeLayer(layer) * float(LM_SIZE / RADIANCE_SIZE);
}

vec3 posAtRadIndex(ivec3 index, int layer) {
    return (vec3(index - RADIANCE_SIZE / 2) + 0.5) * radUnitSizeLayer(layer); // TODO: movable origin
}

vec3 posToLmTextureCoord(vec3 pos, uint layer, vec3 origin) {
    return ((pos - origin) / lmSizeLayer(layer)) + 0.5;
}

ivec3 radIndexAtPos(vec3 pos, out int layer) {
    ivec3 index = lmIndexAtPos(pos, vec3(0.0), layer);
    index.xyz /= (LM_SIZE / RADIANCE_SIZE);
    return index; // TODO: movable origin
}

vec3 radIndexAtPosF(vec3 pos, out int layer) {
    vec3 index = lmIndexAtPosF(pos, vec3(0.0), layer);
    index.xyz /= float(LM_SIZE / RADIANCE_SIZE);
    return index;
}

// turns a normal image index into a normalized texture index
vec3 radIndexTexture(vec3 index, int layer) {
    vec3 texIndex = index * (1.0 / float(RADIANCE_SIZE));
    return texIndex;
}

bool marchRay(sampler3D[LM_LAYERS] SDFImages, inout vec3 pos, vec3 dir, vec3 sdfOrigin, float threshold, uint samples, inout float totalDist) {
    float dist = totalDist;

    vec3 dPos = dir * threshold;

    uint layer = lmLayerAtPos(pos, sdfOrigin);

    for (uint i = 0; i < samples; i++) {
        pos += dir * dist;

        vec3 idx = pos - sdfOrigin; // TODO: sdfOrigin varying between layers
        bool outOfLayer = maximum(abs(idx)) > lmHalfSizeLayer(layer);
        bool coneTooBig = threshold * totalDist > 0.5 * lmUnitSizeLayer(layer);
        bool increaseLayer = outOfLayer || coneTooBig;
        layer = increaseLayer ? layer + 1 : layer;

        float mult = lmMultsLayer(layer);

        vec3 texIdx = (pos - sdfOrigin) * mult + 0.5; // TODO: sdfOrigin varying between layers
        dist = texture(SDFImages[layer], texIdx).x;
        totalDist += dist;

        if (dist <= threshold * totalDist) { // hit or out of bounds
            return layer < LM_LAYERS;
        }
    }

    return false;
}

vec3 normalizeZeroIfNaN(vec3 v) {
    return v == vec3(0.0) ? v : normalize(v);
}

// Originally sourced from https://www.shadertoy.com/view/ldfSWs
// Returns vec3(NaN) if normal == vec3(0.0)
// pos should be normalized according to the texture
vec3 calcNormalSDF(sampler3D sdf, vec3 pos, float eps) {
    const vec3 v1 = vec3( 1.0,-1.0,-1.0);
    const vec3 v2 = vec3(-1.0,-1.0, 1.0);
    const vec3 v3 = vec3(-1.0, 1.0,-1.0);
    const vec3 v4 = vec3( 1.0, 1.0, 1.0);

    return normalize(v1 * texture(sdf, pos + v1*eps).x +
                     v2 * texture(sdf, pos + v2*eps).x +
                     v3 * texture(sdf, pos + v3*eps).x +
                     v4 * texture(sdf, pos + v4*eps).x);
}

vec3 calcNormalSDF(sampler3D sdf, vec3 pos) {
    return calcNormalSDF(sdf, pos, 0.002);
}

vec3 evaluateRGBSphericalHarmonics(vec3 dir, vec3[SH_CS] coefs) {
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;

    vec3 s = vec3(0.0);

    s += coefs[0] * 0.28209479;

    s += coefs[1] * 0.48860251 * y;
    s += coefs[2] * 0.48860251 * z;
    s += coefs[3] * 0.48860251 * x;

    // s += coefs[4] * 1.09254843 * x * y;
    // s += coefs[5] * 1.09254843 * y * z;
    // s += coefs[6] * 0.31539156 * (3 * z * z - 1);
    // s += coefs[7] * 1.09254843 * x * z;
    // s += coefs[8] * 0.54627421 * (x * x - y * y);

    return s;
}

// credit to https://ericpolman.com/2016/06/28/light-propagation-volumes/
vec4 dirToCosineLobe(vec3 dir) {
    //dir = normalize(dir);
    return vec4(SH_cosLobe_C0, -SH_cosLobe_C1 * dir.y, SH_cosLobe_C1 * dir.z, -SH_cosLobe_C1 * dir.x);
}

// credit to https://iquilezles.org/articles/distfunctions/
// p is the sample point relative to the box center,
// b is the corner of the box relative to the box center
float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// Function to calculate overlapping volume between a cube and a cuboid
float overlappingVolume(vec3 cubePosition, float cubeSize, vec3 cuboidMin, vec3 cuboidMax) {
    vec3 cubeMin = cubePosition - 0.5 * cubeSize;
    vec3 cubeMax = cubePosition + 0.5 * cubeSize;

    // The ranges of positions along each dimension
    vec3 minRange = max(cubeMin, cuboidMin);
    vec3 maxRange = min(cubeMax, cuboidMax);
    
    vec3 overlapDimensions = max(vec3(0.0), maxRange - minRange);
    
    float overlappingVolume = overlapDimensions.x * overlapDimensions.y * overlapDimensions.z;
    return overlappingVolume;
}
