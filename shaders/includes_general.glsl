#define EPSILON 1e-4

#define SH_cosLobe_C0 0.886226925 // sqrt(pi)/2
#define SH_cosLobe_C1 1.02332671 // sqrt(pi/3)
// #define SH_cosLobe_C2 0.495415912 // sqrt(5*pi)/8

#define SH_norm_C0 0.28209479 // used to normalize l=0, m=0

struct Material {
    vec3 reflectance;
    vec3 emittance;
};

struct PackedVoxel {
    uvec2 emittance;
    uint reflectance;
    uint normalAndIntersections;
};

struct Voxel {
    vec3 emittance;
    vec3 reflectance;
    vec3 normal;
    float intersections;
};

PackedVoxel packVoxel(Voxel v) {
    return PackedVoxel(
        uvec2(packHalf2x16(v.emittance.rg), packHalf2x16(vec2(v.emittance.b, 0.0))),
        packUnorm4x8(vec4(v.reflectance, 0.0)),
        packSnorm4x8(vec4(v.normal, 1.0 / (v.intersections + 1.0)))
    );
}

Voxel unpackVoxel(PackedVoxel v) {
    vec4 normalAndIntersections = unpackSnorm4x8(v.normalAndIntersections);

    return Voxel(
        vec3(unpackHalf2x16(v.emittance.x), unpackHalf2x16(v.emittance.y).x),
        unpackUnorm4x8(v.reflectance).rgb,
        normalAndIntersections.xyz,
        1.0 / normalAndIntersections.w - 1.0
    );
}

vec3 rotateWithQuat(vec4 q, vec3 v) {
    vec3 t = q.w * v + cross(q.xyz, v);
    return 2.0 * cross(q.xyz, t) + v;
}

float maximum(vec3 v) {
    return max(max(v.x, v.y), v.z);
}

int radLayerAtPos(vec3 v, vec3 origin) {
    const float NORM = 1.0 / (float(RADIANCE_SIZE / 2) * RADIANCE_UNIT);
    return int(log2(max(maximum(abs(v - origin)) * NORM, 0.5)) + 1.001);
}

float radUnitSizeLayer(int layer) {
    return float(1 << layer) * RADIANCE_UNIT;
}

vec3 posAtRadIndex(ivec3 index, int layer, vec3 origin) {
    return origin + (vec3(index - RADIANCE_SIZE / 2) + 0.5) * radUnitSizeLayer(layer);
}

vec3 radTextureIndexAtPos(vec3 pos, int layer, vec3 origin) {
    return (pos - origin) / float(radUnitSizeLayer(layer)) * (1.0 / float(RADIANCE_SIZE)) + 0.5;
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
    return vec4(SH_cosLobe_C0, SH_cosLobe_C1 * dir.y, SH_cosLobe_C1 * dir.z, SH_cosLobe_C1 * dir.x);
}

struct AABB {
    vec3 center;
    vec3 halfExtents;
};

bool intersectAABBTriangleSAT(vec3[3] tri, vec3 aabbHalfExtents, vec3 axis) {
    float p0 = dot(tri[0], axis);
    float p1 = dot(tri[1], axis);
    float p2 = dot(tri[2], axis);

    float minP = min(p0, min(p1, p2));
    float maxP = max(p0, max(p1, p2));

    // return abs(maxP) < abs(dot(aabbHalfExtents, axis)) || abs(minP) < abs(dot(aabbHalfExtents, axis)) || (minP < -abs(dot(aabbHalfExtents, axis)) && maxP > abs(dot(aabbHalfExtents, axis)));
    return !(max(-maxP, minP) > dot(aabbHalfExtents, abs(axis)));
}

// credit to https://bronsonzgeb.com/index.php/2021/05/29/gpu-mesh-voxelizer-part-2/
bool intersectAABBTriangle(vec3[3] tri, AABB aabb, out vec3 triangleNormal) {
    tri[0] -= aabb.center;
    tri[1] -= aabb.center;
    tri[2] -= aabb.center;

    vec3 ab = normalize(tri[1] - tri[0]);
    vec3 bc = normalize(tri[2] - tri[1]);
    vec3 ca = normalize(tri[0] - tri[2]);

    //Cross ab, bc, and ca with (1, 0, 0)
    vec3 a00 = vec3(0.0, -ab.z, ab.y);
    vec3 a01 = vec3(0.0, -bc.z, bc.y);
    vec3 a02 = vec3(0.0, -ca.z, ca.y);

    //Cross ab, bc, and ca with (0, 1, 0)
    vec3 a10 = vec3(ab.z, 0.0, -ab.x);
    vec3 a11 = vec3(bc.z, 0.0, -bc.x);
    vec3 a12 = vec3(ca.z, 0.0, -ca.x);

    //Cross ab, bc, and ca with (0, 0, 1)
    vec3 a20 = vec3(-ab.y, ab.x, 0.0);
    vec3 a21 = vec3(-bc.y, bc.x, 0.0);
    vec3 a22 = vec3(-ca.y, ca.x, 0.0);

    triangleNormal = normalize(cross(ab, bc));

    if (
        !intersectAABBTriangleSAT(tri, aabb.halfExtents, a00) ||
        !intersectAABBTriangleSAT(tri, aabb.halfExtents, a01) ||
        !intersectAABBTriangleSAT(tri, aabb.halfExtents, a02) ||
        !intersectAABBTriangleSAT(tri, aabb.halfExtents, a10) ||
        !intersectAABBTriangleSAT(tri, aabb.halfExtents, a11) ||
        !intersectAABBTriangleSAT(tri, aabb.halfExtents, a12) ||
        !intersectAABBTriangleSAT(tri, aabb.halfExtents, a20) ||
        !intersectAABBTriangleSAT(tri, aabb.halfExtents, a21) ||
        !intersectAABBTriangleSAT(tri, aabb.halfExtents, a22) ||
        !intersectAABBTriangleSAT(tri, aabb.halfExtents, vec3(1.0, 0.0, 0.0)) ||
        !intersectAABBTriangleSAT(tri, aabb.halfExtents, vec3(0.0, 1.0, 0.0)) ||
        !intersectAABBTriangleSAT(tri, aabb.halfExtents, vec3(0.0, 0.0, 1.0)) ||
        !intersectAABBTriangleSAT(tri, aabb.halfExtents, triangleNormal)
    ) {
        return false;
    }

    return true;
}
