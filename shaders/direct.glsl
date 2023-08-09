#version 460

#include "includes_general.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(constant_id = 0) const float RATIO_X = 1.0;
layout(constant_id = 1) const float RATIO_Y = 1.0;

const vec2 RATIO = vec2(RATIO_X, RATIO_Y);

layout(binding = 0) uniform restrict readonly RealTimeBuffer {
    vec4 rotation;
    vec3 position;
    ivec3 lightmapOrigin; // TODO: different origin per layer
    ivec4 deltaLightmapOrigins[LM_LAYERS];
} rt;

layout(binding = 1, rgba16) uniform restrict writeonly image2D colorImage;

layout(binding = 2) uniform sampler3D SDFImages[LM_LAYERS];

layout(binding = 3, rgba16f) uniform image3D[LM_LAYERS * 4] radianceImages;

vec3[4] loadSHCoefs(ivec3 index, int layer) {
    vec3[4] coefs;
    for (int i = 0; i < 4; i++) {
        coefs[i] = imageLoad(radianceImages[i * LM_LAYERS + layer], index).rgb;
    }
    return coefs;
}

vec3 sampleRadiance(ivec4 index, vec3 dir) {
    vec3[4] coefs = loadSHCoefs(index.xyz, index.w);
    return evaluateRGBSphericalHarmonics(dir, coefs);
}

// TODO: movable radiance volume origin
vec3 sampleRadianceInterpolated(vec3 radIndexF, int layer, vec3 dir) {
    ivec3 V = ivec3(radIndexF); // corner 1
    ivec3 W = min(V + 1, ivec3(RADIANCE_SIZE - 1)); // corner 2
    
    vec3 weights = radIndexF - vec3(V);

    vec3 c1 = mix(sampleRadiance(ivec4(V,               layer), dir),
                  sampleRadiance(ivec4(W.x,  V.yz,      layer), dir),
                  weights.x);
    vec3 c2 = mix(sampleRadiance(ivec4(V.x,  W.y,  V.z, layer), dir),
                  sampleRadiance(ivec4(W.xy, V.z,       layer), dir),
                  weights.x);
    vec3 c3 = mix(sampleRadiance(ivec4(V.xy, W.z,       layer), dir),
                  sampleRadiance(ivec4(W.x,  V.y,  W.z, layer), dir),
                  weights.x);
    vec3 c4 = mix(sampleRadiance(ivec4(V.x,  W.yz,      layer), dir),
                  sampleRadiance(ivec4(W,               layer), dir),
                  weights.x);

    c1 = mix(c1, c2, weights.y);
    c2 = mix(c3, c4, weights.y);

    return mix(c1, c2, weights.z);
}

const float NEAR_CLIPPING = 1.0;

// TODO: change to fragment shader
void main() {
    const ivec2 VIEWPORT = ivec2(imageSize(colorImage).xy);
    const ivec2 IPOS = ivec2(gl_GlobalInvocationID.xy);

    // maps FragCoord to xy range [-1.0, 1.0]
    const vec2 NORM_COORD = RATIO * (IPOS * 2.0 / VIEWPORT - 1.0);
    const vec3 DIRECTION = normalize(vec3(NORM_COORD.x, 1.0, NORM_COORD.y));

    vec3 position = rt.position;
    vec4 rotation = rt.rotation;
    vec3 lmOrigin = rt.lightmapOrigin;
    
    vec3 dir = rotateWithQuat(rotation, DIRECTION);
    float totalDist = NEAR_CLIPPING;
    bool isHit = marchRay(SDFImages, position, dir, lmOrigin, 1e-3, 128, totalDist);

    if (!isHit) {
        vec3 color = vec3(dir.z * 0.5 + 0.5);
        color = pow(color, vec3(5.0)) * 0.1;
        imageStore(colorImage, IPOS, vec4(color, 0.0));
        return;
    }

    int radLayer;
    vec3 radIndexF = radIndexAtPosF(position, radLayer);
    vec3 color = sampleRadianceInterpolated(radIndexF, radLayer, -dir);
    
    // TODO: hit direction reprojection (calculate outgoing in direct.glsl so that the direction towards the SH probes can be used instead)

    imageStore(colorImage, IPOS, vec4(color, 0.0));
}
