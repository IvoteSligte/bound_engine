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

layout(binding = 2) uniform sampler3D sdfTextures[LM_LAYERS];

layout(binding = 3) uniform sampler3D radianceTextures[LM_LAYERS * 4];

vec3[4] loadSHCoefs(vec3 index, int layer) {   
    vec3 texIndex = radIndexTexture(index, layer);
    vec3[4] coefs;
    for (int i = 0; i < 4; i++) {
        coefs[i] = texture(radianceTextures[layer * LM_LAYERS + i], texIndex).rgb;
    }
    return coefs;
}

vec3 sampleRadiance(vec3 position, vec3 dir) {
    int layer;
    vec3 index = radIndexAtPosF(position, layer);
    if (layer >= LM_LAYERS) { return vec3(0.0); }
    vec3[4] coefs = loadSHCoefs(index, layer);
    return evaluateRGBSphericalHarmonics(dir, coefs);
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
    bool isHit = marchRay(sdfTextures, position, dir, lmOrigin, 1e-3, 128, totalDist);

    if (!isHit) {
        vec3 color = vec3(dir.z * 0.5 + 0.5);
        color = pow(color, vec3(5.0)) * 0.1;
        imageStore(colorImage, IPOS, vec4(color, 0.0));
        return;
    }

    vec3 color = sampleRadiance(position, -dir);
    // TODO: hit direction reprojection (calculate outgoing in direct.glsl so that the direction towards the SH probes can be used instead)

    imageStore(colorImage, IPOS, vec4(color, 0.0));
}
