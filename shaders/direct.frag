#version 460

#include "includes_general.glsl"

layout(location = 0) out vec3 fragColor;

layout(binding = 0) uniform restrict readonly RealTimeBuffer {
    vec4 rotation;
    vec3 position;
    ivec3 lightmapOrigin; // TODO: different origin per layer
    ivec4 deltaLightmapOrigins[LM_LAYERS];
    vec2 screenSize;
    float fov;
} rt;

layout(binding = 1) uniform sampler3D sdfTextures[LM_LAYERS];

layout(binding = 2) uniform sampler3D radianceTextures[LM_LAYERS * SH_CS];

vec3[SH_CS] loadSHCoefs(vec3 index, int layer) {   
    vec3 texIndex = radIndexTexture(index, layer);
    vec3[SH_CS] coefs;
    for (int i = 0; i < SH_CS; i++) {
        coefs[i] = texture(radianceTextures[layer * LM_LAYERS + i], texIndex).rgb;
    }
    return coefs;
}

vec3 sampleRadiance(vec3 position, vec3 dir) {
    int layer;
    vec3 index = radIndexAtPosF(position, layer);
    if (layer >= LM_LAYERS) { return vec3(0.0); }
    vec3[SH_CS] coefs = loadSHCoefs(index, layer);
    return evaluateRGBSphericalHarmonics(dir, coefs);
}

const float NEAR_CLIPPING = 1.0;

// maps screen coords to [-1, 1] range
vec2 normalizedScreenCoords() {
    vec2 a = vec2(rt.screenSize.x / rt.screenSize.y, -1.0);
    vec2 n = 2.0 * gl_FragCoord.xy / rt.screenSize - 1.0;
    return a * rt.fov * n;
}

void main() {
    // maps FragCoord to xy range [-1.0, 1.0]
    const vec2 NORM_COORD = normalizedScreenCoords();
    const vec3 DIRECTION = normalize(vec3(NORM_COORD.x, 1.0, NORM_COORD.y));

    vec3 position = rt.position;
    vec4 rotation = rt.rotation;
    vec3 lmOrigin = rt.lightmapOrigin;
    
    vec3 dir = rotateWithQuat(rotation, DIRECTION);
    float totalDist = NEAR_CLIPPING;
    bool isHit = marchRay(sdfTextures, position, dir, lmOrigin, 1e-3, 128, totalDist);

    if (!isHit) {
        vec3 color = vec3(dir.z * 0.5 + 0.5);
        fragColor = pow(color, vec3(5.0)) * 0.1;
        return;
    }

    fragColor = sampleRadiance(position, dir);
}
