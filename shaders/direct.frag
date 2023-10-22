#version 460

#include "includes_general.glsl"

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;

layout(location = 0) out vec3 fragColor;

layout(binding = 0) uniform restrict readonly RealTimeBuffer {
    mat4 projection_view;
    vec3 position;
} rt;

layout(binding = 3) uniform sampler3D radianceTextures[LM_LAYERS * SH_CS];

vec3[SH_CS] loadSHCoefs(vec3 position, int layer, vec3 origin) {   
    vec3 texIndex = radTextureIndexAtPos(position, layer, origin);
    vec3[SH_CS] coefs;
    for (int i = 0; i < SH_CS; i++) {
        coefs[i] = texture(radianceTextures[layer * LM_LAYERS + i], texIndex).rgb;
    }
    return coefs;
}

vec3 sampleRadiance(vec3 position, vec3 dir) {
    vec3 origin = vec3(0.0); // TODO: movable origin
    
    int layer = radLayerAtPos(position, origin);
    if (layer >= LM_LAYERS) { return vec3(0.0); }
    vec3[SH_CS] coefs = loadSHCoefs(position, layer, origin);
    return evaluateRGBSphericalHarmonics(dir, coefs);
}

void main() {
    vec3 direction = normalize(fragPosition - rt.position);
    // normal direction is the most accurate (for diffuse lighting)
    fragColor = sampleRadiance(fragPosition - EPSILON * direction, -fragNormal);
}

// FIXME: looking straight down gives a black screen; sampling problem or rasterized rendering problem
