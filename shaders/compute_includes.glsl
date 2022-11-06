const uint MAX_OBJECTS = 16;
const uint MAX_MATERIALS = 8;

struct Material {
    vec3 reflectance;
    vec3 emittance;
};

struct Object {
    vec3 pos;
    float radiusSquared;
    uint mat;
};

float luminanceFromRGB(vec3 rgb) {
    return dot(rgb, vec3(0.2126, 0.7152, 0.0722));
}

float computeWeight(vec4 normalDepth, vec4 cNormalDepth, float luminance, float cLuminance, uint material, uint cMaterial) {
    float weightDepth = abs(normalDepth.w - cNormalDepth.w);
    float weightNormal = pow(max(dot(normalDepth.xyz, cNormalDepth.xyz), 0.0), 128.0); // to a power, maybe
    float weightLuminance = abs(luminance - cLuminance);

    return exp(-weightDepth - max(weightLuminance, 0.0)) * weightNormal * float(material == cMaterial);
}
