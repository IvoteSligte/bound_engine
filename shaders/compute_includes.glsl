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
    return min(dot(rgb, vec3(0.2126, 0.7152, 0.0722)), 1.0);
}
