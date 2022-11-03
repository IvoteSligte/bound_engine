const uint MAX_OBJECTS = 8;
const uint MAX_MATERIALS = 4;

struct Material {
    vec3 reflectance;
    vec3 emittance;
};

struct Object {
    vec3 pos;
    float radiusSquared;
    uint mat;
};
