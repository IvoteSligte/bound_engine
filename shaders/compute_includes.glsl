const uint MAX_OBJECTS = 16;
const uint MAX_MATERIALS = 16;

struct Material {
    vec3 reflectance;
    vec3 emittance;
};

// struct Object {
//     uint material;
//     vec3 velocity;
//     vec4 rotation;
// };

float luminanceFromRGB(vec3 rgb) {
    return clamp(dot(rgb, vec3(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
}
