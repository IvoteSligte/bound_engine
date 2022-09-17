#version 460

layout(location = 0) out vec4 fragColor;

layout(push_constant) uniform PushConstantData {
    vec4 rot;
    vec3 pos;
} pc;

const uint MAX_MATERIALS = 9; // default (air) is always included
const uint MAX_OBJECTS = 8;
const uint MAX_LIGHTS = 8;

const uint MAX_BOUNCES = 16;
const float BOUNCE_THRESHOLD = 0.1;

const float UNIT_SIZE = 0.1;

struct Material {
    vec3 color;
    float diffuse;
    float specular;
    float shine;
    float ambient;
    float mirror;
    float transmission;
    float refIdx;
};

struct Object {
    vec3 pos;
    float size;
};

struct Light {
    vec3 pos;
    vec3 color; // length(color) = strength
};

struct Ray {
    vec3 origin; // origin
    vec3 dir; // direction
    vec3 color; // color left after previous absorptions
    uint index; // object/material the ray is currently in
    uint num;
};

layout(binding = 0) uniform readonly MutableData {
    uint matCount;
    uint objCount;
    uint lightCount;
    Material mats[MAX_MATERIALS];
    Object objs[MAX_OBJECTS];
    Light lights[MAX_LIGHTS];
} buf;

layout(binding = 1) uniform readonly ConstantBuffer {
    vec2 view; // window size
    vec2 ratio; // window height / width
} cs;

vec3 rotate(vec4 q, vec3 v) {
    vec3 t = cross(q.xyz, v) + q.w * v;
    return v + 2.0 * cross(q.xyz, t);
}

// TODO: fix transmission
void main() {
    // maps FragCoord to xy range [-1.0, 1.0]
    vec2 normCoord = gl_FragCoord.xy * 2 / cs.view - 1.0;
    // maps normCoord to a different range (e.g. for FOV and non-square windows)
    normCoord *= cs.ratio;

    fragColor.xyz = vec3(0.0);
    vec3 viewDir = normalize(rotate(pc.rot, vec3(normCoord.x, 1.0, normCoord.y)));

    Ray rays[MAX_BOUNCES];
    rays[0] = Ray(pc.pos, viewDir, vec3(1.0), buf.matCount - 1, 0);
    uint count = 1;
    
    while (count > 0) {
        Ray ray = rays[count - 1];
        count -= 1;

        float dist = 1e20;

        for (uint i = 0; i < buf.objCount; i++) {
            Object object = buf.objs[i];

            float a = dot(ray.dir, normalize(object.pos - ray.origin)); // a > 0.0 if hit
            float b = distance(object.pos, ray.origin);
            float c = a * b;
            float d = sqrt(object.size * object.size - b * b + c * c); // 0.5 * width at hitpoint
            float e = c - d; // distance to hitpoint

            if (e < dist && a > 0.0) {
                fragColor.xyz = buf.mats[i].color * d / object.size;
                dist = e;
            }
        }
    }
}