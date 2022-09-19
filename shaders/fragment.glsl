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
    vec3 reflectance;
    float diffuse;
    float specular;
    float shine;
    float mirror;
    float transmission;
    float refIdx;
};

struct Object {
    vec3 pos;
    float size;
};

struct Ray {
    vec3 origin; // origin
    vec3 dir; // direction
    vec3 color; // color left after previous absorptions
    uint index; // object/material the ray is currently in
    uint depth;
};

struct Index {
    uint i;
};

layout(binding = 0) uniform readonly MutableData {
    uint matCount;
    uint objCount;
    uint lightCount;
    Material mats[MAX_MATERIALS];
    Object objs[MAX_OBJECTS];
    Index lights[MAX_LIGHTS];
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
// TODO: fix stuff
// TODO: make intersection function
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

        if (max(max(ray.color.r, ray.color.g), ray.color.b) <= BOUNCE_THRESHOLD || ray.depth + 1 >= MAX_BOUNCES) {
            continue;
        }

        uint index = MAX_OBJECTS;
        float dist = 1e20;

        for (uint i = 0; i < buf.objCount; i++) {
            Object object = buf.objs[i];

            float a = dot(ray.dir, normalize(object.pos - ray.origin)); // a > 0.0 if hit
            float b = distance(object.pos, ray.origin);
            float c = a * b;
            float d = sqrt(object.size * object.size - b * b + c * c); // 0.5 * width at hitpoint
            float e = c - d; // distance to hitpoint

            if (a > 0.0 && e < dist) {
                dist = e;
                index = i;
            }
        }
        
        if (index == MAX_OBJECTS) {
            continue;
        }


        Material material = buf.mats[index];
        vec3 hitpoint = ray.dir * dist + ray.origin;
        vec3 normal = normalize(hitpoint - buf.objs[index].pos);


        // base
        Ray newRay;
        newRay.origin = hitpoint;

        // mirror reflection
        newRay.color = ray.color * material.reflectance;
        newRay.dir = reflect(ray.dir, normal);
        newRay.index = buf.matCount - 1; // air
        newRay.depth = ray.depth + 1;
        rays[count] = newRay;
        count += 1;

        // ray into object
        newRay.color = ray.color * material.transmission;
        newRay.dir = refract(ray.dir, normal, buf.mats[ray.index].refIdx / buf.mats[index].refIdx);
        newRay.index = index;
        newRay.depth = ray.depth + 1;
        rays[count] = newRay;
        count += 1;


        for (uint l = 0; l < buf.lightCount; l++) {
            uint light = buf.lights[l].i;
            Object lightObject = buf.objs[light];
            Material lightMaterial = buf.mats[light];

            vec3 dir = normalize(lightObject.pos - hitpoint);
            float lightDist = distance(lightObject.pos, hitpoint);

            float near = 1e20;

            for (uint i = 0; i < buf.objCount; i++) {
                Object object = buf.objs[i];

                float a = dot(dir, normalize(object.pos - hitpoint)); // a > 0.0 if hit
                float b = distance(object.pos, hitpoint);
                float c = a * b;
                float d = sqrt(object.size * object.size - b * b + c * c); // 0.5 * width at hitpoint
                float e = c - d; // distance to hitpoint

                if (a > 0.0 && e < lightDist && i != light) {
                    near = min(near, sqrt(b * b - c * c) - object.size);
                }
            }

            // float diffuse = max(dot(normal, dir), 0.0);
            // float specular = diffuse * max(pow(dot(-ray.dir, reflect(-dir, normal)), material.shine), 0.0);
            // diffuse *= material.diffuse;
            // specular *= material.specular;
            
            float shadow = clamp(near + 1.0, 0.0, 1.0) * clamp(near + 1.0, 0.0, 1.0); // shadow is non-linear

            fragColor.rgb += shadow * ray.color * material.reflectance * lightMaterial.reflectance;
        }
    }
}