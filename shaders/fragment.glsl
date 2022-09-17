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

        for (uint l = 0; l < buf.lightCount; l++) {
            Light light = buf.lights[l];

            float a = dot(ray.dir, normalize(light.pos - ray.origin)); // a > 0.0 if hit
            float b = distance(light.pos, ray.origin);
            float c = a * b;
            float d = sqrt(1.0 - b * b + c * c); // 0.5 * width at hitpoint
            float e = c - d; // distance to hitpoint

            if (a > 0.0 && e < dist) {
                float lightFallOff = e * UNIT_SIZE * e * UNIT_SIZE + 1.0;
                float absorbFallOff = buf.mats[ray.index].transmission / (e * UNIT_SIZE + 1.0);
                fragColor.xyz += ray.color * light.color * d / lightFallOff / absorbFallOff;
            }
        }
        
        if (index == MAX_OBJECTS) {
            continue;
        }


        vec3 realP = ray.dir * dist + ray.origin; // world-space point (ray endpoint)
        vec3 normal = normalize(realP - buf.objs[index].pos);

        ray.color *= buf.mats[ray.index].transmission / (dist * UNIT_SIZE + 1.0);
        // light scatters because it is not a perfect laser
        // TODO: add specular and mirror (currently diffuse)
        ray.color /= dist * UNIT_SIZE * dist * UNIT_SIZE + 1.0;

        // mirror reflection
        vec3 mirror = ray.color * buf.mats[index].mirror;
        // MAX function can be replaced with OR operator
        if (max(max(mirror.r, mirror.g), mirror.b) > BOUNCE_THRESHOLD && count < MAX_BOUNCES && ray.num + 1 < MAX_BOUNCES) {
            Ray newRay;
            newRay.color = mirror;
            newRay.origin = realP;
            newRay.dir = reflect(ray.dir, normal);
            newRay.index = buf.matCount - 1; // air
            newRay.num = ray.num + 1;
            rays[count] = newRay;
            count += 1;
        }

        // ray into object
        vec3 trans = ray.color;
        if (max(max(mirror.r, mirror.g), mirror.b) > BOUNCE_THRESHOLD && count < MAX_BOUNCES && ray.num + 1 < MAX_BOUNCES) {
            Ray newRay;
            newRay.color = trans;
            newRay.origin = realP;
            newRay.dir = refract(ray.dir, normal, buf.mats[ray.index].refIdx / buf.mats[index].refIdx);
            newRay.index = index;
            newRay.num = ray.num + 1;
            rays[count] = newRay;
            count += 1;
        }


        for (uint l = 0; l < buf.lightCount; l++) {
            Light light = buf.lights[l];

            vec3 dir = normalize(light.pos - realP); // not sure if normalize() is necessary

            float a = dot(ray.dir, dir); // a > 0.0 if hit
            float b = distance(light.pos, realP);
            float c = a * b;
            float d = sqrt(1.0 - b * b + c * c); // 0.5 * width at hitpoint
            float e = c - d; // distance to hitpoint

            float near = 1e20;

            for (uint i = 0; i < buf.objCount; i++) {
                Object object = buf.objs[i];

                float a = dot(dir, normalize(object.pos - realP)); // a > 0.0 if hit
                float b1 = distance(object.pos, realP);
                float c = a * b1;
                float d = sqrt(object.size * object.size - b1 * b1 + c * c); // 0.5 * width at hitpoint
                float e = c - d; // distance to hitpoint
                float f = sqrt(b1 * b1 - c * c); // min distance of ray to sphere center

                if (a > 0.0 && e < b) {
                    near = min(near, f - object.size);
                }
            }

            float diffuse = max(dot(normal, dir), 0.0);
            float specular = diffuse * max(pow(dot(-ray.dir, reflect(-dir, normal)), buf.mats[index].shine), 0.0);
            diffuse *= buf.mats[index].diffuse;
            specular *= buf.mats[index].specular;
            
            float shadow = clamp(near + 1.0, 0.0, 1.0) * clamp(near + 1.0, 0.0, 1.0); // shadow is non-linear

            float lightFallOff = b * UNIT_SIZE * b * UNIT_SIZE + 1.0;

            fragColor.rgb += ((diffuse + specular) * shadow + buf.mats[index].ambient) * ray.color * buf.mats[index].color * light.color / lightFallOff;
        }
    }
}