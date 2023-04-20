float distanceToObject(vec3 origin, vec3 dir, Object obj) {
    vec3 v = obj.position - origin;
    vec2 m = v * mat2x3(dir, v); // two dot products calculated using one matrix multiplication
    float d = (m.x * m.x - m.y) + obj.radiusSquared;
    return d < 0.0 ? 0.0 : m.x - sqrt(d);
}

bool hitsBounds(vec3 origin, vec3 dir, Object obj) {
    vec3 v = obj.position - origin;
    vec2 m = v * mat2x3(dir, v); // two dot products calculated using one matrix multiplication
    return m.y < obj.radiusSquared || (((-m.x) * m.x + m.y) < obj.radiusSquared && m.x > EPSILON);
}

RayResult traceRay(vec3 origin, vec3 dir) {
    RayResult result = RayResult(FLT_MAX, 0, 0);

    for (uint i = 0; i < MAX_OBJECTS; i++) {
        Object obj = objBuffer.objects[i]; // TODO: shared memory

        float dist = distanceToObject(origin, dir, obj);

        if (dist > EPSILON && dist < result.distanceToHit) {
            // is a leaf, store data
            result = RayResult(dist, i, obj.material);
        }
    }

    return result;
}

float[4] distanceToObject4(vec3 origin, mat4x3 dirs, Object obj) {
    vec3 v = obj.position - origin;
    vec4 m = v * dirs; // dot products calculated using one matrix multiplication
    vec4 d = m * m + (obj.radiusSquared - dot(v, v));
    return float[4](
        d.x < 0.0 ? 0.0 : m.x - sqrt(d.x),
        d.y < 0.0 ? 0.0 : m.y - sqrt(d.y),
        d.z < 0.0 ? 0.0 : m.z - sqrt(d.z),
        d.w < 0.0 ? 0.0 : m.w - sqrt(d.w)
    );
}

bool hitsBounds4(vec3 origin, mat4x3 dirs, Object obj) {
    vec3 v = obj.position - origin;
    float a = dot(v, v);
    vec4 m = v * dirs; // dot products calculated using one matrix multiplication
    uvec4 cond1 = uvec4(lessThan((-m) * m + a, vec4(obj.radiusSquared)));
    uvec4 cond2 = uvec4(greaterThan(m, vec4(EPSILON)));
    return a < obj.radiusSquared || any(bvec4(cond1 & cond2));
}
