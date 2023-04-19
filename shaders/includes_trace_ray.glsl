float distanceToObject(vec3 origin, vec3 dir, Bounds bnd) {
    vec3 v = bnd.position - origin;
    vec2 m = v * mat2x3(dir, v); // two dot products calculated using one matrix multiplication
    float d = (m.x * m.x - m.y) + bnd.radiusSquared;
    return d < 0.0 ? 0.0 : m.x - sqrt(d);
}

bool hitsBounds(vec3 origin, vec3 dir, Bounds bnd) {
    vec3 v = bnd.position - origin;
    vec2 m = v * mat2x3(dir, v); // two dot products calculated using one matrix multiplication
    return m.y < bnd.radiusSquared || (((-m.x) * m.x + m.y) < bnd.radiusSquared && m.x > EPSILON);
}

RayResult traceRayWithBVH(vec3 origin, vec3 dir) {
    RayResult result = RayResult(FLT_MAX, 0, 0);

    uint currIdx = bvh.root;

    while (currIdx != 0) {
        Bounds curr = bvh.nodes[currIdx];

        if (curr.material == 0) {
            currIdx = hitsBounds(origin, dir, curr) ? curr.child : curr.next;
            continue;
        }

        float dist = distanceToObject(origin, dir, curr);

        if (dist > EPSILON && dist < result.distanceToHit) {
            // is a leaf, store data
            result = RayResult(dist, currIdx, curr.material);
        }

        // move to next node
        currIdx = curr.next;
    }

    return result;
}

float[4] distanceToObject4(vec3 origin, mat4x3 dirs, Bounds bnd) {
    vec3 v = bnd.position - origin;
    vec4 m = v * dirs; // dot products calculated using one matrix multiplication
    vec4 d = m * m + (bnd.radiusSquared - dot(v, v));
    return float[4](
        d.x < 0.0 ? 0.0 : m.x - sqrt(d.x),
        d.y < 0.0 ? 0.0 : m.y - sqrt(d.y),
        d.z < 0.0 ? 0.0 : m.z - sqrt(d.z),
        d.w < 0.0 ? 0.0 : m.w - sqrt(d.w)
    );
}

bool hitsBounds4(vec3 origin, mat4x3 dirs, Bounds bnd) {
    vec3 v = bnd.position - origin;
    float a = dot(v, v);
    vec4 m = v * dirs; // dot products calculated using one matrix multiplication
    uvec4 cond1 = uvec4(lessThan((-m) * m, vec4(bnd.radiusSquared - a)));
    uvec4 cond2 = uvec4(greaterThan(m, vec4(EPSILON)));
    return a < bnd.radiusSquared || any(bvec4(cond1 & cond2));
}

RayResult[4] traceRayWithBVH4(vec3 origin, mat4x3 dirs) {
    RayResult results[4] = RayResult[4](
        RayResult(FLT_MAX, 0, 0),
        RayResult(FLT_MAX, 0, 0),
        RayResult(FLT_MAX, 0, 0),
        RayResult(FLT_MAX, 0, 0)
    );

    uint currIdx = bvh.root;

    while (currIdx != 0) {
        Bounds curr = bvh.nodes[currIdx];

        if (curr.material == 0) {
            currIdx = hitsBounds4(origin, dirs, curr) ? curr.child : curr.next;
            continue;
        }

        float[4] dists = distanceToObject4(origin, dirs, curr);

        for (uint i = 0; i < 4; i++) {
            if (dists[i] > EPSILON && dists[i] < results[i].distanceToHit) {
                // is a leaf, store data
                results[i] = RayResult(dists[i], currIdx, curr.material);
            }
        }

        // move to next node
        currIdx = curr.next;
    }

    return results;
}
