float distanceToObject(Ray ray, Bounds bnd) {
    vec3 v = bnd.position - ray.origin;
    vec2 m = v * mat2x3(ray.direction, v); // two dot products calculated using one matrix multiplication
    float d = (m.x * m.x - m.y) + bnd.radiusSquared;
    return d < 0.0 ? 0.0 : m.x - sqrt(d);
}

bool hitsBounds(Ray ray, Bounds bnd) {
    vec3 v = bnd.position - ray.origin;
    vec2 m = v * mat2x3(ray.direction, v); // two dot products calculated using one matrix multiplication
    return m.y < bnd.radiusSquared || (((-m.x) * m.x + m.y) < bnd.radiusSquared && m.x > EPSILON);
}

float traceRayWithBVH(inout Ray ray) {
    ray.objectHit = 0;
    ray.materialHit = 0;
    float distanceToHit = FLT_MAX;

    uint currIdx = bvh.root;

    while (currIdx != 0) {
        Bounds curr = bvh.nodes[currIdx];

        if (curr.material == 0) {
            currIdx = hitsBounds(ray, curr) ? curr.child : curr.next;
            continue;
        }

        float dist = distanceToObject(ray, curr);

        if (dist > EPSILON && dist < distanceToHit) {
            // is a leaf, store data
            distanceToHit = dist;
            ray.objectHit = currIdx;
            ray.materialHit = curr.material;
        }

        // move to next node
        currIdx = curr.next;
    }

    return distanceToHit;
}

vec2 distanceToObject2(vec3 origin, vec3 dir1, vec3 dir2, Bounds bnd) {
    vec3 v = bnd.position - origin;
    vec3 m = v * mat3x3(v, dir1, dir2); // three dot products calculated using one matrix multiplication
    vec2 d = (m.yz * m.yz - m.x) + bnd.radiusSquared;
    return vec2(d.x < 0.0 ? 0.0 : m.y - sqrt(d.x), d.y < 0.0 ? 0.0 : m.z - sqrt(d.y));
}

bool hitsBounds2(vec3 origin, vec3 dir1, vec3 dir2, Bounds bnd) {
    vec3 v = bnd.position - origin;
    vec3 m = v * mat3x3(v, dir1, dir2); // three dot products calculated using one matrix multiplication
    bvec2 cond1 = lessThan((-m.yz) * m.yz, vec2(bnd.radiusSquared - m.x));
    bvec2 cond2 = greaterThan(m.yz, vec2(EPSILON));
    return m.x < bnd.radiusSquared || (cond1.x && cond2.x) || (cond1.y && cond2.y);
}

vec2 traceRayWithBVH2(inout Ray ray1, inout Ray ray2) {
    ray1.objectHit = 0;
    ray1.materialHit = 0;
    ray2.objectHit = 0;
    ray2.materialHit = 0;
    vec2 distancesToHit = vec2(FLT_MAX);

    uint currIdx = bvh.root;

    while (currIdx != 0) {
        Bounds curr = bvh.nodes[currIdx];

        if (curr.material == 0) {
            currIdx = hitsBounds2(ray1.origin, ray1.direction, ray2.direction, curr) ? curr.child : curr.next;
            continue;
        }

        vec2 dists = distanceToObject2(ray1.origin, ray1.direction, ray2.direction, curr);

        if (dists.x > EPSILON && dists.x < distancesToHit.x) {
            // is a leaf, store data
            distancesToHit.x = dists.x;
            ray1.objectHit = currIdx;
            ray1.materialHit = curr.material;
        }

        if (dists.y > EPSILON && dists.y < distancesToHit.y) {
            // is a leaf, store data
            distancesToHit.y = dists.y;
            ray2.objectHit = currIdx;
            ray2.materialHit = curr.material;
        }

        // move to next node
        currIdx = curr.next;
    }

    return distancesToHit;
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

struct RayResult {
    float distanceToHit;
    uint objectHit;
    uint materialHit;
};

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
