float distanceToObject(Ray ray, Bounds bnd) {
    vec3 v = bnd.position - ray.origin;
    vec2 m = v * mat2x3(ray.direction, v); // two dot products calculated using one matrix multiplication
    float d = (m.x * m.x - m.y) + bnd.radiusSquared;
    return d < 0.0 ? 0.0 : m.x - sqrt(d);
}

bool hitsBounds(Ray ray, Bounds bnd) {
    vec3 v = bnd.position - ray.origin;
    vec2 m = v * mat2x3(ray.direction, v); // two dot products calculated using one matrix multiplication
    return m.y < bnd.radiusSquared || (((-m.x) * m.x + m.y) < bnd.radiusSquared && m.x > 1e-5);
}

void traceRayWithBVH(inout Ray ray) {
    ray.objectHit = 0;
    ray.materialHit = 0;
    float distanceToHit = 1e20;
    uint nodeHit = 0;

    uint currIdx = bvh.root;

    while (currIdx != 0) {
        Bounds curr = bvh.nodes[currIdx];

        if (curr.material == 0) {
            currIdx = hitsBounds(ray, curr) ? curr.child : curr.next;
            continue;
        }

        float dist = distanceToObject(ray, curr);

        if (dist > 1e-5 && dist < distanceToHit) {
            // is a leaf, store data
            distanceToHit = dist;
            nodeHit = currIdx;
            ray.objectHit = currIdx;
            ray.materialHit = curr.material;
        }

        // move to next node
        currIdx = curr.next;
    }

    ray.origin = (ray.direction * distanceToHit) + ray.origin;
}