// float distanceToObject(Ray ray, Bounds bnd, out bool isInside) {
//     vec3 v = bnd.position - ray.origin;
//     vec2 m = v * mat2x3(ray.direction, v); // two dot products calculated using one matrix multiplication
//     isInside = m.y < bnd.radiusSquared;
//     float d = (m.x * m.x - m.y) + bnd.radiusSquared;
//     return d < 0.0 ? 0.0 : m.x - sqrt(d);
// }

bool hitsObject(Ray ray, Bounds bnd, out bool isInside, out vec2 m) {
    vec3 v = bnd.position - ray.origin;
    m = v * mat2x3(ray.direction, v); // two dot products calculated using one matrix multiplication
    isInside = m.y < bnd.radiusSquared;
    return ((-m.x) * m.x + m.y) < bnd.radiusSquared && m.x > 0.0;
}

void traceRayWithBVH(inout Ray ray, out vec3 hitObjPosition) {
    ray.objectHit = 0;
    hitObjPosition = vec3(0.0);
    float distanceToHit = 1e20;
    uint nodeHit = 0;

    uint currIdx = bvh.root;

    while (currIdx != 0) {
        Bounds curr = bvh.nodes[currIdx];

        bool isInside;
        vec2 m;
        bool isHit = hitsObject(ray, curr, isInside, m);

        // not a leaf, move to child
        if (curr.leaf == 0 && (isInside || isHit)) {
            currIdx = curr.child;
            continue;
        }

        // is a leaf and is hit
        if (isHit) {
            float dist = m.x - sqrt((m.x * m.x - m.y) + curr.radiusSquared);
            if (dist < distanceToHit) {
                // is a leaf, store data
                distanceToHit = dist;
                nodeHit = currIdx;
                hitObjPosition = curr.position;
                ray.objectHit = curr.leaf;
            }
        }

        // move to next node
        currIdx = curr.next;
    }

    ray.origin = (ray.direction * distanceToHit) + ray.origin;
}