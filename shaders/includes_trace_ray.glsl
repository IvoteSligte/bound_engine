void traceRayWithBVH(inout Ray ray, out vec3 hitObjPosition) {
    ray.objectHit = 0;
    hitObjPosition = vec3(0.0);
    float distanceToHit = 1e20;
    uint nodeHit = 0;

    uint curr_idx = bvh.root;

    while (curr_idx != 0) {
        Bounds curr = bvh.nodes[curr_idx];

        bool is_inside;
        float d = distanceToObject(ray, curr, is_inside);
        bool is_hit = d > 0.0 && d < distanceToHit;

        // not a leaf, move to child
        if (curr.leaf == 0 && (is_inside || is_hit)) {
            curr_idx = curr.child;
            continue;
        }

        if (is_hit) {
            // is a leaf, store data
            distanceToHit = d;
            nodeHit = curr_idx;
            hitObjPosition = curr.position;
            ray.objectHit = curr.leaf;
        }

        // move to next node
        curr_idx = curr.next;
    }

    ray.origin = (ray.direction * distanceToHit) + ray.origin;
}