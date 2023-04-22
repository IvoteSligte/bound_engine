bool marchRay(inout vec3 pos, vec3 dir, vec3 lmOrigin) {
    const float THRESHOLD = 0.0;

    float dist;
    for (uint i = 0; i < 1000; i++) {
        int lmLayer = lmLayerAtPos(pos, lmOrigin);
        vec3 idx = pos * (1.0 / float(LM_SIZE)) / LM_UNIT_SIZES[lmLayer] + 0.5;
        dist = texture(SDFImages[lmLayer], idx).x;
        pos += dir * dist;

        if (lmLayer >= LM_COUNT) { // out of bounds
            return false;
        } else if (dist <= THRESHOLD) { // hit
            return true;
        }
    }

    return false;
}
