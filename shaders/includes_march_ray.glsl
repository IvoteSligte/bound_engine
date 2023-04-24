bool marchRay(inout vec3 pos, vec3 dir, vec3 lmOrigin, float threshold) {
    float totalDist = 0.0;
    float dist;
    for (uint i = 0; i < 100; i++) {
        int lmLayer = lmLayerAtPos(pos, lmOrigin);
        vec3 idx = (pos - lmOrigin) * ((1.0 / float(LM_SIZE)) / LM_UNIT_SIZES[lmLayer]) + 0.5;
        dist = texture(SDFImages[lmLayer], idx).x;
        totalDist += dist;
        pos += dir * dist;

        if (lmLayer >= LM_COUNT) { // out of bounds
            return false;
        } else if (dist <= threshold * totalDist) { // hit
            return true;
        }
    }

    return false;
}
