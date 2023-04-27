bool marchRay(inout vec3 pos, vec3 dir, vec3 lmOrigin, float threshold, inout float totalDist) {
    float dist = totalDist;
    for (uint i = 0; i < 100; i++) {
        pos += dir * dist;
        int lmLayer = lmLayerAtPos(pos, lmOrigin); // TODO: improve
        vec3 idx = (pos - lmOrigin) * ((1.0 / float(LM_SIZE)) / LM_UNIT_SIZES[lmLayer]) + 0.5; // TODO: improve
        dist = texture(SDFImages[lmLayer], idx).x;
        totalDist += dist;

        if (lmLayer >= LM_COUNT) { // out of bounds
            return false;
        } else if (dist <= threshold * totalDist) { // hit
            return true;
        }
    }

    return false;
}
