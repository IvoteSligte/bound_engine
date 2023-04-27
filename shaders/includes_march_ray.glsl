bool marchRay(inout vec3 pos, vec3 dir, vec3 lmOrigin, float threshold, inout float totalDist) {
    float dist = totalDist;

    int lmLayer = lmLayerAtPos(pos, lmOrigin);
    float mult = (1.0 / float(LM_SIZE)) / LM_UNIT_SIZES[lmLayer];
    vec3 idx = (pos - lmOrigin) * mult;

    for (uint i = 0; i < 100; i++) {
        pos += dir * dist;

        idx += (dir * dist) * mult;

        if (any(greaterThan(abs(idx), vec3(0.5)))) {
            idx *= 0.5;
            mult *= 0.5;
            lmLayer += 1;
        } else if (lmLayer > 0 && all(lessThan(abs(idx), vec3(0.25)))) {
            idx *= 2.0;
            mult *= 2.0;
            lmLayer -= 1;
        }

        dist = texture(SDFImages[lmLayer], clamp(idx + 0.5, 0.0, 1.0)).x;
        totalDist += dist;

        if (lmLayer >= LM_COUNT) { // out of bounds
            return false;
        } else if (dist <= threshold * totalDist) { // hit
            return true;
        }
    }

    return false;
}
