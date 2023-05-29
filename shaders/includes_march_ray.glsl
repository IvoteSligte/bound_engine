bool marchRay(inout vec3 pos, vec3 dir, vec3 lmOrigin, float threshold, uint samples, inout float totalDist) {
    float dist = totalDist;

    vec3 dPos = dir * threshold;

    int lmLayer = lmLayerAtPos(pos, lmOrigin);

    for (uint i = 0; i < samples; i++) {
        pos += dir * dist;

        vec3 idx = pos - lmOrigin; // TODO: lmOrigin varying between layers
        bool outOfLayer = maximum(abs(idx)) > LAYER_COMPS[lmLayer];
        bool coneTooBig = threshold * totalDist > 0.5 * LM_UNIT_SIZES[lmLayer];
        bool increaseLayer = outOfLayer || coneTooBig;
        lmLayer = increaseLayer ? lmLayer + 1 : lmLayer;

        float mult = MULTS[lmLayer];

        vec3 texIdx = (pos - lmOrigin) * mult + 0.5; // TODO: lmOrigin varying between layers
        dist = texture(SDFImages[lmLayer], clamp(texIdx, 0.0, 1.0)).x;
        totalDist += dist;

        if (dist <= threshold * totalDist) { // hit or out of bounds
            return lmLayer < LM_COUNT;
        }
    }

    return false;
}
