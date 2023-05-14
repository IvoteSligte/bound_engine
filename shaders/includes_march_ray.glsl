bool marchRay(inout vec3 pos, vec3 dir, vec3 lmOrigin, float threshold, uint samples, inout float totalDist) {
    float dist = totalDist;

    float inv_threshold = 1.0 / threshold;
    vec3 dPos = dir * threshold;

    int lmLayer = lmLayerAtPos(pos, lmOrigin);

    for (uint i = 0; i < samples; i++) {
        pos += dir * dist;

        vec3 idx = pos - lmOrigin; // TODO: lmOrigin varying between layers
        bool increaseLayer = maximum(abs(idx)) > LAYER_COMPS[lmLayer];
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
