// TODO: try applying Newton-Raphson iteration
bool marchRay(inout vec3 pos, vec3 dir, vec3 lmOrigin, float threshold, uint samples, inout float totalDist) {
    float dist = totalDist;

    float inv_threshold = 1.0 / threshold;
    vec3 dPos = dir * threshold;

    int lmLayer = lmLayerAtPos(pos, lmOrigin);

    for (uint i = 0; i < samples; i++) {
        pos += dir * dist;

        vec3 idx = pos - lmOrigin; // TODO: lmOrigin varying between layers
        bool increaseLayer = any(greaterThan(abs(idx), vec3(float(LM_SIZE) * LM_UNIT_SIZES[lmLayer] * 0.5)));
        bool decreaseLayer = lmLayer > 0 && all(lessThan(abs(idx), vec3(float(LM_SIZE) * LM_UNIT_SIZES[lmLayer] * 0.25)));
        lmLayer += int(increaseLayer) - int(decreaseLayer); // one of: -1, 0, 1

        float mult = (1.0 / float(LM_SIZE)) / LM_UNIT_SIZES[lmLayer]; // TODO: consts

        vec3 texIdx = (pos - lmOrigin) * mult + 0.5; // TODO: lmOrigin varying between layers
        dist = texture(SDFImages[lmLayer], clamp(texIdx, 0.0, 1.0)).x;
        totalDist += dist;

        if (dist <= threshold * totalDist) { // hit or out of bounds
            return lmLayer < LM_COUNT;
        }
    }

    return false;
}
