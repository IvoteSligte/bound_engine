bool marchRay(inout vec3 pos, vec3 dir, vec3 lmOrigin) {
    const float THRESHOLD = 1e-4;

    float dist;
    for (uint i = 0; i < 1000; i++) {
        int lmLayer = lightmapLayerAtPos(pos, lmOrigin);
        dist = texture(SDFImages[lmLayer], pos * (1.0 / 256.0) * LM_UNIT_SIZES[lmLayer] + 0.5).x;
        pos += dir * dist;

        if (lmLayer >= LM_COUNT) { // out of bounds
            return false;
        } else if (dist <= THRESHOLD) { // hit
            return true;
        }
    }

    return false;
}
