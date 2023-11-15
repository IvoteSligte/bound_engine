#version 460

#include "includes_general.glsl"

// position of the fragment in world space
layout(location = 0) in vec3 fragPosition;
// layout(location = 1) in vec3 fragNormal;

layout(location = 0) out vec3 fragColor;

layout(binding = 0) uniform restrict readonly RealTimeBuffer {
    mat4 projection_view;
    vec3 position;
} rt;

layout(binding = 3) buffer Grid {
    GridCell cells[CELLS][CELLS][CELLS];
} grid[3];

float getCellColor(GridCell cell) {
    // length(vec3(0.0)) = undefined
    if (cell.vector == vec3(0.0)) {
        return 0.0;
    }
    return length(cell.vector) / float(cell.counter);
}

void main() {
    vec3 direction = normalize(fragPosition - rt.position);
    // TODO: changeable cell size (fixed at 1.0 currently)
    ivec3 index = ivec3(fragPosition + direction * EPSILON) + (CELLS / 2);
    
    // out of bounds
    if (any(greaterThanEqual(abs(index), ivec3(CELLS)))) {
        fragColor = vec3(0.0);
        return;
    }
    for (int i = 0; i < 3; i++) {
        GridCell cell = grid[i].cells[index.x][index.y][index.z];
        fragColor[i] = getCellColor(cell);
    }
}
