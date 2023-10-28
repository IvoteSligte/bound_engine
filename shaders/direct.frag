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

layout(binding = 2) buffer Grid {
    GridCell cells[CELLS][CELLS][CELLS];
} grid;

void main() {
    // vec3 direction = normalize(fragPosition - rt.position);
    // TODO: changeable cell size (fixed at 1.0 currently)
    ivec3 index = ivec3(fragPosition) + (CELLS / 2);
    
    // out of bounds
    if (any(greaterThanEqual(abs(index), ivec3(CELLS)))) {
        fragColor = vec3(0.0);
        return;
    }
    GridCell cell = grid.cells[index.x][index.y][index.z];

    // length(vec3(0.0)) = undefined
    if (cell.vector == vec3(0.0)) {
        fragColor = vec3(0.0);
        return;
    }
    float red = length(cell.vector) / float(cell.counter);
    fragColor = vec3(red); // DEBUG:
}
