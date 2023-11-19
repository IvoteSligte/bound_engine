#version 460

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

#include "includes_general.glsl"

// cleared every frame
layout(binding = 0) writeonly buffer Grid {
    GridCell cells[CELLS][CELLS][CELLS];
} grid;

layout(binding = 1, r32f) uniform writeonly image3D energyGrid;

void main() {
    const ivec3 index = ivec3(gl_GlobalInvocationID);
    GridCell empty = GridCell(vec3(0.0), 0);
    grid.cells[index.x][index.y][index.z] = empty;
    imageStore(energyGrid, index, vec4(0.0));
}