#version 460

#include "includes_general.glsl"

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(binding = 0) uniform restrict readonly ObjectBuffer {
    Object objects[MAX_OBJECTS];
} objBuffer;

layout(binding = 1) uniform restrict readonly MaterialBuffer {
    Material materials[MAX_MATERIALS];
} mats;

layout(binding = 2) buffer writeonly RadianceBuffer {
    Radiance radiances[LM_LAYERS][RADIANCE_SIZE][RADIANCE_SIZE][RADIANCE_SIZE];
    Material materials[LM_LAYERS][RADIANCE_SIZE][RADIANCE_SIZE][RADIANCE_SIZE];
} cache;

uint calculateMaterialIntersect(vec3 position, uint lmLayer) {
    for (uint i = 0; i < MAX_OBJECTS; i++) {
        Object obj = objBuffer.objects[i];

        float dist = distance(position, obj.position);

        if (dist < obj.radius + 2.0 * radUnitSizeLayer(lmLayer)) {
            return obj.material;
        }
    }

    return 0;
}

void main() {
    const uint LAYER = gl_GlobalInvocationID.x / RADIANCE_SIZE;
    const uvec3 IIL = uvec3(gl_GlobalInvocationID.x % RADIANCE_SIZE, gl_GlobalInvocationID.yz);

    vec3 position = posAtRadIndex(uvec4(IIL, LAYER)); // TODO: movable radiance cache origin

    uint matIndex = calculateMaterialIntersect(position, LAYER); // bottleneck // TODO: object acceleration structure

    cache.materials[LAYER][IIL.x][IIL.y][IIL.z] = mats.materials[matIndex];
}
