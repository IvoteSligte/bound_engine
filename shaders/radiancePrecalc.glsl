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

Material calculateMaterialIntersect(vec3 position, uint lmLayer) {
    Material combinedMat = Material(vec3(0.0), vec3(0.0));

    for (uint i = 0; i < MAX_OBJECTS; i++) {
        Object obj = objBuffer.objects[i];

        float dist = sdAABB(position, obj.position, obj.radius);

        if (dist < 0.71 * radUnitSizeLayer(lmLayer) + obj.radius) {
             // TODO: multiply by how much of the voxel is occupied by the object
            Material mat = mats.materials[obj.material];
            combinedMat.reflectance += mat.reflectance;
            combinedMat.emittance += mat.emittance;
        }
    }

    return combinedMat;
}

void main() {
    const int LAYER = int(gl_GlobalInvocationID.x / RADIANCE_SIZE);
    const ivec3 IIL = ivec3(gl_GlobalInvocationID.x % RADIANCE_SIZE, gl_GlobalInvocationID.yz);

    vec3 position = posAtRadIndex(ivec4(IIL, LAYER)); // TODO: movable radiance cache origin
    Material material = calculateMaterialIntersect(position, LAYER); // bottleneck // TODO: object acceleration structure

    cache.materials[LAYER][IIL.x][IIL.y][IIL.z] = material;
}
