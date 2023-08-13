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
    PackedVoxel voxels[LM_LAYERS][RADIANCE_SIZE][RADIANCE_SIZE][RADIANCE_SIZE];
} cache;

void calculateIntersect(vec3 position, uint layer, out Voxel voxel) {
    voxel = Voxel(vec3(0.0), vec3(0.0), vec3(0.0));
    
    float unit = (1.0 + EPSILON) * radUnitSizeLayer(layer);

    for (uint i = 0; i < MAX_OBJECTS; i++) {
        Object obj = objBuffer.objects[i];
        Material mat = mats.materials[obj.material];

        vec3 objMin = obj.position - obj.radius;
        vec3 objMax = obj.position + obj.radius;

        bool intersects = cubeCuboidIntersect(position, unit, objMin, objMax);
        if (intersects) {
            voxel.emittance += mat.emittance;
            voxel.reflectance += mat.reflectance;
            voxel.normal += normalizeZeroIfNaN(position - obj.position);
        }
    }
    if (voxel.normal != vec3(0.0)) {
        voxel.normal = normalize(voxel.normal);
    }
}

void main() {
    const int LAYER = int(gl_GlobalInvocationID.x / RADIANCE_SIZE);
    const ivec3 IIL = ivec3(gl_GlobalInvocationID.x % RADIANCE_SIZE, gl_GlobalInvocationID.yz);

    vec3 position = posAtRadIndex(IIL, LAYER); // TODO: movable radiance cache origin
    Voxel voxel;
    calculateIntersect(position, LAYER, voxel); // bottleneck // TODO: object acceleration structure

    cache.voxels[LAYER][IIL.x][IIL.y][IIL.z] = packVoxel(voxel);
}
