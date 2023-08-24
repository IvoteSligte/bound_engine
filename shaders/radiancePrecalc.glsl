#version 460

#include "includes_general.glsl"

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(binding = 0) buffer restrict readonly VertexBuffer {
    vec4 vertices[];
} vertexBuffer;

layout(binding = 1) buffer restrict readonly VertexIndexBuffer {
    uint indices[];
} vertexIndexBuffer;

layout(binding = 2) buffer restrict readonly MaterialIndexBuffer {
    uint materials[];
} matIdxBuffer;

layout(binding = 3) uniform restrict readonly MaterialBuffer {
    Material materials[MAX_MATERIALS];
} matBuffer;

layout(binding = 4) buffer writeonly RadianceBuffer {
    PackedVoxel voxels[LM_LAYERS][RADIANCE_SIZE][RADIANCE_SIZE][RADIANCE_SIZE];
} cache;

Voxel calculateIntersect(vec3 position, int layer) {
    Voxel voxel = Voxel(vec3(0.0), vec3(0.0), vec4(0.0));
    float intersections = 0.0;
    
    float unit = radUnitSizeLayer(layer);
    AABB aabb = AABB(position, vec3(unit * 0.5 + EPSILON));

    for (uint i = 0; i < vertexIndexBuffer.indices.length(); i += 3) {
        vec3[3] tri;
        tri[0] = vertexBuffer.vertices[vertexIndexBuffer.indices[i    ]].xyz;
        tri[1] = vertexBuffer.vertices[vertexIndexBuffer.indices[i + 1]].xyz;
        tri[2] = vertexBuffer.vertices[vertexIndexBuffer.indices[i + 2]].xyz;

        // TODO: use the surface area of the triangle inside the voxel to determine the weight
        vec3 normal;
        bool intersects = intersectAABBTriangle(tri, aabb, normal);
        if (intersects) {
            Material mat = matBuffer.materials[matIdxBuffer.materials[i / 3]];
            voxel.emittance += mat.emittance;
            voxel.reflectance += mat.reflectance;
            voxel.normalSH += dirToCosineLobe(normal);
            intersections += 1.0;
        }
    }

    if (intersections > 1.0) {
        voxel.emittance /= intersections;
        voxel.reflectance /= intersections;
        voxel.normalSH /= intersections;
    }

    return voxel;
}

void main() {
    const int LAYER = int(gl_GlobalInvocationID.x / RADIANCE_SIZE);
    const ivec3 IIL = ivec3(gl_GlobalInvocationID.x % RADIANCE_SIZE, gl_GlobalInvocationID.yz);

    vec3 origin = vec3(0.0); // TODO: movable origin
    vec3 position = posAtRadIndex(IIL, LAYER, origin);

    Voxel voxel = calculateIntersect(position, LAYER); // bottleneck // TODO: object acceleration structure
    cache.voxels[LAYER][IIL.x][IIL.y][IIL.z] = packVoxel(voxel);
}
