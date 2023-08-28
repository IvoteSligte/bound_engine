#version 460

layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 normal;

layout(binding = 0) uniform restrict readonly RealTimeBuffer {
    mat4 projection_view;
    vec3 position;
} rt;

layout(binding = 1) buffer restrict readonly VertexBuffer {
    vec4 vertices[];
} vertexBuffer;

layout(binding = 2) buffer restrict readonly VertexIndexBuffer {
    uint indices[];
} vertexIndexBuffer;

vec3 calculateNormal() {
    uint i = gl_VertexIndex / 3 * 3; // rounds down to the nearest multiple of 3

    vec3 a = vertexBuffer.vertices[vertexIndexBuffer.indices[i    ]].xyz;
    vec3 b = vertexBuffer.vertices[vertexIndexBuffer.indices[i + 1]].xyz;
    vec3 c = vertexBuffer.vertices[vertexIndexBuffer.indices[i + 2]].xyz;

    vec3 ab = normalize(b - a);
    vec3 bc = normalize(c - b);

    return normalize(cross(ab, bc));
}

void main() {
    vec3 position = vertexBuffer.vertices[vertexIndexBuffer.indices[gl_VertexIndex]].xyz;
    gl_Position = rt.projection_view * vec4(position, 1.0);
    worldPosition = position;
    normal = calculateNormal();
}
