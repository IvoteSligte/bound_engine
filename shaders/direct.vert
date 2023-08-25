#version 460

layout(location = 0) in vec4 position;

layout(location = 0) out vec3 worldPosition;

layout(binding = 0) uniform restrict readonly RealTimeBuffer {
    mat4 projection_view;
    vec3 position;
} rt;

void main() {
    gl_Position = rt.projection_view * vec4(position.xyz, 1.0);
    worldPosition = position.xyz;
}
