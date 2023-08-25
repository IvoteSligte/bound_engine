#version 460

layout(location = 0) in vec4 position;

layout(location = 0) out vec3 worldPosition;

layout(binding = 0) uniform restrict readonly RealTimeBuffer {
    vec4 rotation;
    vec3 position;
    ivec3 lightmapOrigin;
    ivec4 deltaLightmapOrigins[LM_LAYERS];
    vec2 screenSize;
    float fov;
    mat4 projection_view;
} rt;

void main() {
    gl_Position = rt.projection_view * vec4(position.xyz, 1.0);
    worldPosition = position.xyz;
}
