#version 460

const vec2[3] VERTICES = vec2[](
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0)
);

void main() {
    gl_Position = vec4(VERTICES[gl_VertexIndex], 0.0, 1.0);
}