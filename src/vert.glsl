#version 450

// Vertex positions
layout(location = 0) in vec2 position;
// Per-instance data
layout(location = 1) in vec2 position_offset;
layout(location = 2) in vec3 color;


layout(location = 0) out vec3 out_color;

layout(set = 0, binding = 0) uniform Data {
    /* mat4 world; */
    /* mat4 view; */
    /* mat4 proj; */
    vec2 tetramino_offset;
} uniforms;

void main() {
    /* mat4 worldview = uniforms.view * uniforms.world; */
    /* gl_Position = uniforms.proj * worldview * vec4(position + position_offset, 0.0, 1.0); */
    gl_Position = vec4(position + position_offset + uniforms.tetramino_offset, 0.0, 1.0);

    out_color = color;
}

