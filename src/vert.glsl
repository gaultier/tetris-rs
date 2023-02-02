#version 450

// Vertex positions
layout(location = 0) in vec2 position;
// Per-instance data
layout(location = 1) in vec2 position_offset;
layout(location = 2) in vec3 color;


layout(location = 0) out vec3 out_color;

layout(set = 0, binding = 0) uniform Data {
    float scale;
} uniforms;

void main() {
    gl_Position =  vec4(uniforms.scale * (position + position_offset), 0.0, 1.0);

    out_color = color;
}

