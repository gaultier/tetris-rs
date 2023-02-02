#version 450

layout(location = 0) in vec2 in_color;
layout(location = 0) out vec4 f_color;

void main() {
    f_color = vec4((in_color+vec2(1.0, 1.0))*vec2(0.5, 0.5), 0.0, 1.0);
}
