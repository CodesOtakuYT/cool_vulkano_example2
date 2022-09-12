#version 460

layout(location = 0) in vec2 position;
layout(location = 1) in vec4 color;
layout(location = 2) in vec2 uv;

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec2 tex_coords;

layout(push_constant) uniform PushConstantData {
    float time;
    float x;
    float y;
} pc;

void main() {
    out_color = color;
    gl_Position = vec4(position, 0.0, 1.0);
    tex_coords = uv/2.0 - vec2(0.5);
}