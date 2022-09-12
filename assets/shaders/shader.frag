#version 460

layout(location = 0) out vec4 f_color;
layout(location = 0) in vec4 in_color;
layout(location = 1) in vec2 tex_coords;

layout(set = 0, binding = 0) uniform sampler2D tex;

void main() {
    f_color = texture(tex, tex_coords);
}