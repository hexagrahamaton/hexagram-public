#version 300 es
#ifdef GL_ES
precision mediump float;
#endif

in vec2 position;
void main() {
  gl_Position = vec4(position.xy, 0, 1);
}
