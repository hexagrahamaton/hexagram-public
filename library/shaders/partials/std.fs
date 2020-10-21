#ifdef GL_ES
  precision highp float;
#endif
#version 300 es

uniform float counter;
uniform vec2 size;
uniform sampler2D lastFrame;
uniform sampler2D image;

float sr3 = sqrt(3.);

float pi = 3.151592653;
float tau = 6.2831853;

float sr2 = sqrt(2.);
float sr3 = sqrt(3.);

vec4 white = vec4(1., 1., 1., 1.);
vec4 black = vec4(0., 0., 0., 1.);
