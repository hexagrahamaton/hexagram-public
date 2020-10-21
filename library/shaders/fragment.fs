void main()
{
  float q = 1. / size.x;
  float time = mod(counter, duration) / duration;
  vec2 uv = gl_FragCoord.xy / size;
  vec4 c = black;
  c += vec4(uv, 1, 1);
  gl_FragColor = c;
}
