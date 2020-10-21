bool isnan(float n) {
  return !(n <= 0. || 0. <= n);
}

vec4 rgb2hsv(vec4 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec4(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x, 1.);
}

vec4 hsv2rgb(vec4 c)
{
    vec4 K = vec4(1., 2. / 3., 1. / 3., 3.);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return vec4(c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y), 1.0);
}

vec4 hexbin(vec2 uv, float rad) {
  uv = uv * 2. - 1.;
  float res = rad / 3.;

  vec2 baseUv = uv * baseRes;
  uv *= res;

  vec2 r = vec2(1., 1. / sqrt(3.));
  r = vec2(r.y, r.x);
  vec2 h = r * 0.5;
  
  vec2 a = mod(uv, r) - h;
  vec2 b = mod(uv - h, r) - h;

  float delta = length(a) - length(b);
  vec2 gv;
  if (delta < 0.) {
    gv = a;
  }
  else {
    gv = b;
  }
  // vec2 gv = length(a) < length(b) ? a : b;
  float hextant = mod(floor(mod(atan(gv.x, gv.y) + tau, tau) / tau * 6.) + 5., 6.) / 6.;

  a = mod(baseUv, r) - h;
  b = mod(baseUv - h, r) - h;
  vec2 coord = length(a) < length(b) ? a : b;
  coord = ((uv - gv) / res + 1.) / 2.;
  gv = gv * 1.5 + 0.5;
  return vec4(gv, coord);
}

vec2 scaleUv(vec2 uv, float fac) {
  return uv * fac + (1. - fac) / 2.;
}

vec2 rotUv(vec2 uv, float angle) {
  vec2 cart = uv * 2. - 1.;
  float dir = atan(cart.y, cart.x);
  float mag = pow(cart.x * cart.x + cart.y * cart.y, 0.5);
  dir += angle;
  float x = cos(dir);
  float y = sin(dir);
  cart = vec2(x, y) * mag;
  return (cart + 1.) / 2.;
}

vec2 uv2p(vec2 uv) {
  vec2 cart = uv * 2. - 1.;
  float theta = atan(cart.y, cart.x);
  float mag = pow(cart.x * cart.x + cart.y * cart.y, 0.5);
  return vec2(mag, theta);
}

vec2 p2uv(vec2 p) {
  float x = cos(p.y);
  float y = sin(p.y);
  vec2 cart = vec2(x, y) * p.x;
  return (cart + 1.) / 2.;
}

vec3 uv2hex(vec2 c) {
  c = c * 2. - 1.;
  vec3 hex;
  hex.x =  c.x * 2. / sr3;
  hex.y = c.y - c.x * 1. / sr3;
  //-(sr3 * c.y + sr3 / 2. * c.x);

  hex.z = -hex.x - hex.y;
  return hex;
}

vec2 hex2uv(vec3 hex) {
  vec2 c;
  c.x = hex.x * sr3 / 2.;
  c.y = hex.y + hex.x * 0.5;
  return c / 2. + 0.5;
}

vec4 alphaMult(vec4 a, vec4 b) {
  return b * b.a + a * (1. - b.a);
}

vec3 uv2hex(vec2 c) {
  c = c * 2. - 1.;
  vec3 hex;
  hex.x =  c.x * 2. / sr3;
  hex.y = c.y - c.x * 1. / sr3;
  //-(sr3 * c.y + sr3 / 2. * c.x);

  hex.z = -hex.x - hex.y;
  return hex;
}

vec2 hex2uv(vec3 hex) {
  vec2 c;
  c.x = hex.x * sr3 / 2.;
  c.y = hex.y + hex.x * 0.5;
  return c / 2. + 0.5;
}
