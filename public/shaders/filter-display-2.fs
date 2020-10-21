#version 300 es
#ifdef GL_ES
  precision highp float;
#endif

// FILTER DISPLAY 2

#define MAX_DIST 30.
#define SURF_DIST 0.0001
#define ITER 128

// #version 300 es
uniform float counter;
uniform vec2 size;
uniform sampler2D lastFrame;
uniform float duration;
uniform float time;

out vec4 fragColor;

float pi = 3.141592653;
float tau = 6.2831853;

uniform sampler2D bufferImage;
uniform sampler2D image1;
uniform sampler2D image2;
uniform sampler2D image3;
// vec4 bg = vec4(0.812, 0.812, 0.644, 1.);
vec4 bg = vec4(0., 0., 0., 1.);

float baseRes = 1./3.;
float radialRes = 1.5;

float sr3 = sqrt(3.);
float sr2 = sqrt(2.);

vec4 white = vec4(1.,1.,1.,1.);
vec4 black = vec4(0.,0.,0.,1.);

// in vec4 gl_FragCoord;
// out vec4 gl_FragColor;

bool isnan(float n) {
  return !(n <= 0. || 0. <= n);
}

vec2 cmul(vec2 a, vec2 b) {
  return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

vec2 clog(vec2 z) {
  return vec2(
    log(length(z)),
    atan(z.y, z.x)
  );
}

vec2 cpow (vec2 a, vec2 b) {
  float aarg = atan(a.y, a.x);
  float amod = length(a);

  float theta = log(amod) * b.y + aarg * b.x;

  return vec2(
    cos(theta),
    sin(theta)
  ) * pow(amod, b.x) * exp(-aarg * b.y);
}

vec4 rgb2hsv(vec4 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec4(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x, c.w);
}

vec4 hsv2rgb(vec4 c)
{
    vec4 K = vec4(1., 2. / 3., 1. / 3., 3.);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return vec4(c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y), c.w);
}

vec4 hexbin(vec2 uv, float rad) {
  float res = rad / 3.;
  vec2 baseUv = uv;
  uv *= res;

  vec2 r = vec2(1., 1. / sr3);
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

  a = mod(baseUv, r) - h;
  b = mod(baseUv - h, r) - h;
  vec2 coord = length(a) < length(b) ? a : b;
  coord = (uv - gv) / res;
  gv *= 3.;
  return vec4(gv, coord);
}

vec2 scaleUv(vec2 uv, float fac) {
  return uv * fac + (1. - fac) / 2.;
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

vec3 cart2hex(vec2 c, float s) {
  vec3 hex;
  hex.x =  c.x * 2. / sr3 * s;
  hex.y = (c.y - c.x * 1. / sr3) * s;
  hex.z = -hex.x - hex.y;
  return hex;
}

vec2 hex2cart(vec2 c, float s) {
  vec2 cart = vec2(
    (1. / sr3 * c.x - 1./3. * c.y) * s,
    (2./3. * c.y) * s
  );
  return cart;
}

mat3 rotx(mat3 p, float a) {
  float ca = cos(a);
  float sa = sin(a);
  return mat3(
    1, 0, 0,
    0, ca, sa,
    0, -sa, ca
  ) * p;
}

mat3 roty(mat3 p, float a) {
  float ca = cos(a);
  float sa = sin(a);
  return mat3(
    ca, 0, sa,
    0, 1, 0,
    -sa, 0, ca
  ) * p;
}

mat3 rotz(mat3 p, float a) {
  float ca = cos(a);
  float sa = sin(a);
  return mat3(
    ca, sa, 0,
    -sa, ca, 0,
    0, 0, 1
  ) * p;
}

vec2 rot(vec2 p, float a) {
  float ca = cos(a);
  float sa = sin(a);
  return mat2(
    ca, sa,
    -sa, ca
  ) * p;
}

vec4 hexRing(vec4 c, vec4 fg, vec2 uv, float rad, float width) {
  vec3 hex = cart2hex(uv * 2. - 1., 1.);
  float u = abs(hex.x);
  float v = abs(hex.y);
  float w = abs(hex.z);
  float tot = (u + v + w) / 2.;
  float diff = abs(tot - rad);
  float p = diff / width;
  float widthPx = size.x * width;
  float delta = sr3 / widthPx;
  p = clamp(p, 0., 1.);
  p = smoothstep(0.5 - delta, 0.5 + delta, p);
  c = c + fg * (1. - p);
  return c;
}

vec4 qmul(vec4 a, vec4 b) {
  return vec4(
    a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w,
    a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z,
    a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y,
    a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x
  );
}

vec4 alphamul(vec4 a, vec4 b) {
  return vec4(b.rgb * b.a + a.rgb * (1. - b.a), a.a);
}

float torus(vec3 p, vec2 t)
{
  vec2 q = vec2(length(p.xz)-t.x,p.y);
  return length(q)-t.y;
}

float octahedron(vec3 p, float s)
{
  p = abs(p);
  float m = p.x+p.y+p.z-s;
  vec3 q;
       if( 3.0*p.x < m ) q = p.xyz;
  else if( 3.0*p.y < m ) q = p.yzx;
  else if( 3.0*p.z < m ) q = p.zxy;
  else return m*0.57735027;
    
  float k = clamp(0.5*(q.z-q.y+s),0.0,s); 
  return length(vec3(q.x,q.y-s+k,q.z-k)); 
}

float box(vec3 p, vec3 b)
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float ve(vec3 p, float s) {
  float oct = octahedron(p, s) - 0.01;
  float cube = box(p, vec3(1,1,1) * s * 0.5) - 0.01;
  return max(oct, cube);
}

float tet(vec3 p, float s)
{
    return length(max(vec4(
         (p.x + p.z) - p.y - s,
        -(p.x + p.z) - p.y - s,
         (p.x - p.z) + p.y - s,
        -(p.x - p.z) + p.y - s
        ), 0.0)) * (1. / sr3);
}


float clamp3(float n, float d) {
  return mod(n + d/2., d) - d/2.;
}


float getDist(vec3 p) {
  vec4 tex = texture(bufferImage, p.xy * 0.5 + 0.5);
  float d = p.z - (tex.x + tex.y + tex.z);
  return d;
}

vec3 getNormal(vec3 p) {
  float d = getDist(p);
  vec2 e = vec2(0.0001, 0);
  return normalize(d - vec3(
    getDist(p - e.xyy),
    getDist(p - e.yxy),
    getDist(p - e.yyx)
  ));
}

vec2 march(vec3 r, vec3 d) {
  float m = 0.;
  float e = 1000.;
  vec3 p;
  for (int i = 0; i < ITER; i++) {
    p = r + d * m;
    float dist = getDist(p);
    e = min(dist, e);
    m += dist;
    if (m > MAX_DIST || m < .001)
      break;
  }
  return vec2(m, e);
}

float stdlight(vec3 p, vec3 n) {
  mat3 l;
  float d = 2.;
  l[0].xyz = vec3(1, 0, 0);
  l[1].xyz = vec3(0, 1, 0);
  l[2].xyz = vec3(0, 0, 1);
  l += d;

  float lum = 0.;
  for (int i = 0; i <3; i++) {
    vec3 d =   normalize(l[i] - p);
    float dt = dot(d, n);
    lum +=  clamp(dt, 0., 1.) / length(l[i] - p) * 1.1;
    vec2 sm = march(p + n, d);
    lum *= clamp(sm.y * 2., 0., 1.);
  }
  return lum;
}

float midlight(vec3 p, vec3 n) {
  float lum = 0.;
  float dt = dot(p, n);
  lum +=  clamp(dt * -1., 0., 1.) / length(p) / 2.;
  return lum;
}

float toplight(vec3 p, vec3 n) {
  float lum = 0.;
  vec3 r = reflect(normalize(p), n);
  lum = pow(abs(clamp(r.z, -1., 1.)), 8.) /3.;
  return clamp(lum, 0., 1.);
}

vec3 subhex(vec3 hex, float s) {
  return fract(hex / s) * 2. - 1.;
}

float amax(vec3 v) {
  return max(max(abs(v.x), abs(v.y)), abs(v.z));
}

float amax(vec2 v) {
  return max(abs(v.x), abs(v.y));
}

float osc(float n) {
  return sin(n * tau) * 0.5 + 0.5;
}

float submetric(vec3 hex, float amt) {
  return mix(amax(hex), 1. - length(hex) / sr3, amt);
}

float calc(vec2 cv, float q) {
  float t = fract(time + q);
  float r = 1. - osc(length(cv)  - t);
  float val = 0.;
  vec3 hex = cart2hex(cv, 4.);

  vec3 cell = subhex(hex, 1.);
  float cur = submetric(
      cell,
      0.
    );
  val = fract(cur);

  return val;
}

vec3 col_b = vec3(1./24., 1./48., 1./72.);
vec3 col_w = 1. - vec3(1./36., 1./24., 1./12.);

float samp(vec2 uv, float q) {
  vec4 c = texture(bufferImage, uv);
  float d1 = 0.;
  float d2 = 0.;

  for (float i =0.; i < 18.; i++) {
    vec2 inc = vec2(cos(i / 18. * tau), sin(i / 18. * tau)) * q;
    vec2 p = uv - inc;
    vec4 s = texture(bufferImage, p);
    float d = length(s - c);
    d1 = max(d1, d);
  }
  float a = smoothstep(0.1, 0.2, d1);
  return a;
}

float ease(float n) {
  return (sin((n - 0.5) * pi) + 1.)/2.;
}

float pulse(float n, float o) {
  float v;
  n = (n - 0.25 + o) * tau * 2.;
  v = pow(sin(n), 1.);
  v = v * sin(n);
  return v;
}

vec4 scanz(vec4 c, vec2 uv)
{
  vec3 col = c.rgb;
  float count =240.;
  vec2 cv = uv.yx * 2. - 1.;
  vec3 hex = cart2hex(cv, 1.);
  float y = uv.y;// + sin((dot(hex.x, hex.y) + dot(hex.y, hex.z) + dot(hex.z, hex.x) + r + f) * tau / r) * 1. /count /2.;
  float a = (y * count) * tau;
  vec3 sl = vec3(sin(a), cos(a), sin(a - pi));

  vec4 pcolor = rgb2hsv(vec4(col, 1.));
  col += hsv2rgb(pcolor).xyz * sl * 0.5;
  return vec4(col, c.a);
}

vec4 rgb(vec2 uv) {
  float e = 1./120.;
  vec2 ro = vec2(pulse(time - uv.y, 0.) * e, 0);
  vec2 go = vec2(pulse(time + uv.y, 1./3.) * e, 0);
  vec2 bo = vec2(pulse(time - uv.y, 2./3.) * e, 0);
  float r = texture(bufferImage, uv + ro).r;
  float g = texture(bufferImage, uv + go).g;
  float b = texture(bufferImage, uv + bo).b;
  return vec4(r, g, b, 1);
}

vec4 rgb2(vec2 uv) {
  vec2 cv = uv * 2. - 1.;
  vec2 z = cv;
  vec3 c;
  for (int i = 0; i < 3; i++) {
    cv = z;
    float e = sin((time * 3. + float(i)) * tau / 3. + atan(cv.y, cv.x));
    cv = rot(cv, e * tau  / 180.);
    uv = cv * 0.5 + 0.5;
    c[i] = texture(bufferImage, uv)[i];
  }

  return vec4(c, 1);
}

vec4 rgb3(vec2 uv) {
  vec2 cv = uv * 2. - 1.;
  vec2 z = cv;
  vec3 c;
  for (int i = 0; i < 3; i++) {
    cv = z;
    float ii = float(i);
    float e = sin((ii/3. + time) * tau + atan(cv.y, cv.x));
    cv += e * 1./120.;
    uv = cv * 0.5 + 0.5;
    c[i] = texture(bufferImage, uv)[i];
  }
  return vec4(c, 1);
}

void main()
{
  float q = 1. / 1080.;
  float w = 1. / 216.;
  float t = time * tau;
  vec2 uv = gl_FragCoord.xy / size;
  vec2 cv = uv * 2. - 1.;

  // cv = hexbin(cv.yx, 1.).yx;
  // uv = cv * 0.5 + 0.5;

  vec4 c = vec4(col_w, 1);
  // c.rgb += length(texture(lastFrame, scaleUv(uv, 59./60.)).rgb) * 2./6.;
  // c = texture(lastFrame, (cv * 13./12.) * 0.5 + 0.5);
  // c = max(c, texture(bufferImage, uv));
  c = mix(c, texture(bufferImage, uv),1.);
  // c = rgb3(uv);
  // vec2 n = sin(fract(rot(uv, (uv.x + uv.y) * tau) * 360.) * tau);
  // c.rgb += fract(n.x + n.y) * 0.2 - 0.1;
  // float b = samp(uv, q);
  // c.rgb -= vec3(0) + b;//alphamul(c, vec4(col_b, b));
  // c = scanz(c, uv);
  // c.rg = uv;
  fragColor = c;
} 
