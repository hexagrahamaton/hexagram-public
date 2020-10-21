#version 300 es
#ifdef GL_ES
  precision highp float;
#endif

#define MAX_DIST 30.
#define SURF_DIST 0.000001
#define ITER 128

// #version 300 es
uniform float counter;
uniform vec2 size;
uniform sampler2D lastFrame;
uniform sampler2D buffer;
uniform float duration;
uniform float time;

float pi = 3.141592653;
float tau = 6.2831853;

uniform sampler2D image1;
uniform sampler2D image2;
uniform sampler2D image3;

out vec4 fragColor;

vec2 unit = vec2(1, 0);

// vec4 bg = vec4(0.812, 0.812, 0.644, 1.);
vec4 bg = vec4(0., 0., 0., 1.);

float baseRes = 1./3.;
float radialRes = 1.5;

float sr3 = sqrt(3.);
float sr2 = sqrt(2.);

vec4 white = vec4(1.,1.,1.,1.);
vec4 black = vec4(0.,0.,0.,1.);

#define STEPS 2
vec3 col_b = vec3(1./24., 1./48., 1./72.);
vec3 col_w = 1. - vec3(1./36., 1./24., 1./12.);
vec3 col_g = 1./6. + vec3(1./24., 1./48., 1./72.);
float q = 1./720.;

vec3 col0 = vec3(142,100,157) / 256.;
vec3 col1 = vec3(202,66,54) / 256.;
vec3 col2 = vec3(245,130,51) / 256.;
vec3 col3 = vec3(255,204,50) / 256.;
vec3 col4 = vec3(87,166,69) / 256.;
vec3 col5 = vec3(76,149,193) / 256.;

int seq[64] = int[64](
  077,
  000,
  042,
  021,
  072,
  027,
  020,
  002,
  073,
  067,
  070,
  007,
  057,
  075,
  010,
  004,
  046,
  031,
  060,
  003,
  045,
  051,
  001,
  040,
  047,
  071,
  041,
  036,
  022,
  055,
  016,
  034,
  017,
  074,
  005,
  050,
  012,
  024,
  061,
  043,
  076,
  037,
  006,
  030,
  024,
  032,
  056,
  032,
  056,
  035,
  044,
  011,
  013,
  064,
  054,
  015,
  033,
  066,
  023,
  062,
  063,
  014,
  052,
  025
);

// in vec4 gl_FragCoord;
// out vec4 gl_FragColor;

bool isnan(float n) {
  return !(n <= 0. || 0. <= n);
}

vec4 qmul(vec4 a, vec4 b) {
  return vec4(
    a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w,
    a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z,
    a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y,
    a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x
  );
}

vec4 qsqr( vec4 a )
{
    return vec4( a.x*a.x - dot(a.yzw,a.yzw), 2.0*a.x*(a.yzw) );
}

vec4 qcube( vec4 a )
{
  return a * ( 4.0*a.x*a.x - dot(a,a)*vec4(3.0,1.0,1.0,1.0) );
}


vec4 alphamul(vec4 a, vec4 b) {
  return vec4(b.rgb * b.a + a.rgb * (1. - b.a), a.a);
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

float cmod (vec2 z) {
  return length(z);
}

vec2 cexp(vec2 z) {
  return vec2(cos(z.y), sin(z.y)) * exp(z.x);
}

vec2 cpow (vec2 z, float x) {
  float r = length(z);
  float theta = atan(z.y, z.x) * x;
  return vec2(cos(theta), sin(theta)) * pow(r, x);
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

vec2 csqrt (vec2 z) {
  float t = sqrt(2.0 * (cmod(z) + (z.x >= 0.0 ? z.x : -z.x)));
  vec2 f = vec2(0.5 * t, abs(z.y) / t);

  if (z.x < 0.0) f.xy = f.yx;
  if (z.y < 0.0) f.y = -f.y;

  return f;
}

vec2 cdiv (vec2 a, vec2 b) {
  float e, f;
  float g = 1.0;
  float h = 1.0;

  if( abs(b.x) >= abs(b.y) ) {
    e = b.y / b.x;
    f = b.x + b.y * e;
    h = e;
  } else {
    e = b.x / b.y;
    f = b.x * e + b.y;
    g = e;
  }

  return (a * g + h * vec2(a.y, -a.x)) / f;
}

vec2 sinhcosh (float x) {
  vec2 ex = exp(vec2(x, -x));
  return 0.5 * (ex - vec2(ex.y, -ex.x));
}

vec2 catan (vec2 z) {
  float a = z.x * z.x + (1.0 - z.y) * (1.0 - z.y);
  vec2 b = clog(vec2(1.0 - z.y * z.y - z.x * z.x, -2.0 * z.x) / a);
  return 0.5 * vec2(-b.y, b.x);
}

vec2 catanh (vec2 z) {
  float oneMinus = 1.0 - z.x;
  float onePlus = 1.0 + z.x;
  float d = oneMinus * oneMinus + z.y * z.y;

  vec2 x = vec2(onePlus * oneMinus - z.y * z.y, z.y * 2.0) / d;

  vec2 result = vec2(log(length(x)), atan(x.y, x.x)) * 0.5;

  return result;
}

vec2 casin (vec2 z) {
  vec2 a = csqrt(vec2(
    z.y * z.y - z.x * z.x + 1.0,
    -2.0 * z.x * z.y
  ));

  vec2 b = clog(vec2(
    a.x - z.y,
    a.y + z.x
  ));

  return vec2(b.y, -b.x);
}

vec2 casinh (vec2 z) {
  vec2 res = casin(vec2(z.y, -z.x));
  return vec2(-res.y, res.x);
}

vec2 cacot (vec2 z) {
  return catan(vec2(z.x, -z.y) / dot(z, z));
}

vec2 cacoth(vec2 z) {
  return catanh(vec2(z.x, -z.y) / dot(z, z));
}


vec2 csin (vec2 z) {
  return sinhcosh(z.y).yx * vec2(sin(z.x), cos(z.x));
}

vec2 csinh (vec2 z) {
  return sinhcosh(z.x) * vec2(cos(z.y), sin(z.y));
}

vec2 ccos (vec2 z) {
  return sinhcosh(z.y).yx * vec2(cos(z.x), -sin(z.x));
}

vec2 ccosh (vec2 z) {
  return sinhcosh(z.x).yx * vec2(cos(z.y), sin(z.y));
}

vec2 ctan (vec2 z) {
  vec2 e2iz = cexp(2.0 * vec2(-z.y, z.x));

  return cdiv(
    e2iz - vec2(1, 0),
    vec2(-e2iz.y, 1.0 + e2iz.x)
  );
}

vec2 ctanh (vec2 z) {
  z *= 2.0;
  vec2 sch = sinhcosh(z.x);
  return vec2(sch.x, sin(z.y)) / (sch.y + cos(z.y));
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

vec2 rot(vec2 p, float a) {
  float ca = cos(a);
  float sa = sin(a);
  return mat2(
    ca, sa,
    -sa, ca
  ) * p;
}

vec3 rotc(vec3 p) {
  p.yz = rot(p.yz, 54.735 / 180. * pi);
  p.xy = rot(p.xy, 135. / 180. * pi);
  return p;
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

vec2 angle2vec(float a) {
  return vec2(sin(a), cos(a));
}

vec2 dreflect(vec2 cv, float a) {
  vec2 n = angle2vec(a);
  cv -= n * min(0., dot(cv, n)) * 2.;
  return cv;
}

vec2 folder(vec2 cv) {
  cv.y -= 0.5/sr3;
  cv.x = abs(cv.x);

  vec2 vec = angle2vec(pi *5./6.);
  float dt = dot(cv - vec2(0.5, 0.), vec);
  cv -= vec * max(0., dt) * 2.;

  vec2 n = angle2vec(tau / 3.);
  cv.x += .5;

  float inc = 1.;
  for (int i = 0; i < 4; i++) {
    cv = cv * 3. - vec2(1.5, 0.);
    cv.x = abs(cv.x);
    cv.x -= 0.5;
    cv -= n * min(0., dot(cv, n)) * 2.;
    inc *= 3.;
  }
  cv = cv / inc;

  return cv;
}

void hexRing(inout vec4 c, vec2 uv, float rad, float width) {
  vec3 hex = uv2hex(uv);
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
  c = c + white * (1. - p);
}

vec3 subhex(vec3 hex, float s) {
  return fract(hex * s / 2.) * 2. - 1.;
}

float amax(vec4 v) {
  return max(max(max(v.w, abs(v.x)), abs(v.y)), abs(v.z));
}

float amax(vec3 v) {
  return max(max(abs(v.x), abs(v.y)), abs(v.z));
}

float amax(vec2 v) {
  return max(abs(v.x), abs(v.y));
}


float osc(float n) {
  return sin((n - 0.25) * tau) * 0.5 + 0.5;
}

vec2 osc(vec2 n) {
  return sin(n * tau) * 0.5 + 0.5;
}
vec3 osc(vec3 n) {
  return sin(n * tau) * 0.5 + 0.5;
}
vec4 osc(vec4 n) {
  return sin(n * tau) * 0.5 + 0.5;
}

float isostep(float r, float q, float n) {
  float q2 = q / 2.;
  return smoothstep(1. + q2, 1. - q2, n);
}

float xsum(float s, float q) {
  return s + q - 2. * s * q;
}

float submetric(vec3 hex, float amt) {
  return mix(amax(hex), 1.0 - length(hex) / sr3, amt);
}

float grid(vec2 cv, float q) {
  float t = fract(time + q);
  float r = 1. - osc(length(cv)  - t);
  float val = 0.;
  vec3 hex = cart2hex(cv, 4.);

  vec3 cell = subhex(hex, 1.);
  float cur = submetric(
      cell,
      0.
    );// * (1.5 + r) * 2.;
  val = fract(cur);

  return val;
}

float torus(vec3 p, vec2 t)
{
  vec2 q = vec2(length(p.xy) - t.x, p.z);
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

float mand(vec2 c, inout vec3 t) {
  vec3 hex = cart2hex(c, 1.);
  float tt =0.;//= time * tau;
  vec2 u = rot(unit.yx, tt + 0.);
  vec2 v = rot(unit.yx, tt + tau * 1./3.);
  vec2 w = rot(unit.yx, tt + tau * 2./3.);
  u *= 5.;
  v *= 5.;
  w *= 5.;
  vec2 z = c;
  float r = length(z);
  t = vec3(
    length(c - u),
    length(c - v),
    length(c - w)
  );
  // z = casin(z);
  // z = ctan(z);
  z = cmul(z, cpow(vec2(0., 1.), time * 4.));
  for (int i = 0; i < 32; i++) {
    z = cpow(z, 7.);
    z = ccos(z);
    z += c;
    r = length(z);
    t.x = min(length(z - u), t.x);
    t.y = min(length(z - v), t.y);
    t.z = min(length(z - w), t.z);
    if (r > 1000.)
      break;
  }
  t *= 3.;
  return r;
}

float box(vec2 p, vec2 s) {
  return length(abs(p) - abs(s));
}

float cross(vec3 p, float r) {
  float e = 40.;
  float a = box(p, vec3(e, 1, 1) * r);
  float b = box(p, vec3(1, e, 1) * r);
  float c = box(p, vec3(1, 1, e) * r);
  return min(min(a, b), c);
}

float smin(float a, float b, float k) {
  float h = max(k - abs(a - b), 0.) / k;
  return min(a, b) - pow(h, 3.) * k / 6.;
}

float prism(vec3 p, vec2 h) {
  vec3 k = vec3(-sr3/2., 0.5, sr3/3.);
  p = abs(p);
  p.xy -= 2. * min(dot(k.xy, p.xy), 0.) * k.xy;
  vec2 d = vec2(
    length(p.xy - vec2(clamp(p.x, -k.z * h.x, k.z * h.x), h.x)) * sign(p.y - h.x),
    p.z - h.y
  );
  return min(max(d.x, d.y), 0.0) + length(max(d, 0.));
}

// Doesn't work lol
float prism2(vec3 p, vec2 dim) {
  vec3 hex = cart2hex(p.xy, 1.);
  float r = amax(hex) - dim.x;
  float d = abs(p.z) - dim.y;
  if (r > 0. && d > 0.)
    return length(vec2(max(0., d), r));
  else if (r < 0. && d < 0.)
    return max(r, d);
  else {
    float ar = abs(r);
    float ad = abs(d);
    float s = step(0., r) * step(0., d);
    s = s + (1. - s) * -1.;
    float dist = min(ad, ar) * s;
    return dist;
  }
}

float getDist(vec4 p) {
  vec3 c = p.xyz;
  c = sin((c * 2. + time) * tau) * 0.07;
  c = rotc(c);
  c.xy = rot(c.xy, time * tau);
  float cube = box(c, vec3(0.1));
  return cube * 0.5;
}

vec4 getNormal(vec4 p) {
  float d = getDist(p);
  vec2 e = vec2(0.0001, 0);
  return normalize(d - vec4(
    getDist(p - e.xyyy),
    getDist(p - e.yxyy),
    getDist(p - e.yyxy),
    getDist(p - e.yyyx)
  ));
}

vec2 march(vec4 r, vec4 d) {
  float m = 0.;
  float e = 1000.;
  vec4 p = r;
  for (int i = 0; i < ITER; i++) {
    // p.z = fract(p.z);
    p = fract(p + 0.5) - 0.5;
    float dist = getDist(p);
    e = min(dist, e);
    m += dist;
    p = r + d * m;
    if (m > MAX_DIST || abs(dist) < SURF_DIST)
      break;
  }
  return vec2(m, e);
}

float stdlight(vec4 p, vec4 n) {
  float d = 1.5;
  float lum = 0.;
  for (int i = 0; i <6; i++) {
    float ii = float(i);
    vec4 l = vec4(cos(ii * tau / 6.), sin(ii * tau / 6.), 1, 0);

    vec4 v = l * d - p;
    vec4 d = normalize(v);
    float dt = dot(d, n);
    float distFactor = length(v);
    distFactor = min(1., 1. / pow(distFactor, 2.));
    float val = clamp(dt, 0., 0.1) * distFactor;
    vec2 sm = march(p + n, d);
    // val *= clamp(sm.y * 2., 0., 1.);
    lum += val;
    // if (sm.x < length(l[i] - p))
    //   lum *= 0.;
  }
  return lum;
}

float midlight(vec4 p, vec4 n) {
  float lum = 0.;
  float dt = dot(p, n);
  lum +=  clamp(dt * -1., 0., 1.) / length(p) / 1.;
  return lum;
}

float toplight(vec4 p, vec4 n) {
  float lum = 0.;
  vec3 r = reflect(vec3(0, 0, 1), n.xyz);
  lum = pow(abs(clamp(r.z, -1., 1.)),8.) /1.;
  return clamp(lum, 0., 1.);
}

vec3 flakefold(vec2 cv) {
  float t= time;
  float qf = 0.01;
  float w = 1./360.;

  float s = 2.;
  float stp = 2.;
  vec2 n = angle2vec(tau / 6.);

  vec3 hx = cart2hex(cv, 2./3.);
  float r = amax(hx);
  float val = smoothstep(1., 1. - q * qf * s * 2./3., r);
  float sum = 0.;
  float F_STEPS = float(STEPS);
  float epoch = floor(t * F_STEPS);
  float sec = fract(t * F_STEPS);
  float cur = 0.;
  for (int i = 0; i < STEPS; i++) {
    float epochCoef = clamp(epoch - float(i), 0., 1.);
    float secCoef = (epoch - float(i) + 0.) == 0. ? 1. : 0.;
    float coef = max(epochCoef, secCoef * sec);
    float dt = dot(cv, n);
    if (i > 0)
      cv -= n * min(0., dt) * 2.;
    cv *= stp;
    if (i > 0)
      cv.y -= (2. + coef * 1.);
 
    cv.y = abs(cv.y);
    cv.x = -abs(cv.x);

    s *= stp;

    vec3 hex = cart2hex(cv, 1. / coef);
    float r = amax(hex);
    float d = smoothstep(1., 1. - q * qf * s / coef, r);
    // d += 1. - step(1., r);
    d = clamp(d, 0., 1.);
    // if (i > 0)
    sum = xsum(sum, d);
    cur += 1.;
  }
  val = xsum(val, sum);
  return vec3(val);
}

float flexhull(vec2 cv, float r, float q, float t) {
  vec3 hex = cart2hex(cv, 1./r);
  float d = amax(hex);
  float e = length(hex);
  d = mix(d, e, t);
  d = smoothstep(1., 1. - q/r, d);
  return clamp(d, 0., 1.);
}

void main() { 
  vec2 uv = gl_FragCoord.xy / size;
  vec2 cv = uv * 2. - 1.;
  cv *= 1.5;
  // cv.x = abs(cv.x);
  // vec2 p = vec2(length(cv), atan(cv.y, cv.x));
  // p.y = p.y * 3.;
  // cv = vec2(cos(p.y), sin(p.y)) * p.x;
  cv = cv.yx;
  cv = folder(cv);
  uv = cv * 0.5 + 0.5;
  uv.y = 1. - uv.y;
  fragColor = texture(image1, uv);
  return;
  vec4 c = black;
  vec3 tr;
  float s = mand(cv, tr);
  float nan = isnan(s) ? 1. : 0.;
  s = step(1., s);
  c.rgb += s - tr;
  c.rgb = col_w * (1. - nan) + nan * col_b;
  fragColor = vec4(c.rgb, 1.);
  return;
  cv = folder(cv /1.25) * 1.25;
  cv = rot(cv, time * tau);

  uv = cv * 0.5 + 0.5;
  c = texture(image1, vec2(uv.x, 1. - uv.y));
  vec4 c2 = texture(image2, vec2(uv.x, 1. - uv.y));
  float m = step(0., cv.y);
  c = m * c2 + (1. - m) * c;
  c = rgb2hsv(c);
  // c.y = max(0.2, c.y);
  c.z = max(0.2, c.z);
  c = hsv2rgb(c);
  fragColor = vec4(c.rgb, 1.);
}
