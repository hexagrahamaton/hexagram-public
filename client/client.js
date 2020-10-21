const canvas = document.querySelector('#thecanvas');
const button = document.querySelector('button');
const status = document.querySelector('#framez');

const DIM = 720 * 4;
canvas.width = canvas.height = DIM;
// const ctx = canvas.getContext('2d');
const gl = canvas.getContext('webgl2');

const FRAMES = 10;
const TOTAL_FRAMES = 10;
const INTERVAL = 50;

let lastImg = new Image();

var uniforms = {
  image: null,
  image2: null,
  image3: null,
  image4: null,
  image5: null,
  buffer: null,
  lastFrame: null,
  size: null,
  nbStates: 7,
  threshold: 2,
  counter: 0,
  colors: [
    1, 0, 0,
    1, 1, 0,
    0, 1, 0,
    0, 1, 1,
    0, 0, 1,
    1, 0, 1
  ]
};

const setTimer = (fn) => {
  if (INTERVAL == 0)
    requestAnimationFrame(fn);
  else
    setTimeout(fn, INTERVAL);
}

let play = true;
window.addEventListener('keydown', (ev) => {
  if (ev.ctrlKey) {
    if (ev.key == 's') {
      promptDownload();
    }
    else {
      return;
    }
    ev.preventDefault();
  }
  else if (ev.key == 'Tab') {
    togglePlay();
  }
  else if (ev.key == 'r') {
    resetCounter();
  }
  else if (ev.key == ' ') {
    play && togglePlay();
    animate();
  }
  else {
    return;
  }
  ev.preventDefault();
});
// const TEMPLATE_PATH = 'thin-one.png';
const TEMPLATE_PATH = 'ha-2-text.png';
const TEMPLATE_PATH2 = 'ha-2-tri-raw.png';
// const TEMPLATE_PATH3 = 'hexaware-1-petal.png';
// const TEMPLATE_PATH4 = 'hexaware-1-text.png';
// const TEMPLATE_PATH5 = 'hexaware-1-lines.png';
const IMG_PATH1 = 'frames/%2.png';
const IMG_PATH2 = 'frames/%2.png';
// const IMG_PATH = 'frames-postflake/frame-%5.png';
const VERT_PATH = 'shaders/passthru.vs';
const FRAG_PATH = 'shaders/gha1-display.fs';
const STEP_PATH = 'shaders/g4.fs';
// const STEP_PATH = 'shaders/fracbulb.fs';

let frames = Array(TOTAL_FRAMES).fill();
let framesLoaded = 0;


let counter;

function togglePlay() {
      play = !play;
    if (play)
      animate();
}

function resetCounter() {
  uniforms.counter = 0;
  status.value = 0;
  randomizeCurrTex();
  lastImg = new Image();
  resetTexture(frameTex);
  play || animate();
}

async function promptDownload() {
  let uri = await canvas.toDataURL('image/png', 1);
  let a = document.createElement('a');
  a.href = uri;
  a.download = `frame-${('0000' + uniforms.counter).slice(-4)}.png`;
  a.click();
}

let recording = false;

let template, template2, template3, template4, template5;
let templateLoaded = false;

let displayText, vertText, updateText;

fetch(VERT_PATH).then((res) => res.text()).then((data) => {
  vertText = data;
  tryInit();
});

fetch(FRAG_PATH).then((res) => res.text()).then((data) => {
  displayText = data;
  tryInit();
});

fetch(STEP_PATH).then((res) => res.text()).then((data) => {
  updateText = data;
  tryInit();
});

function tryInit() {
  if (templateLoaded && displayText && vertText && updateText && framesLoaded == TOTAL_FRAMES)
    init();
}

button.onclick = () => {
  recording = !recording;
  if (recording) {
    resetCounter();
    fetch('/reset', {
      method: 'POST',
      mode: 'no-cors',
      headers: {'Content-Type': `text/plain`},
      body: 'ohai i can haz reset plx?'
    });
  }
  button.innerHTML = recording ? 'Stop' : 'Start';
  button.classList.toggle('active', recording);
}

if (typeof TEMPLATE_PATH != 'undefined') {
  template = new Image();
  template.src = TEMPLATE_PATH;
  template.onload = () => {
    templateLoaded = true;
    tryInit();
  };
}

if (typeof TEMPLATE_PATH2 != 'undefined') {
  template2 = new Image();
  template2.src = TEMPLATE_PATH2;
  template2.onload = () => {
    templateLoaded = true;
    tryInit();
  };
}

if (typeof TEMPLATE_PATH3 != 'undefined') {
  template3 = new Image();
  template3.src = TEMPLATE_PATH3;
  template3.onload = () => {
    templateLoaded = true;
    tryInit();
  };
}

if (typeof TEMPLATE_PATH4 != 'undefined') {
  template4 = new Image();
  template4.src = TEMPLATE_PATH4;
  template4.onload = () => {
    templateLoaded = true;
    tryInit();
  };
}

if (typeof TEMPLATE_PATH5 != 'undefined') {
  template5 = new Image();
  template5.src = TEMPLATE_PATH5;
  template5.onload = () => {
    templateLoaded = true;
    tryInit();
  };
}

for (let i = 0; i < FRAMES; i++) {
  let img = new Image();
  let n = parseInt(IMG_PATH1.match(/%(\d)/)[1]);
  img.src = IMG_PATH1.replace(/%\d/, ('0'.repeat(n) + i).slice(-n));
  img.onload = () => {
    frames[i] = img;
    framesLoaded += 1;
    tryInit();
    // console.log(`Frame ${i} loaded lol`);
  };
}
for (let i = 0; i < FRAMES; i++) {
  let img = new Image();
  let n = parseInt(IMG_PATH2.match(/%(\d)/)[1]);
  img.src = IMG_PATH2.replace(/%\d/, ('0'.repeat(n) + i).slice(-n));
  img.onload = () => {
    frames[i + FRAMES] = img;
    framesLoaded += 1;
    tryInit();
    // console.log(`Frame ${i} loaded lol`);
  };
}

let displayProgram;
let updateProgram;

var quadBuf;
var programInfo = {};
var fbInfo = [];
var currTex = 0;
var attachments;
var running = true;

let imgTex, imgTex2, imgTex3, imgTex4, frameTex;

let vertexArray;
let vertexBuffer;
let vertexNumComponents;
let vertexCount;

let previousTime = 0.0;
let degreesPerSecond = 90.0;
const pixel = new Uint8Array([0x0, 0x0, 0x0,0xff]);

function init() {
  updateProgram = twgl.createProgramInfo(gl, [vertText, updateText]);
  displayProgram = twgl.createProgramInfo(gl, [vertText, displayText]);

  quadBuf = twgl.createBufferInfoFromArrays(gl, {
    position: {
      numComponents: 2,
      data: new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1])
    }
  });
  attachments = [{format: gl.RGBA}];
  fbInfo.push(twgl.createFramebufferInfo(gl, attachments));
  fbInfo.push(twgl.createFramebufferInfo(gl, attachments));
  randomizeCurrTex();

  let gui = new dat.GUI();
  gui.close();
  gui.add(uniforms, 'nbStates', 3, 12, 1).onChange(randomizeCurrTex);
  gui.add(uniforms, 'threshold', 2, 4, 1).onChange(randomizeCurrTex);
  gui.add(window, 'running').onChange(animate);

  imgTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, imgTex);
  const level = 0;
  const internalFormat = gl.RGBA;
  const width = 1;
  const height = 1;
  const border = 0;
  const srcFormat = gl.RGBA;
  const srcType = gl.UNSIGNED_BYTE;
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, border, srcFormat, srcType, pixel);

  imgTex2 = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, imgTex2);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, border, srcFormat, srcType, pixel);

  imgTex3 = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, imgTex3);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, border, srcFormat, srcType, pixel);

  imgTex4 = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, imgTex4);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, border, srcFormat, srcType, pixel);

  imgTex5 = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, imgTex5);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, border, srcFormat, srcType, pixel);

  frameTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, frameTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, border, srcFormat, srcType, pixel);

  resetCounter();
  animate();
}

function resetTexture(texture, img=pixel) {
  gl.bindTexture(gl.TEXTURE_2D, texture);
  const level = 0;
  const internalFormat = gl.RGBA;
  const width = 1;
  const height = 1;
  const border = 0;
  const srcFormat = gl.RGBA;
  const srcType = gl.UNSIGNED_BYTE;
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, border, srcFormat, srcType, img);
}

function animate() {

  // let imgSampler = gl.getUniformLocation(displayProgram.program, 'image');
  gl.bindTexture(gl.TEXTURE_2D, imgTex);

  let img = template; //frames[uniforms.counter % TOTAL_FRAMES];
  let img2 = template2; //frames[uniforms.counter % TOTAL_FRAMES];
  let img3 = template3; //frames[uniforms.counter % TOTAL_FRAMES];
  let img4 = template4;
  let img5 = template5;

  gl.bindTexture(gl.TEXTURE_2D, imgTex);
  if (img) {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  }
  uniforms.image = imgTex;

  gl.bindTexture(gl.TEXTURE_2D, imgTex2);
  if (img2) {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img2);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  }
  uniforms.image2 = imgTex2;

  gl.bindTexture(gl.TEXTURE_2D, imgTex3);
  if (img3) {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img3);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  }
  uniforms.image3 = imgTex3;

  gl.bindTexture(gl.TEXTURE_2D, imgTex4);
  if (img4) {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img4);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  }
  uniforms.image4 = imgTex4;

  gl.bindTexture(gl.TEXTURE_2D, imgTex5);
  if (img5) {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img5);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  }
  uniforms.image5 = imgTex5;

  if (lastImg.src) {
    gl.bindTexture(gl.TEXTURE_2D, frameTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, lastImg);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.MIRRORED_REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.MIRRORED_REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  }

  uniforms.lastFrame = frameTex;

  // gl.uniform1i(uSampler, 0);

  gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
  uniforms.size = [gl.drawingBufferWidth, gl.drawingBufferHeight];
  uniforms.buffer = fbInfo[currTex].attachments[0];
  currTex = (currTex + 1) % 2;

  gl.bindFramebuffer(gl.FRAMEBUFFER, fbInfo[currTex].framebuffer);
  gl.useProgram(updateProgram.program);
  twgl.setBuffersAndAttributes(gl, displayProgram, quadBuf);
  twgl.setUniforms(updateProgram, uniforms);
  twgl.drawBufferInfo(gl, quadBuf, gl.TRIANGLE_STRIP);
  uniforms.buffer = fbInfo[currTex].attachments[0];

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.useProgram(displayProgram.program);
  twgl.setBuffersAndAttributes(gl, updateProgram, quadBuf);
  twgl.setUniforms(displayProgram, uniforms);
  twgl.drawBufferInfo(gl, quadBuf, gl.TRIANGLE_STRIP);

  status.value = uniforms.counter;
  uniforms.counter += 1;
  endFrame();
}

async function endFrame() {
  let dataUrl = await canvas.toDataURL('image/png', 1);
  lastImg.src = dataUrl;
  recording && postFrame(dataUrl);
  lastImg.onload = () => {
    play && setTimer(animate);
  }

}

function randomizeCurrTex() {
  var w = gl.drawingBufferWidth;
  var h = gl.drawingBufferHeight;
  var len = w * h * 4;
  var data = new Uint8Array(len);
  for (i = 0; i < len; i++) {
    data[i] = 127; //Math.random() * uniforms.nbStates;
  }

  gl.bindTexture(gl.TEXTURE_2D, fbInfo[currTex].attachments[0]);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
}


async function postFrame(dataUrl) {
  try {
    await fetch('/', {
      method: 'POST',
      mode: 'no-cors',
      headers: {'Content-Type': `text/plain`},
      body: dataUrl
    });
  }
  catch (err) {
    console.error(err);
  }
}
