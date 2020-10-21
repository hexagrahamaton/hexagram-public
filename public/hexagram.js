const canvas = document.querySelector('#thecanvas');
const button = document.querySelector('button');
const status = document.querySelector('#framez');

let DIM = 1080 * 1;
// DIM = 1080 * 4; 
const DURATION = 120;

let VERT_PATH = 'shaders/passthru3.vs';
let FRAG_PATH = 'shaders/filter-display-2.fs';
let STEP_PATH2 = 'shaders/filter-display.fs'; 
// FRAG_PATH = 'shaders/simple-display.fs';
// STEP_PATH2 = 'shaders/simple-display.fs';
let STEP_PATH = 'shaders/template-1.fs';
// STEP_PATH = 'shaders/fractoid-blobs.fs';

let canvasFrames = [];
document.querySelectorAll('.canvas-frame').forEach((e, i) => {
  let frame = new CanvasFrame('canvas' + i, {canvas: e, dim: DIM});
  e.addEventListener('dblclick', () => frame.loadImageFromPrompt());
  canvasFrames.push(frame);
});

canvas.width = canvas.height = DIM;

const gl = canvas.getContext('webgl2');

const INTERVAL = 25;

let lastImg = new Image();

var uniforms = {
  image1: null,
  image2: null,
  image3: null,
  bufferImage: null,
  lastFrame: null,
  size: null,
  nbStates: 7,
  threshold: 2,
  counter: 0,
  duration: DURATION,
  time: 0
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

let counter;

function togglePlay() {
      play = !play;
    if (play)
      animate();
}

function resetCounter() {
  uniforms.counter = 0;
  uniforms.time = 0;
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

let displayText, vertText, updateText, updateText2;

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

fetch(STEP_PATH2).then((res) => res.text()).then((data) => {
  updateText2 = data;
  tryInit();
});

function tryInit() {
  if (displayText && vertText && updateText && updateText2)
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

let displayProgram;
let updateProgram;
let updateProgram2;

var quadBuf;
var programInfo = {};
var fbInfo = [];
var currTex = 0;
var attachments;
var running = true;

let imgTex1, imgTex2, imgTex3, frameTex, tbuffer;

let vertexArray;
let vertexBuffer;
let vertexNumComponents;
let vertexCount;

let previousTime = 0.0;
let degreesPerSecond = 90.0;
const pixel = new Uint8Array([0x0, 0x0, 0x0,0xff]);

function init() {
  updateProgram = twgl.createProgramInfo(gl, [vertText, updateText]);
  updateProgram2 = twgl.createProgramInfo(gl, [vertText, updateText2]);
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
  tbuffer = twgl.createFramebufferInfo(gl, attachments);
  randomizeCurrTex();

  let gui = new dat.GUI();
  gui.close();
  gui.add(uniforms, 'nbStates', 3, 12, 1).onChange(randomizeCurrTex);
  gui.add(uniforms, 'threshold', 2, 4, 1).onChange(randomizeCurrTex);
  gui.add(window, 'running').onChange(animate);

  const level = 0;
  const internalFormat = gl.RGBA;
  const width = 1;
  const height = 1;
  const border = 0;
  const srcFormat = gl.RGBA;
  const srcType = gl.UNSIGNED_BYTE;

  imgTex1 = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, imgTex1);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, border, srcFormat, srcType, pixel);

  imgTex2 = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, imgTex2);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, border, srcFormat, srcType, pixel);

  imgTex3 = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, imgTex3);
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

async function animate() {
  canvasFrames.forEach((e) => e.draw());

  gl.bindTexture(gl.TEXTURE_2D, imgTex1);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, canvasFrames[0].canvas);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  uniforms.image1 = imgTex1;

  gl.bindTexture(gl.TEXTURE_2D, imgTex2);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, canvasFrames[1].canvas);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  uniforms.image2 = imgTex2;

  gl.bindTexture(gl.TEXTURE_2D, imgTex3);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, canvasFrames[2].canvas);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  uniforms.image3 = imgTex3;



  // gl.uniform1i(uSampler, 0);

  gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
  uniforms.size = [gl.drawingBufferWidth, gl.drawingBufferHeight];

  if (lastImg.src) {
    // gl.bindFramebuffer(gl.FRAMEBUFFER, fbInfo[currTex].framebuffer);
    gl.bindTexture(gl.TEXTURE_2D, frameTex);
    // gl.copyTexImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 0, 0, DIM, DIM, 0 );
    //   // void gl.texImage2D(target, level, internalformat, format, type, ImageData? pixels);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, lastImg);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.MIRRORED_REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.MIRRORED_REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  }

  uniforms.lastFrame = frameTex;


  uniforms.bufferImage = fbInfo[currTex].attachments[0];
  // currTex = (currTex + 1) % 2;

  uniforms.lastFrame = uniforms.bufferImage;

  gl.bindFramebuffer(gl.FRAMEBUFFER, tbuffer.framebuffer);
  gl.useProgram(updateProgram.program);
  twgl.setBuffersAndAttributes(gl, updateProgram, quadBuf);
  twgl.setUniforms(updateProgram, uniforms);
  twgl.drawBufferInfo(gl, quadBuf, gl.TRIANGLE_STRIP);

  uniforms.bufferImage = tbuffer.attachments[0];
  uniforms.lastFrame = fbInfo[currTex].attachments[0];
  currTex = (currTex + 1) % 2;

  gl.bindFramebuffer(gl.FRAMEBUFFER, fbInfo[currTex].framebuffer);
  gl.useProgram(updateProgram2.program);
  twgl.setBuffersAndAttributes(gl, updateProgram2, quadBuf);
  twgl.setUniforms(updateProgram2, uniforms);
  twgl.drawBufferInfo(gl, quadBuf, gl.TRIANGLE_STRIP);

  uniforms.bufferImage = fbInfo[currTex].attachments[0];

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.useProgram(displayProgram.program);
  twgl.setBuffersAndAttributes(gl, displayProgram, quadBuf);
  twgl.setUniforms(displayProgram, uniforms);
  twgl.drawBufferInfo(gl, quadBuf, gl.TRIANGLE_STRIP);

  status.value = uniforms.counter;
  uniforms.counter += 1;
  uniforms.time = (uniforms.counter / uniforms.duration) % 1;
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
    data[i] = 0; //Math.random() * uniforms.nbStates;
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

function qmul(a, b) {
  a = {x: a[0], y: a[1], z: a[2], w: a[3]};
  b = {x: b[0], y: b[1], z: b[2], w: b[3]};
  return [
    a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w,
    a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z,
    a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y,
    a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x
  ];
}
