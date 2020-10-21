window.addEventListener('DOMContentLoaded', () => {
  console.log('wedge');
  for (let group of document.querySelectorAll('.tabs')) {
    let tabs = Array.from(group.querySelectorAll('.tab-tab'));
    let panes = Array.from(group.querySelectorAll('.tab-pane'));
    let body = group.querySelector('.tab-body');
    tabs.forEach((tab, idx) => {
      tab.onclick = () => {
        tabs.forEach((tab) => tab.classList.remove('selected'));
        panes.forEach((pane) => pane.classList.remove('active'));
        tabs[idx].classList.add('selected');
        panes[idx].classList.add('active');
      }
    });
  }
});

const tau = Math.PI * 2;
const hext = tau / 6;
let rad = 120;
let dia = rad * 2;
let met = new Path2D();
let {cos, sin} = Math;

let innerHex = Array(6).fill().map((_, i) => [sin(i * hext) * dia, cos(i * hext) * dia]);
let outerHex = Array(6).fill().map((_, i) => [sin(i * hext) * dia * 2, cos(i * hext) * dia * 2]);
let fHex = Array(6).fill().map((_, i) => [sin(i * hext) * dia * 3, cos(i * hext) * dia * 3]);
let hexes = [innerHex, outerHex];

met.arc(0, 0, rad, 0, tau);
for (let i = 0; i < 2; i++) for (let j = 0; j < 6; j++) {
  let [x, y] = hexes[i][j];
  met.moveTo(x + rad, y);
  met.arc(x, y, rad, 0, tau);
}
for (let i = 0; i < 2; i++) {
  let hex = hexes[i];
  met.moveTo(...hex[0]);
  for (let j = 1; j < 6; j++) {
    met.lineTo(...hex[j]);
  }
  met.closePath();
}

let met2 = new Path2D();
for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
  let hex = hexes[i];
  met.moveTo(...hex[j]);
  met.lineTo(...hex[j + 2]);
  met.lineTo(...hex[j + 4]);
  met.closePath();
}

for (let j = 0; j < 6; j ++) {
  let pt0 = hexes[1][j];
  let pt1 = hexes[0][(j + 2) % 6];
  let pt2 = hexes[0][(j + 4) % 6];
  let opp = hexes[1][(j + 3) % 6];
  met.moveTo(...pt0);
  met.lineTo(...pt1);
  met.lineTo(...pt2);
  met.closePath();
  if (j % 2 == 0) {
    met.moveTo(...pt0);
    met.lineTo(...opp);
    met.closePath();
  }
}

// window.drawFn1 = (ctx) => {
//   let canvas = ctx.canvas;
//   ctx.lineJoin = 'miter';
//   ctx.lineCap = 'flat';
//   ctx.lineWidth = 2;
//   ctx.fillStyle = null;
//   ctx.strokeStyle = '#000';
//   ctx.clearRect(-canvas.width, -canvas.height, canvas.width * 2, canvas.height * 2);
//   ctx.stroke(met);
//   ctx.lineJoin = 'flat';
//   ctx.stroke(met2);
// };

// const sr3 = Math.sqrt(3);
// window.drawFn2 = (ctx) => {
//   // ctx.globalCompositeOperation = 'xor';
//   ctx.lineJoin = 'round';
//   ctx.fillStyle = '#000';
//   ctx.strokeStyle = null;
//   let r = rad / sr3 * 2;
//   let s = r / 2;
//   let canvas = ctx.canvas;
//   ctx.clearRect(-canvas.width, -canvas.height, canvas.width * 2, canvas.height * 2);

//   ctx.rect(-canvas.width, -canvas.height, canvas.width * 2, canvas.height * 2);
//   ctx.fill();
//   ctx.strokStyle = null;
//   for (let i = -5; i < 6; i++) for (let j = -5; j < 6; j++) {
//     ctx.beginPath();
//     for (let j = 0; j < 6; j++) {
      
//     }
//     ctx.arc(i * s * 1.5, (j + i % 2 / 2) * s * sr3, r, 0, tau);
//     ctx.fill();
//   }
//   // ctx.restore();
// }
