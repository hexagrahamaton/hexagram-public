const fs = require('fs');
const http = require('http');
const pth = require('path');
const {spawn, execSync} = require('child_process');

const express = require('express');

const config = require('./config');
const util = require('./util');

// ---

class App {
  constructor(opts={}) {
    this.app = express();
    this.config = util.merge({}, config, opts);
    this.resetIndex();
    this.idxChars = this.config.imageFilename.match(/\#+/)[0].length;

    this.app.use(express.static('./public'));
    this.app.post('/reset', (req, res) => {
      this.resetIndex();
      res.end('sure lol');
    })
    this.app.post('/', (req, res) => {
      this.processData(req);
      res.end('lgtm');
    });
  }

  resetIndex() {
    this.idx = this.config.imageStartIndex;
  }

  start() {
    execSync(`mkdir -p ${config.output}`);
    config.saveVideo && this.startEncoder();

    this.app.listen(config.port, () => {
      console.log(`Listening on port ${config.port} lol...`);
    });
    process.on('SIGINT', () => {
      if (this.child) {
        this.child.stdin.end();
        this.child.on('exit', () => {
          console.log('Exiting thing...');
          process.exit();
        });
      }
      else {
        process.exit();
      }
    });
  }

  startEncoder() {
    let args = [
      '-y',
      '-c:v', 'png',
      '-r', `${config.fps}`,
      '-f', 'image2pipe',
      '-i', '-',
      '-pix_fmt', 'yuv420p',

      '-vf', `scale=${config.width}x${config.height}`,
      '-c:v', config.codec,
      '-crf', `${config.crf}`,
      pth.join(config.output, config.videoFilename),
    ];
    this.child = spawn('ffmpeg', args, {stdio: ['pipe', 'pipe', 'pipe']});
    this.child.on('exit', () => console.log('Exiting encoder...'));
    this.child.stdout.on('data', (data) => {
      console.log(`ENCODER: ${data}`);
    });
    this.child.stderr.on('data', (data) => {
      console.error(`ENCODER: ${data}`);
    });
  }

  async processData(req) {
    let data = '';
    req.on('data', (chunk) => data += chunk);
    req.on('end', () => {
      let match = data.match(/:([\w\/]+);/);
      let ext = config.mimeTypes[match[1]];
      let base64 = data.slice(data.indexOf(',') + 1);
      let buf = Buffer.from(base64, 'base64');
      if (config.saveImages) {
        let filepath = pth.join(config.output, config.imageFilename + ext);
        let idxString = ('0'.repeat(this.idxChars) + (this.idx ++)).slice(-this.idxChars);
        filepath = filepath.replace(/\#+/, idxString);
        console.log(`Writing "${filepath}"...`)
        fs.writeFileSync(filepath, buf);
      }
      if (config.saveVideo) {
        this.child.stdin.write(buf);
      }
    });
  }
}

module.exports = {App};
