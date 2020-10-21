#!/usr/bin/env node

const spawn = require('child_process').spawn;

const config = require('../server/config');
const {App} = require('../server/server');

const app = new App();

if (require.main == module) {
  if (config.env == 'development') {
    spawn('bin/sass.sh', ['--watch'], {stdio: ['pipe', process.stdout, 'pipe']});
  }
  app.start();
}
else {
  module.exports = app;
}
