const fs = require('fs');
const process = require('process');
const pth = require('path');

const yaml = require('yaml');

const util = require('./util');

const DEFAULT_ENV = 'development';
const SERVER_CONFIG_FILEPATH = 'config/server.yml';
const MIME_TYPES_FILEPATH = 'config/mime_types.yml';

class Config {
  constructor() {
    let env = process.env.env || DEFAULT_ENV;
    let rootConfig = getYamlFile(SERVER_CONFIG_FILEPATH);
    util.merge(this, rootConfig.default, rootConfig[env], {env});
    this.mimeTypes = getYamlFile(MIME_TYPES_FILEPATH);
  }
}

function getYamlFile(filename) {
  let text = fs.readFileSync(util.join(filename), 'utf8');
  let obj = yaml.parse(text);
  return obj;
}

module.exports = new Config();
