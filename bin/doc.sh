#!/bin/sh

cd $(dirname "$0")
cd ..
BINPATH=node_modules/.bin/jsdoc

$BINPATH -c config/jsdoc/config.json --readme README.md && echo 'Done'
