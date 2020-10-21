#!/bin/sh

cd $(dirname "$0")
cd ..
BINPATH=node_modules/.bin/node-sass

SOURCE=client/scss/style.scss
TARGET=public/hexagram.css

if [ "$1" = "--watch" ]; then
  BINPATH="$BINPATH --watch"
fi

$BINPATH $SOURCE $TARGET
