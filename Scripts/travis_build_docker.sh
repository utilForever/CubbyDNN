#!/usr/bin/env bash

set -e

if [ $# -eq 0 ]
  then
    docker build -t utilforever/cubbydnn .
else
    docker build -f $1 -t utilforever/cubbydnn:$2 .
fi
docker run utilforever/cubbydnn