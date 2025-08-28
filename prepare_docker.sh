#!/bin/bash

docker run -it --rm --gpus all --name GaussianCube \
  --ipc=host \
  -u $(id -u):$(id -g) \
  -e XDG_CACHE_HOME=/workspace/.cache \
  -e MPLCONFIGDIR=/workspace/.cache/.config/matplotlib \
  -e HUGGING_FACE_TOKEN=${HUGGING_FACE_TOKEN} \
  -v /storage_local/kwang/repos/GaussianCube:/workspace \
  -w /workspace \
  --entrypoint /bin/bash \
  gaussiancube:latest