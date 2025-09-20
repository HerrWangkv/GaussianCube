#!/bin/bash

docker run -it --rm --gpus all --name GaussianCube \
  --privileged \
  --ipc=host \
  -u $(id -u):$(id -g) \
  --device /dev/fuse \
  -e XDG_CACHE_HOME=/workspace/.cache \
  -e MPLCONFIGDIR=/workspace/.cache/.config/matplotlib \
  -e HUGGING_FACE_TOKEN=${HUGGING_FACE_TOKEN} \
  -v /storage_local/kwang/repos/GaussianCube:/workspace \
  -v /mrtstorage/datasets/public/nuscenes.sqfs:/data/nuscenes.sqfs \
  -w /workspace \
  --entrypoint /bin/bash \
  gaussiancube:latest -c "
    # Create mount point
    mkdir -p /data/nuscenes
    # Mount the squashfs to temporary location
    squashfuse /data/nuscenes.sqfs /data/nuscenes
    # Start interactive bash session
    exec /bin/bash
  "
