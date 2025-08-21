#!/bin/bash

docker run -it --rm --gpus all --name GaussianCube \
  --privileged \
  --ipc=host \
  --device /dev/fuse \
  --cap-add SYS_ADMIN \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HUGGING_FACE_TOKEN=${HUGGING_FACE_TOKEN} \
  -v /storage_local/kwang/repos/GaussianCube:/GaussianCube \
  -v /mrtstorage/datasets/public/nuscenes.sqfs:/data/nuscenes.sqfs \
  -w /GaussianCube \
  --entrypoint /bin/bash \
  gaussiancube:latest -c "
    # Create mount point
    mkdir -p /data/nuscenes
    
    # Mount the squashfs to temporary location
    squashfuse /data/nuscenes.sqfs /data/nuscenes
    
    # Start interactive bash session
    exec /bin/bash
  "