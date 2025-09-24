#! /bin/bash

for scene_idx in {0..9}
do
    python gt.py --scene-idx $scene_idx
done