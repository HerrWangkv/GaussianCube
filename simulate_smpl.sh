#! /bin/bash

for scene_idx in {4..7}
do
    python simulate_smpl.py --scene-idx $scene_idx --hz 120
done