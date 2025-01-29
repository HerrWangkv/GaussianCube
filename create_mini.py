import os
import json
from nuscenes.nuscenes import NuScenes

mini_sample_data_tokens = []
nusc = NuScenes(version='v1.0-mini', dataroot='/storage_local/kwang/nuscenes/raw', verbose=True)
for scene in nusc.scene:
    sample_token = scene['first_sample_token']
    while sample_token != '':
        sample = nusc.get('sample', sample_token)
        for sample_data_token in sample['data'].values():
            mini_sample_data_tokens.append(sample_data_token)
        sample_token = sample['next']

file_path = "/storage_local/kwang/nuscenes/insSeg/v1.0-trainval/nuinsseg.json"
with open(file_path, 'r') as f:
    anns = json.load(f)

mini_anns = []
for d in anns:
    if d['sample_data_token'] in mini_sample_data_tokens:
        mini_anns.append(d)
os.mkdir("/storage_local/kwang/nuscenes/insSeg/v1.0-mini")
file_path = "/storage_local/kwang/nuscenes/insSeg/v1.0-mini/nuinsseg.json"
with open(file_path, 'w') as f:
    json.dump(mini_anns, f)