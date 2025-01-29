import json
import os
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

def main():
    min_visibility = 4
    cache_dir = "/storage_local/kwang/nuscenes/insSeg/v1.0-trainval/cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    nusc = NuScenes(version='v1.0-trainval', dataroot='/storage_local/kwang/nuscenes/raw', verbose=True)
    scene_indices = {scene['name']: i for i, scene in enumerate(nusc.scene)}
    visible_objects = [[] for _ in range(850)]
    file_path = "/storage_local/kwang/nuscenes/insSeg/v1.0-trainval/nuinsseg.json"
    print("Loading annotations...")
    with open(file_path, 'r') as f:
        anns = json.load(f)
    print("Organizing annotations...")
    for ann in tqdm(anns):
        if int(ann['visibility_token']) >= min_visibility and len(ann['instance_mask']) != 0:
            sample_data_token = ann['sample_data_token']
            sample_data = nusc.get('sample_data', sample_data_token)
            sample_token = sample_data['sample_token']
            sample = nusc.get('sample', sample_token)
            scene_token = sample['scene_token']
            scene = nusc.get('scene', scene_token)
            scene_idx = scene_indices[scene['name']]
            ann_token = ann['token']
            visible_objects[scene_idx].append((ann_token, ann['category_name']))

    print("Saving annotations...")
    for i, visible_objects_scene in enumerate(tqdm(visible_objects)):
        with open(os.path.join(cache_dir, f'{i}_{min_visibility}.json'), 'w') as f:
            json.dump(visible_objects_scene, f)

if __name__ == "__main__":
    main()