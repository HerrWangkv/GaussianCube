import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import yaml
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.map_expansion.map_api import NuScenesMap
from pyquaternion import Quaternion
import warnings
from typing import List
import math
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
import time
import json
from tqdm import tqdm
import argparse


from dataset.nuscenes.nuinsseg import NuInsSeg, CAM_SENSOR
import pandas as pd

warnings.filterwarnings('ignore')

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        ret = func(*args, **kwargs)
        print(f"Done in {time.time() - start_time:.2f} seconds")
        print("="*6)
        return ret
    return wrapper

#################################################
################## Data loader ##################
#################################################

class NuScenesObjects:
    def __init__(self, version, data_root, ins_seg_root, split, verbose=True, min_visibility=4):
        '''
        Args:
            version (str): version of the dataset, e.g. 'v1.0-trainval'
            data_root (str): directory of the dataset
            ins_seg_root (str): directory of the instance segmentation annotations
            verbose (bool): whether to print information of the dataset
            seqs (list): list of scene indices to load
        '''
        self.version = version
        self.nusc = NuInsSeg(version=version, data_root=data_root, ins_seg_root=ins_seg_root, verbose=verbose) 
        self.split = split
        self.min_visibility = min_visibility
        if isinstance(self.split, str):
            self.seqs = []
            self.accumulate_seqs()
        else:
            self.seqs = self.split
            print(f"Number of scenes: {len(self.seqs)}")
            print("="*6)
        self.cache_dir = os.path.join(ins_seg_root, version, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.obj_list = [] # [(seq_idx, frame_idx, ann_token)]
        self._accumulate_objects()

    def accumulate_seqs(self):
        if self.version == 'v1.0-mini' and not self.split.startswith('mini_'):
            self.split = 'mini_' + self.split
        assert self.split in ['train', 'val', 'test', 'mini_train', 'mini_val'], f"Invalid split: {self.split}"
        scene_names = create_splits_scenes()[self.split]
        for i in range(len(self.nusc.scene)):
            if self.nusc.scene[i]['name'] in scene_names:
                self.seqs.append(i)
        print(f"Current split: {self.split}, number of scenes: {len(self.seqs)}")
        print("="*6)
    
    @timer
    def _accumulate_objects(self):
        print('Accumulating objects...')
        cat_num = {}
        for seq_idx in tqdm(self.seqs):
            if os.path.exists(os.path.join(self.cache_dir, f'{seq_idx}_{self.min_visibility}.json')):
                with open(os.path.join(self.cache_dir, f'{seq_idx}_{self.min_visibility}.json'), 'r') as f:
                    obj_list = json.load(f)
            else:
                obj_list = []
                scene_data = self.nusc.scene[seq_idx]
                sample_token = scene_data['first_sample_token']
                frame_idx = 0
                while sample_token != '':
                    sample = self.nusc.get('sample', sample_token)
                    for ann_token in sample['anns']:
                        ann = self.nusc.get('sample_annotation', ann_token)
                        if self.nusc.is_visible(sample, ann_token, self.min_visibility):
                            obj_list.append((ann_token, ann['category_name']))
                    sample_token = sample['next']
                    frame_idx += 1
                with open(os.path.join(self.cache_dir, f'{seq_idx}_{self.min_visibility}.json'), 'w') as f:
                    json.dump(obj_list, f)
            
            for _, cat in obj_list:
                if cat not in cat_num:
                    cat_num[cat] = 0
                cat_num[cat] += 1
            self.obj_list.extend(obj_list)
        cat_num_df = pd.DataFrame(list(cat_num.items()), columns=['Category', 'Count'])
        cat_num_df = cat_num_df.sort_values(by='Count', ascending=True).reset_index(drop=True)
        print(cat_num_df)
        print(f"Number of objects: {len(self.obj_list)}")
        print("="*6)

    def _summarize_params(self, sample, ann, visible_cams):
        cam_front_visible = "CAM_FRONT" in visible_cams
        if not cam_front_visible:
            visible_cams.append("CAM_FRONT")
        cam_tokens = {cam: sample['data'][cam] for cam in visible_cams}
        cam_data = {cam: self.nusc.get('sample_data', cam_tokens[cam]) for cam in cam_tokens}
        ego_pose = self.nusc.get('ego_pose', cam_data['CAM_FRONT']['ego_pose_token'])
        cam_calib_tokens = {cam: cam_data[cam]['calibrated_sensor_token'] for cam in cam_tokens}
        intrinsics = {cam: np.array(self.nusc.get('calibrated_sensor', cam_calib_tokens[cam])['camera_intrinsic']) for cam in cam_tokens}
        
        cam_front_calib_token = cam_calib_tokens["CAM_FRONT"]
        cam_front_calib_data = self.nusc.get('calibrated_sensor', cam_front_calib_token)
        cam_front_to_ego = np.eye(4)
        cam_front_to_ego[:3, :3] = Quaternion(cam_front_calib_data['rotation']).rotation_matrix
        cam_front_to_ego[:3, 3] = np.array(cam_front_calib_data['translation'])
        extrinsics = {}
        for cam in cam_calib_tokens.keys():
            if cam == "CAM_FRONT":
                extrinsics[cam] = np.eye(4)
            else:
                cam_to_ego = np.eye(4)
                calib_token = cam_calib_tokens[cam]
                calib_data = self.nusc.get('calibrated_sensor', calib_token)
                cam_to_ego[:3, :3] = Quaternion(calib_data['rotation']).rotation_matrix
                cam_to_ego[:3, 3] = np.array(calib_data['translation'])
                extrinsics[cam] = np.linalg.inv(cam_front_to_ego) @ cam_to_ego

        ego_to_world = np.eye(4)
        ego_to_world[:3, :3] = Quaternion(ego_pose['rotation']).rotation_matrix
        ego_to_world[:3, 3] = ego_pose['translation']
        cam_front_to_world = ego_to_world @ cam_front_to_ego
        obj_to_world = np.eye(4)
        obj_to_world[:3, :3] = Quaternion(ann['rotation']).rotation_matrix
        obj_to_world[:3, 3] = ann['translation']
        obj_to_cam_front = np.linalg.inv(cam_front_to_world) @ obj_to_world
        
        if not cam_front_visible:
            visible_cams.remove("CAM_FRONT")
            del intrinsics["CAM_FRONT"]
            del extrinsics["CAM_FRONT"]
    
        return intrinsics, extrinsics, obj_to_cam_front 

    def __len__(self):
        return len(self.obj_list)
    
    def __getitem__(self, idx):
        ann_token = self.obj_list[idx][0]
        ann = self.nusc.get('sample_annotation', ann_token)
        sample_token = ann["sample_token"]
        sample = self.nusc.get('sample', sample_token)

        masks = self.nusc.get_masks(sample, ann_token)
        visible_cams = [cam for cam, mask in masks.items() if mask is not None]
        assert len(visible_cams) != 0, f"Object {ann_token} not visible"

        imgs = {}
        for cam in visible_cams:
            img_name = self.nusc.get('sample_data', sample['data'][cam])['filename']
            img = np.array(Image.open(os.path.join(self.nusc.dataroot, img_name)).convert('RGB')).transpose(2, 0, 1)
            imgs[cam] = img
        
        intrinsics, extrinsics, obj_to_cam_front = self._summarize_params(sample, ann, visible_cams)

        assert len(imgs) == len(intrinsics) == len(extrinsics), "Inconsistent number of images, intrinsics, and extrinsics"

        ret = {
            "size": np.array(ann['size']),
            "category": ann['category_name'],
            "masks": masks,
            "imgs": imgs,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "obj_to_cam_front": obj_to_cam_front,
        }
        return ret
    
    def vis(self, idx):
        data = self.__getitem__(idx)
        print(f"Object category: {data['category']}")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        for i, ax in enumerate(axes.flat):
            if cams[i] not in data['imgs']:
                ax.imshow(np.zeros((900, 1600, 3)))
            else:
                img = data['imgs'][cams[i]]
                mask = data['masks'][cams[i]]
                obj_img = img * mask
                ax.imshow(obj_img.transpose(1, 2, 0))
            ax.set_title(cams[i])
            ax.axis('off')
        plt.tight_layout()
        plt.savefig("obj.png", bbox_inches='tight')
        plt.close()
        self.nusc.render_annotation(self.obj_list[idx][0], out_path="gt.png")

# nusc = NuScenesObjects(version='v1.0-mini', data_root='/storage_local/kwang/nuscenes/raw', ins_seg_root='/storage_local/kwang/nuscenes/insSeg', split='train')
# nusc.vis(11)
# breakpoint()