import os
import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt
import glob
import imageio
import argparse

def save_img(nusc, sample, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    for i, ax in enumerate(axes.flat):
        sample_data = nusc.get('sample_data', sample['data'][cams[i]])
        img_path = os.path.join(nusc.dataroot, sample_data['filename'])
        ax.imshow(plt.imread(img_path))
        ax.set_title(cams[i])
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Render 3D objects in a scene.")
    parser.add_argument("scene_idx", type=int, default=0, help="Index of the scene to render.")
    return parser.parse_args()

def main():
    # Initialize NuScenes dataset
    nusc = NuScenes(version='v1.0-mini', dataroot='/storage_local/kwang/nuscenes/raw', verbose=True)

    # Get scene
    args = parse_args()
    scene = nusc.scene[args.scene_idx]
    first_sample_token = scene['first_sample_token']
    sample = nusc.get('sample', first_sample_token)
    t = 0
    os.makedirs("videos/gt_images", exist_ok=True)
    while sample:
        save_img(nusc, sample, f"videos/gt_images/frame_{t:04d}.png")

        # Move to the next sample
        if sample['next'] == '':
            break
        sample = nusc.get('sample', sample['next'])
        t += 1

    # Generate a video from the saved frames
    frame_paths = sorted(glob.glob("videos/gt_images/frame_*.png"))
    with imageio.get_writer('videos/gt_video.mp4', fps=2) as video_writer:
        for frame_path in frame_paths:
            frame = imageio.imread(frame_path)
            video_writer.append_data(frame)

    # Clean up the frame images and remove the folder
    for frame_path in frame_paths:
        os.remove(frame_path)
    os.rmdir("videos/gt_images")

if __name__ == "__main__":
    main()