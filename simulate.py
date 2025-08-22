import os
import random
import argparse
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import contextlib
import io
from omegaconf import OmegaConf
from mpi4py import MPI
from huggingface_hub import hf_hub_download

from model.unet import UNetModel
from model.clip import FrozenCLIPEmbedder
from model.dpmsolver import NoiseScheduleVP, model_wrapper, DPM_Solver
from utils import dist_util, logger
from utils.script_util import create_gaussian_diffusion, init_volume_grid, build_single_viewpoint_cam
from dataset.dataset_render import load_data
from gaussian_renderer import parse_volume_data
import imageio
from tqdm import tqdm
import glob
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from copy import deepcopy
from gsplat.rendering import rasterization
from PIL import Image


MODEL_TYPES = {
    'xstart': 'x_start',
    'v': 'v',
    'eps': 'noise',
}

# Model repository mapping
MODEL_REPOS = {
    "objaverse_v1.0": {
        "repo_id": "BwZhang/GaussianCube-Objaverse",
        "revision": "main",
        "model_path": "v1.0/objaverse_ckpt.pt",
        "mean_path": "v1.0/mean.pt",
        "std_path": "v1.0/std.pt",
        "bound": 0.5
    },
    "objaverse_v1.1": {
        "repo_id": "BwZhang/GaussianCube-Objaverse",
        "revision": "main",
        "model_path": "v1.1/objaverse_ckpt.pt",
        "mean_path": "v1.1/mean.pt",
        "std_path": "v1.1/std.pt",
        "bound": 0.5
    },
    "omniobject3d": {
        "repo_id": "BwZhang/GaussianCube-OmniObject3D-v1.0",
        "revision": "main",
        "model_path": "OmniObject3D_ckpt.pt",
        "mean_path": "mean.pt",
        "std_path": "std.pt",
        "bound": 1.0
    },
    "shapenet_car": {
        "repo_id": "BwZhang/GaussianCube-ShapeNetCar-v1.0",
        "revision": "main",
        "model_path": "shapenet_car_ckpt.pt", 
        "mean_path": "mean.pt",
        "std_path": "std.pt",
        "bound": 0.45
    },
    "shapenet_chair": {
        "repo_id": "BwZhang/GaussianCube-ShapeNetChair-v1.0",
        "revision": "main",
        "model_path": "shapenet_chair_ckpt.pt",
        "mean_path": "mean.pt",
        "std_path": "std.pt",
        "bound": 0.35
    }
}

def download_model_files(model_name):
    """Download model files from Hugging Face Hub."""
    if model_name not in MODEL_REPOS:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(MODEL_REPOS.keys())}")

    model_info = MODEL_REPOS[model_name]
    downloaded_files = {}

    try:
        # Download model checkpoint
        downloaded_files["ckpt"] = hf_hub_download(
            repo_id=model_info["repo_id"],
            filename=model_info["model_path"],
            revision=model_info["revision"]
        )

        # Download mean file
        downloaded_files["mean"] = hf_hub_download(
            repo_id=model_info["repo_id"],
            filename=model_info["mean_path"],
            revision=model_info["revision"]
        )

        # Download std file
        downloaded_files["std"] = hf_hub_download(
            repo_id=model_info["repo_id"],
            filename=model_info["std_path"],
            revision=model_info["revision"]
        )

        downloaded_files["bound"] = model_info["bound"]

    except Exception as e:
        print(f"Error downloading files for {model_name}: {e}")
        raise

    return downloaded_files

# def seed_everything(seed: int):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)

def quat_multiply(quaternion0, quaternion1):
    w0, x0, y0, z0 = torch.split(quaternion0, 1, dim=-1)
    w1, x1, y1, z1 = torch.split(quaternion1, 1, dim=-1)
    w = -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0
    x = x1 * w0 - y1 * z0 + z1 * y0 + w1 * x0
    y = x1 * z0 + y1 * w0 - z1 * x0 + w1 * y0
    z = -x1 * y0 + y1 * x0 + z1 * w0 + w1 * z0

    return torch.concat((w, x, y, z), dim=-1)

def initialize_shared_models():
    """Initialize models once to be shared across all objects."""

    model_and_diffusion_config = OmegaConf.load("configs/objaverse_text_cond.yml")
    downloaded_files = download_model_files("objaverse_v1.1")

    ckpt = downloaded_files["ckpt"]
    mean_file = downloaded_files["mean"]
    std_file = downloaded_files["std"]
    bound = downloaded_files["bound"]

    dist_util.setup_dist()
    torch.cuda.set_device(dist_util.dev())

    model_and_diffusion_config['model']['precision'] = "32"
    model = UNetModel(**model_and_diffusion_config['model'])

    diffusion = create_gaussian_diffusion(**model_and_diffusion_config['diffusion'])
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(dist_util.dev())
    model.eval()

    clip_text_encoder = FrozenCLIPEmbedder()
    clip_text_encoder = clip_text_encoder.eval().to(dist_util.dev())
    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.from_numpy(diffusion.betas).to(dist_util.dev()))

    std_volume = torch.tensor(init_volume_grid(bound=bound, num_pts_each_axis=32)).to(torch.float32).to(dist_util.dev()).contiguous()
    mean = torch.load(mean_file).to(torch.float32).to(dist_util.dev())
    std = torch.load(std_file).to(torch.float32).to(dist_util.dev())

    mean = mean.permute(3, 0, 1, 2).requires_grad_(False).contiguous()
    std = std.permute(3, 0, 1, 2).requires_grad_(False).contiguous()

    return {
        'model': model,
        'diffusion': diffusion,
        'clip_text_encoder': clip_text_encoder,
        'noise_schedule': noise_schedule,
        'std_volume': std_volume,
        'mean': mean,
        'std': std,
        'config': model_and_diffusion_config
    }

def rpy2rotations(roll, pitch, yaw):
    """
    Convert roll, pitch, yaw to rotation matrix.
    """
    import numpy as np
    cr, cp, cy = np.cos(roll), np.cos(pitch), np.cos(yaw)
    sr, sp, sy = np.sin(roll), np.sin(pitch), np.sin(yaw)
    return torch.tensor([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]
    ]).cuda()


def create_objects_batched(objects_info, shared_models, max_batch_size=32):
    """Create multiple objects in a single batched diffusion run."""
    if not objects_info:
        return []

    # Split into smaller batches if needed
    all_created_objects = []
    model = shared_models["model"]
    clip_text_encoder = shared_models["clip_text_encoder"]
    noise_schedule = shared_models["noise_schedule"]
    std_volume = shared_models["std_volume"]
    mean = shared_models["mean"]
    std = shared_models["std"]
    model_and_diffusion_config = shared_models["config"]
    cond_gen = text_cond = True
    image_size = model_and_diffusion_config["model"]["image_size"]

    for i in range(0, len(objects_info), max_batch_size):
        batch_info = objects_info[i : i + max_batch_size]
        batch_size = len(batch_info)
        print(f"Creating batch of {batch_size} objects...")

        # Encode all text prompts
        category_names = [obj_info["category_name"] for obj_info in batch_info]
        batched_text_features = clip_text_encoder.encode(category_names)
        condition, unconditional_condition = {}, {}
        if text_cond:
            condition["cond_text"] = batched_text_features
            unconditional_condition["cond_text"] = torch.zeros_like(
                batched_text_features
            )

        model_fn = model_wrapper(
            model,
            noise_schedule,
            model_type=MODEL_TYPES[
                model_and_diffusion_config["diffusion"]["predict_type"]
            ],
            model_kwargs={},
            guidance_type="uncond" if not cond_gen else "classifier-free",
            guidance_scale=3.5,
            condition=None if not cond_gen else condition,
            unconditional_condition=None if not cond_gen else unconditional_condition,
        )
        dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")

        with torch.no_grad():
            batched_noise = torch.randn(
                (
                    len(batch_info),
                    model_and_diffusion_config["model"]["in_channels"],
                    image_size,
                    image_size,
                    image_size,
                ),
                device=dist_util.dev(),
            )

            with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                samples = dpm_solver.sample(
                    x=batched_noise,
                    steps=100,
                    t_start=1.0,
                    t_end=1 / 1000,
                    order=2,  # Use order 2 for batched processing
                    skip_type="time_uniform",
                    method="adaptive" if text_cond else "multistep",
                )
            samples_denorm = samples * std.unsqueeze(0) + mean.unsqueeze(0)

        # Create Object3D instances for each sample in this batch
        for i, obj_info in enumerate(batch_info):
            obj = Object3D(
                id=obj_info["obj_id"],
                size=torch.tensor(obj_info["size"]).cuda(),
                category_name=obj_info["category_name"],
                initial_gs=parse_volume_data(
                    samples_denorm[i], std_volume, active_sh_degree=0
                ),
            )
            all_created_objects.append(obj)

    return all_created_objects

class Object3D:

    def __init__(self, id, size, category_name, initial_gs):
        self._id = id
        self._size = torch.tensor(size).cuda()
        self._text = category_name
        self._initial_gs = initial_gs
        self.centerlize_and_scale_initial_gs()
        print(f"\tAdd {category_name} {id}")

    def transform_gs(self, transformation_matrix):
        '''
        Args:
            transformation_matrix: 4x4 transformation matrix
        '''
        gs = deepcopy(self._initial_gs)
        gs = self.rotate_gs(transformation_matrix[:3, :3], gs)
        gs = self.translate_gs(transformation_matrix[:3, 3], gs)
        return gs

    def rotate_gs(self, rotation_matrix, gs):
        '''
        Args:
            rotation_matrix: 3x3 rotation matrix
        '''
        rotated_xyz = gs['xyz'].double() @ rotation_matrix.T
        rotated_rotations = F.normalize(quat_multiply(
            torch.tensor(Quaternion(matrix=rotation_matrix.cpu().numpy()).elements).cuda(),
            gs['rots'],
        ))
        gs['xyz'] = rotated_xyz.to(torch.float32)
        gs['rots'] = rotated_rotations.to(torch.float32)
        return gs

    def translate_gs(self, translation, gs):
        '''
        Args:
            translation: 3 translation vector
        '''
        gs['xyz'] += translation
        return gs

    def centerlize_and_scale_initial_gs(self):
        # self._initial_gs = self.rotate_gs(rpy2rotations(0, 0, -np.pi/2), self._initial_gs)
        valid_mask = self._initial_gs['opacities'].squeeze()!=0
        x_min, x_max = self._initial_gs['xyz'][valid_mask, 0].min(), self._initial_gs['xyz'][valid_mask, 0].max()
        y_min, y_max = self._initial_gs['xyz'][valid_mask, 1].min(), self._initial_gs['xyz'][valid_mask, 1].max()
        z_min, z_max = self._initial_gs['xyz'][valid_mask, 2].min(), self._initial_gs['xyz'][valid_mask, 2].max()
        # TODO: some splats are opaque
        center = torch.tensor([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]).cuda()
        self._initial_gs['xyz'] -= center
        initial_size = torch.tensor([x_max - x_min, y_max - y_min, z_max - z_min]).cuda()
        # swap self._size[0] and self._size[1]
        self._size[[0, 1]] = self._size[[1, 0]]
        scale = self._size / initial_size
        self._initial_gs['xyz'] *= scale
        self._initial_gs['scales'] *= scale

def get_obj_to_cam_front(rotation, translation, cam_front_to_world):
    '''
    Args:
        rotation: 3x3 rotation matrix
        translation: 3 translation vector
        cam_front_to_world: 4x4 camera front to world matrix
    '''
    obj_to_world = np.eye(4)
    obj_to_world[:3, :3] = Quaternion(rotation).rotation_matrix
    obj_to_world[:3, 3] = translation
    obj_to_cam_front = np.linalg.inv(cam_front_to_world) @ obj_to_world
    return torch.tensor(obj_to_cam_front).cuda()

def all_to_camera_front(nusc, cam_calib_tokens):
    '''
    Args:
        cam_calib_tokens: list of camera tokens
    '''
    cam_front_calib_token = cam_calib_tokens["CAM_FRONT"]
    cam_front_calib_data = nusc.get('calibrated_sensor', cam_front_calib_token)
    cam_front_to_ego = np.eye(4)
    cam_front_to_ego[:3, :3] = Quaternion(cam_front_calib_data['rotation']).rotation_matrix
    cam_front_to_ego[:3, 3] = np.array(cam_front_calib_data['translation'])
    ret = {}
    for cam in cam_calib_tokens.keys():
        if cam == "CAM_FRONT":
            ret[cam] = np.eye(4)
        else:
            cam_to_ego = np.eye(4)
            calib_token = cam_calib_tokens[cam]
            calib_data = nusc.get('calibrated_sensor', calib_token)
            cam_to_ego[:3, :3] = Quaternion(calib_data['rotation']).rotation_matrix
            cam_to_ego[:3, 3] = np.array(calib_data['translation'])
            ret[cam] = np.linalg.inv(cam_front_to_ego) @ cam_to_ego
    return ret, cam_front_to_ego

def render_gaussian(gaussian, extrinsics, intrinsics, width=533, height=300):
    extrinsics = torch.tensor(extrinsics).float().cuda()
    intrinsics = torch.tensor(intrinsics).float().cuda()
    intrinsics[0] *= width / 1600
    intrinsics[1] *= height / 900
    means = gaussian["xyz"]
    f_dc = gaussian["shs"].squeeze()
    opacities = gaussian["opacities"]
    scales = gaussian["scales"]
    rotations = gaussian["rots"]

    rgbs = torch.sigmoid(f_dc)
    renders, _, _ = rasterization(
        means=means,
        quats=rotations,
        scales=scales,
        opacities=opacities.squeeze(),
        colors=rgbs,
        viewmats=torch.linalg.inv(extrinsics)[None, ...],  # [C, 4, 4]
        Ks=intrinsics[None, ...],  # [C, 3, 3]
        width=width,
        height=height,
        packed=False,
        absgrad=True,
        sparse_grad=False,
        rasterize_mode="classic",
        near_plane=0.1,
        far_plane=10000000000.0,
        render_mode="RGB",
        radius_clip=0.,
        backgrounds=torch.ones(1, 3).cuda(),
    )
    renders = torch.clamp(renders, max=1.0)
    return renders

def render(gs, intrinsics, extrinsics, save_path='render.png'):
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    images = []
    for i in range(len(cams)):
        intrinsic = intrinsics[cams[i]]
        extrinsic = extrinsics[cams[i]]
        img = render_gaussian(gs, extrinsic, intrinsic)
        img = img[0].detach().cpu().numpy()
        img = Image.fromarray((img * 255).astype(np.uint8))
        images.append(img)
    # Arrange images in two rows of three using PIL
    widths = [img.size[0] for img in images]
    heights = [img.size[1] for img in images]
    row_height = max(heights)
    row1_width = sum(widths[:3])
    row2_width = sum(widths[3:])
    final_width = max(row1_width, row2_width)
    final_height = row_height * 2
    final_img = Image.new('RGB', (final_width, final_height))
    # Paste images directly into final_img
    x_offset = 0
    for img in images[:3]:
        final_img.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    x_offset = 0
    for img in images[3:]:
        final_img.paste(img, (x_offset, row_height))
        x_offset += img.size[0]
    final_img.save(save_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Render 3D objects in a scene.")
    parser.add_argument("--scene-idx", type=int, default=0, help="Index of the scene to render.")
    return parser.parse_args()

def main():
    nusc = NuScenes(version="v1.0-mini", dataroot="/data/nuscenes", verbose=True)
    print("Multiple GPUs not yet supported...")
    args = parse_args()
    scene = nusc.scene[args.scene_idx]
    sample = None
    objects = {}
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

    # Initialize shared models once
    shared_models = initialize_shared_models()

    for t in range(scene['nbr_samples']):
        gs = None
        print(f"Processing frame {t}")
        sample_token = scene['first_sample_token'] if sample is None else sample['next']
        sample = nusc.get('sample', sample_token)
        cam_tokens = {cam: sample['data'][cam] for cam in cams}
        cam_data = {cam: nusc.get('sample_data', cam_tokens[cam]) for cam in cam_tokens}
        ego_pose = nusc.get('ego_pose', cam_data['CAM_FRONT']['ego_pose_token'])
        if t == 0:
            cam_calib_tokens = {cam: cam_data[cam]['calibrated_sensor_token'] for cam in cam_tokens}
            intrinsics = {cam: np.array(nusc.get('calibrated_sensor', cam_calib_tokens[cam])['camera_intrinsic']) for cam in cam_tokens}
            extrinsics, cam_front_to_ego = all_to_camera_front(nusc, cam_calib_tokens)
        ego_to_world = np.eye(4)
        ego_to_world[:3, :3] = Quaternion(ego_pose['rotation']).rotation_matrix
        ego_to_world[:3, 3] = ego_pose['translation']
        cam_front_to_world = ego_to_world @ cam_front_to_ego

        # Collect new objects to create in parallel
        new_objects_to_create = []
        existing_objects = []

        for i, ann_token in enumerate(sample['anns']):
            ann = nusc.get('sample_annotation', ann_token)
            size = ann['size']
            inst_token = ann['instance_token']

            if inst_token not in objects:
                # Queue for parallel creation
                new_objects_to_create.append(
                    {
                        "inst_token": inst_token,
                        "obj_id": len(objects) + len(new_objects_to_create),
                        "size": size,
                        "category_name": ann["category_name"],
                        "ann": ann,
                    }
                )
            else:
                # Existing object, process immediately
                obj = objects[inst_token]
                obj_to_cam_front = get_obj_to_cam_front(
                    ann["rotation"], ann["translation"], cam_front_to_world
                )
                obj_gs = obj.transform_gs(obj_to_cam_front)
                existing_objects.append(obj_gs)

        # Create new objects in batch
        new_objects = []
        if new_objects_to_create:
            created_objects = create_objects_batched(
                new_objects_to_create, shared_models
            )

            # Store created objects and transform them
            for obj, obj_info in zip(created_objects, new_objects_to_create):
                objects[obj_info["inst_token"]] = obj

                # Transform the new object
                obj_to_cam_front = get_obj_to_cam_front(
                    obj_info["ann"]["rotation"],
                    obj_info["ann"]["translation"],
                    cam_front_to_world,
                )
                obj_gs = obj.transform_gs(obj_to_cam_front)
                new_objects.append(obj_gs)

        # Combine all gaussian splats
        all_gs = existing_objects + new_objects
        if all_gs:
            gs = all_gs[0]
            for obj_gs in all_gs[1:]:
                for key in gs.keys():
                    gs[key] = torch.vstack([gs[key], obj_gs[key]])
        else:
            gs = None

        # Save the rendered image for the current frame
        os.makedirs(f"videos/{args.scene_idx}/rendered_images", exist_ok=True)
        image_path = f"videos/{args.scene_idx}/rendered_images/frame_{t:04d}.png"
        render(gs, intrinsics, extrinsics, save_path=image_path)

    # Generate a video from the saved frames
    frame_paths = sorted(glob.glob(f"videos/{args.scene_idx}/rendered_images/frame_*.png"))
    with imageio.get_writer(f'videos/{args.scene_idx}/objaverse.mp4', fps=2) as video_writer:
        for frame_path in frame_paths:
            frame = imageio.imread(frame_path)
            video_writer.append_data(frame)

    # Clean up the frame images and remove the folder
    for frame_path in frame_paths:
        os.remove(frame_path)
    os.rmdir(f"videos/{args.scene_idx}/rendered_images")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
