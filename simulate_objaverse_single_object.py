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
import matplotlib.pyplot as plt
from dataset.nuscenes.dataset_nuscenes import NuScenesObjects
from pyquaternion import Quaternion
from copy import deepcopy
from gsplat.rendering import rasterization


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

def rotation_matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts a 3x3 rotation matrix to a quaternion.

    Parameters:
    matrix (torch.Tensor): A 3x3 rotation matrix.

    Returns:
    torch.Tensor: A quaternion [w, x, y, z].
    """
    if matrix.shape != (3, 3):
        raise ValueError("Input matrix must be 3x3.")

    # Compute the trace of the matrix
    trace = torch.trace(matrix)

    if trace > 0:
        S = 2.0 * torch.sqrt(trace + 1.0)
        w = 0.25 * S
        x = (matrix[2, 1] - matrix[1, 2]) / S
        y = (matrix[0, 2] - matrix[2, 0]) / S
        z = (matrix[1, 0] - matrix[0, 1]) / S
    elif (matrix[0, 0] > matrix[1, 1]) and (matrix[0, 0] > matrix[2, 2]):
        S = 2.0 * torch.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        w = (matrix[2, 1] - matrix[1, 2]) / S
        x = 0.25 * S
        y = (matrix[0, 1] + matrix[1, 0]) / S
        z = (matrix[0, 2] + matrix[2, 0]) / S
    elif matrix[1, 1] > matrix[2, 2]:
        S = 2.0 * torch.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        w = (matrix[0, 2] - matrix[2, 0]) / S
        x = (matrix[0, 1] + matrix[1, 0]) / S
        y = 0.25 * S
        z = (matrix[1, 2] + matrix[2, 1]) / S
    else:
        S = 2.0 * torch.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        w = (matrix[1, 0] - matrix[0, 1]) / S
        x = (matrix[0, 2] + matrix[2, 0]) / S
        y = (matrix[1, 2] + matrix[2, 1]) / S
        z = 0.25 * S

    return torch.tensor([w, x, y, z])

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
    
class Object3D:
    def __init__(self, id, size, category_name, ckpt=None):
        self._id = id
        self._size = torch.tensor(size).cuda()
        print(f"\tAdd {category_name} {id}")
        self._text = category_name
        self.ckpt = ckpt
        self.generator = torch.Generator(device="cuda").manual_seed(id)
        self.generate_initial_gs()

    def generate_initial_gs(self):

        model_and_diffusion_config = OmegaConf.load("configs/objaverse_text_cond.yml")
        downloaded_files = download_model_files("objaverse_v1.1")

        ckpt = downloaded_files["ckpt"] if self.ckpt is None else self.ckpt
        mean_file = downloaded_files["mean"]
        std_file = downloaded_files["std"]
        bound = downloaded_files["bound"]
        cond_gen = text_cond =True

        dist_util.setup_dist()
        torch.cuda.set_device(dist_util.dev())
        # seed_everything(dist.get_rank())

        model_and_diffusion_config['model']['precision'] = "32"
        model = UNetModel(**model_and_diffusion_config['model'])

        diffusion = create_gaussian_diffusion(**model_and_diffusion_config['diffusion'])
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.to(dist_util.dev())
        model.eval()

        clip_text_encoder = FrozenCLIPEmbedder()
        clip_text_encoder = clip_text_encoder.eval().to(dist_util.dev())
        text_features = clip_text_encoder.encode(self._text)
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.from_numpy(diffusion.betas).to(dist_util.dev()))

        std_volume = torch.tensor(init_volume_grid(bound=bound, num_pts_each_axis=32)).to(torch.float32).to(dist_util.dev()).contiguous()
        mean = torch.load(mean_file).to(torch.float32).to(dist_util.dev())
        std = torch.load(std_file).to(torch.float32).to(dist_util.dev())

        mean = mean.permute(3, 0, 1, 2).requires_grad_(False).contiguous()
        std = std.permute(3, 0, 1, 2).requires_grad_(False).contiguous()

        image_size = model_and_diffusion_config['model']['image_size']
        sample_shape = (1, model_and_diffusion_config['model']['in_channels'], image_size, image_size, image_size)

        condition, unconditional_condition = {}, {}
        if text_cond:
            condition['cond_text'] = text_features
            unconditional_condition['cond_text'] = torch.zeros_like(text_features)

        model_fn = model_wrapper(
            model,
            noise_schedule,
            model_type=MODEL_TYPES[model_and_diffusion_config['diffusion']['predict_type']],
            model_kwargs={},
            guidance_type='uncond' if not cond_gen else 'classifier-free',
            guidance_scale=3.5,
            condition=None if not cond_gen else condition,
            unconditional_condition=None if not cond_gen else unconditional_condition,
        )
        dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type='dpmsolver++')

        with torch.no_grad():
            noise = torch.randn(sample_shape, device=dist_util.dev(), generator=self.generator)
            with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                samples = dpm_solver.sample(
                    x=noise,
                    steps=100,
                    t_start=1.0,
                    t_end=1/1000,
                    order=3 if not cond_gen else 2,
                    skip_type='time_uniform',
                    method='adaptive' if text_cond else 'multistep',
                )
            samples_denorm = samples * std + mean
        self._initial_gs = parse_volume_data(samples_denorm[0], std_volume, active_sh_degree=0)
        # TODO: Take bounding box as diffusion input
        self.centerlize_and_scale_initial_gs()

    def transform_gs(self, transformation_matrix):
        '''
        Args:
            transformation_matrix: 4x4 transformation matrix
        '''
        gs = deepcopy(self._initial_gs)
        transformation_matrix = torch.tensor(transformation_matrix).cuda()
        gs = self.rotate_gs(transformation_matrix[:3, :3], gs)
        gs = self.translate_gs(transformation_matrix[:3, 3], gs)
        return gs
    
    def rotate_gs(self, rotation_matrix, gs):
        '''
        Args:
            rotation_matrix: 3x3 rotation matrix
        '''
        rotated_xyz = gs['xyz'] @ rotation_matrix.T
        rotated_rotations = F.normalize(quat_multiply(
            rotation_matrix_to_quaternion(rotation_matrix).cuda(),
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
        center = torch.tensor([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]).cuda()
        self._initial_gs['xyz'] -= center
        initial_size = torch.tensor([x_max - x_min, y_max - y_min, z_max - z_min]).cuda()
        # swap self._size[0] and self._size[1]
        self._size[[0, 1]] = self._size[[1, 0]]
        scale = self._size / initial_size
        self._initial_gs['xyz'] *= scale
        self._initial_gs['scales'] *= scale

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

def render(gs, cam, intrinsics, extrinsics, save_path='render.png'):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    for i, ax in enumerate(axes.flat):
        if cams[i] != cam:
            ax.imshow(np.ones((300, 533, 3)))
            ax.set_title(cams[i])
            ax.axis('off')
            continue
        intrinsic = intrinsics
        extrinsic = extrinsics
        img = render_gaussian(gs, extrinsic, intrinsic)
        ax.imshow(img[0].detach().cpu().numpy())
        ax.set_title(cams[i])
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Render 3D objects in a scene.")
    parser.add_argument("--ckpt", default=None, help="Path to the checkpoint file.")
    parser.add_argument("--scene_idx", '-s', type=int, default=0, help="Index of the scene to render.")
    parser.add_argument("--ann_idx", '-a', type=int, default=11, help="Index of the scene to render.")
    return parser.parse_args()

def main():
    args = parse_args()
    nusc = NuScenesObjects(version='v1.0-mini', data_root='/storage_local/kwang/nuscenes/raw', ins_seg_root='/storage_local/kwang/nuscenes/insSeg', split=[args.scene_idx], verbose=True)
    print("Multiplt GPUs not yet supported...")
    nusc.vis(args.ann_idx)
    data, render_params = nusc[args.ann_idx]
    
    obj = Object3D(0, render_params['size'], render_params['category'], args.ckpt)
    obj_gs = obj.transform_gs(render_params['obj_to_cam_front'])
    # Save the rendered image for the current frame
    render(obj_gs, render_params['cam'], render_params['intrinsics'], render_params['extrinsics'])

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
