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
from model.smpl import SMPLinGaussianCube
from model.lora_unet import convert_unet_to_lora
from model.clip import FrozenCLIPEmbedder
from model.dpmsolver import NoiseScheduleVP, model_wrapper, DPM_Solver, expand_dims
from utils import dist_util, logger
from utils.prompt_util import generate_human_prompt
from utils.script_util import create_gaussian_diffusion, init_volume_grid, build_single_viewpoint_cam
from dataset.dataset_render import load_data
from gaussian_renderer import parse_volume_data
import imageio.v2 as imageio
from tqdm import trange
import glob
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from copy import deepcopy
from gsplat.rendering import rasterization
from PIL import Image
from scipy.spatial.transform import Rotation as R, Slerp


MODEL_TYPES = {
    'xstart': 'x_start',
    'v': 'v',
    'eps': 'noise',
}

# Model repository mapping
MODEL_REPOS = {
    "objaverse_v1.1": {
        "repo_id": "BwZhang/GaussianCube-Objaverse",
        "revision": "main",
        "model_path": "v1.1/objaverse_ckpt.pt",
        "mean_path": "v1.1/mean.pt",
        "std_path": "v1.1/std.pt",
        "bound": 0.5,
    },
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


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def quat_multiply(quaternion0, quaternion1):
    w0, x0, y0, z0 = torch.split(quaternion0, 1, dim=-1)
    w1, x1, y1, z1 = torch.split(quaternion1, 1, dim=-1)
    w = -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0
    x = x1 * w0 - y1 * z0 + z1 * y0 + w1 * x0
    y = x1 * z0 + y1 * w0 - z1 * x0 + w1 * y0
    z = -x1 * y0 + y1 * x0 + z1 * w0 + w1 * z0

    return torch.concat((w, x, y, z), dim=-1)


def initialize_shared_models(smpl_lora_path):
    """Initialize models once to be shared across all objects."""

    model_and_diffusion_config = OmegaConf.load("configs/objaverse_text_cond.yml")
    smpl_config = OmegaConf.load("configs/finetune_smpl.yml")
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
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    smpl_model = convert_unet_to_lora(
        model, **smpl_config["lora"], **model_and_diffusion_config["model"]
    )
    smpl_model.load_lora_weights(smpl_lora_path)
    smpl_model.to(dist_util.dev())
    smpl_model.eval()

    clip_text_encoder = FrozenCLIPEmbedder()
    clip_text_encoder = clip_text_encoder.eval().to(dist_util.dev())
    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.from_numpy(diffusion.betas).to(dist_util.dev()))

    std_volume = torch.tensor(init_volume_grid(bound=bound, num_pts_each_axis=32)).to(torch.float32).to(dist_util.dev()).contiguous()
    mean = (
        torch.load(mean_file, weights_only=True).to(torch.float32).to(dist_util.dev())
    )
    std = torch.load(std_file, weights_only=True).to(torch.float32).to(dist_util.dev())

    mean = mean.permute(3, 0, 1, 2).requires_grad_(False).contiguous()
    std = std.permute(3, 0, 1, 2).requires_grad_(False).contiguous()

    neutral_human_model = SMPLinGaussianCube(
        "smpl/SMPL_NEUTRAL.pkl",
        std_volume=std_volume,
        gc_mean=mean,
        gc_std=std,
        device=dist_util.dev(),
    )

    return {
        "smpl_model": smpl_model,
        "neutral_human_model": neutral_human_model,
        "diffusion": diffusion,
        "clip_text_encoder": clip_text_encoder,
        "noise_schedule": noise_schedule,
        "std_volume": std_volume,
        "mean": mean,
        "std": std,
        "config": model_and_diffusion_config,
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


def create_humans_batched(objects_info, shared_models, max_batch_size=8):
    """Create multiple objects in a single batched diffusion run."""
    if not objects_info:
        return []
    # Split into smaller batches if needed
    all_created_objects = []
    model = shared_models["smpl_model"]
    clip_text_encoder = shared_models["clip_text_encoder"]
    std_volume = shared_models["std_volume"]
    mean = shared_models["mean"]
    std = shared_models["std"]
    model_and_diffusion_config = shared_models["config"]
    image_size = model_and_diffusion_config["model"]["image_size"]
    print(f"Creating {len(objects_info)} humans in batches of {max_batch_size}...")
    for i in range(0, len(objects_info), max_batch_size):
        human_models = []
        batch_info = objects_info[i : i + max_batch_size]
        batch_size = len(batch_info)
        print(f"    Creating batch of {batch_size} objects...")

        # Encode all text prompts
        prompts = [obj_info["prompt"] for obj_info in batch_info]
        batched_text_features = clip_text_encoder.encode(prompts)
        condition = {"cond_text": batched_text_features}
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
        with torch.no_grad():
            model_output = model(
                batched_noise,
                torch.tensor([999] * len(batch_info), device=dist_util.dev()),
                **condition,
            )
        samples, _ = torch.chunk(model_output, 2, dim=1)
        samples_denorm = samples * std.unsqueeze(0) + mean.unsqueeze(0)
        for b in range(len(batch_info)):
            if "woman" in batch_info[b]["prompt"]:
                human_model = SMPLinGaussianCube(
                    "smpl/SMPL_FEMALE.pkl",
                    std_volume=std_volume,
                    gc_mean=mean,
                    gc_std=std,
                    device=dist_util.dev(),
                    betas=torch.randn([1, 10]),
                    assign=False,
                )
            else:
                human_model = SMPLinGaussianCube(
                    "smpl/SMPL_MALE.pkl",
                    std_volume=std_volume,
                    gc_mean=mean,
                    gc_std=std,
                    device=dist_util.dev(),
                    betas=torch.randn([1, 10]),
                    assign=False,
                )
            human_model.update_rest_attributes(
                samples_denorm[b],
                assignments=shared_models["neutral_human_model"].assignments,
            )
            human_models.append(human_model)
        # Create Object3D instances for each sample in this batch
        for i, obj_info in enumerate(batch_info):
            obj = Object3D(
                size=torch.tensor(obj_info["size"]).cuda(),
                prompt=obj_info["prompt"],
                human_model=human_models[i],
                std_volume=std_volume,
                inversed_assignments=shared_models[
                    "neutral_human_model"
                ].inversed_assignments,
                # initial_gs=parse_volume_data(
                #     human_models[i].to_x0_denorm(shared_models["neutral_human_model"].inversed_assignments), std_volume, active_sh_degree=0
                # ),
            )
            all_created_objects.append(obj)
    return all_created_objects


class Object3D:
    poses = {
        "stand_to_walk": np.load("smpl/B1 - stand to walk_poses.npz")["poses"][:, :66],
        "walk": np.load("smpl/B3 - walk1_poses.npz")["poses"][:, :66],
    }

    def __init__(
        self,
        size,
        prompt,
        human_model,
        std_volume,
        inversed_assignments,
        pose_key="walk",
        initial_pose_idx=None,
    ):
        self._size = size.clone().detach().cuda()
        self._size[[0, 1]] = self._size[[1, 0]]
        self._text = prompt
        self._human_model = human_model
        self._std_volume = std_volume
        self._inversed_assignments = inversed_assignments
        self._pose_key = pose_key
        self._initial_pose_idx = (
            initial_pose_idx
            if initial_pose_idx is not None
            else random.randint(0, self.poses[self._pose_key].shape[0] - 1)
        )
        initial_orient = torch.zeros([1, 3], device=dist_util.dev())  # global_orient
        initial_orient[0] = torch.from_numpy(
            self.poses[self._pose_key][self._initial_pose_idx, :3]
        ).to(dist_util.dev())
        initial_pose = torch.zeros([1, 69], device=dist_util.dev())  # 23*3 axis-angle
        initial_pose[0, :63] = torch.from_numpy(
            self.poses[self._pose_key][self._initial_pose_idx, 3:66]
        ).to(dist_util.dev())
        self._human_model.apply_pose(
            body_pose=initial_pose, global_orient=initial_orient
        )
        print(f"\tAdd {prompt}")

    @property
    def _gs_in_gaussiancube(self):
        return parse_volume_data(
            self._human_model.to_x0_denorm(self._inversed_assignments),
            self._std_volume,
            active_sh_degree=0,
        )

    @property
    def _centeralized_and_scaled_gs_in_gaussiancube(self):
        gs = deepcopy(self._gs_in_gaussiancube)
        if self._pose_key == "walk":
            gs = self.rotate_gs(rpy2rotations(0, 0, 3 * np.pi / 4), gs)
        elif self._pose_key == "stand_to_walk":
            gs = self.rotate_gs(rpy2rotations(0, 0, -np.pi / 2), gs)
        valid_mask = gs["opacities"].squeeze() != 0
        x_min, x_max = gs["xyz"][valid_mask, 0].min(), gs["xyz"][valid_mask, 0].max()
        y_min, y_max = gs["xyz"][valid_mask, 1].min(), gs["xyz"][valid_mask, 1].max()
        z_min, z_max = (
            gs["xyz"][valid_mask, 2].min(),
            gs["xyz"][valid_mask, 2].max(),
        )
        center = torch.tensor(
            [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
        ).cuda()
        gs["xyz"] -= center
        scale = self._size[2] / (z_max - z_min)
        gs["xyz"] *= scale
        gs["scales"] *= scale
        return gs

    def apply_pose(self, delta_t):
        pose_idx = (delta_t + self._initial_pose_idx) % self.poses[
            self._pose_key
        ].shape[0]
        body_pose = torch.zeros([1, 69], device=dist_util.dev())  # 23*3 axis-angle
        body_pose[0, :63] = torch.from_numpy(
            self.poses[self._pose_key][pose_idx, 3:66]
        ).to(dist_util.dev())
        global_orient = torch.zeros([1, 3], device=dist_util.dev())  # global_orient
        global_orient[0] = torch.from_numpy(
            self.poses[self._pose_key][pose_idx, :3]
        ).to(dist_util.dev())
        self._human_model.apply_pose(body_pose=body_pose, global_orient=global_orient)

    def transform_gs(self, transformation_matrix, delta_t):
        '''
        Args:
            transformation_matrix: 4x4 transformation matrix
        '''
        self.apply_pose(delta_t=delta_t)
        gs = deepcopy(self._centeralized_and_scaled_gs_in_gaussiancube)
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


def _orthonormalize(mat: np.ndarray) -> np.ndarray:
    """Project a 3x3 matrix onto the closest valid rotation matrix using SVD."""
    U, _, Vt = np.linalg.svd(mat)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1
        Rm = U @ Vt
    return Rm


def interpolate_transform_matrix(
    mat1: torch.Tensor, mat2: torch.Tensor, alpha: float
) -> torch.Tensor:
    """
    Interpolate between two 4x4 transformation matrices.

    Args:
        mat1, mat2 : torch.Tensor of shape (4,4)
        alpha      : float in [0,1], interpolation parameter
    Returns:
        torch.Tensor of shape (4,4), same dtype/device as inputs
    """
    device = mat1.device
    dtype = mat1.dtype

    # --- Translation ---
    t1 = mat1[:3, 3].cpu().numpy()
    t2 = mat2[:3, 3].cpu().numpy()
    translation = (1 - alpha) * t1 + alpha * t2

    # --- Rotation ---
    r1 = R.from_matrix(mat1[:3, :3].cpu().numpy())
    r2 = R.from_matrix(mat2[:3, :3].cpu().numpy())
    key_times = [0, 1]
    key_rots = R.from_quat([r1.as_quat(), r2.as_quat()])
    slerp = Slerp(key_times, key_rots)
    interp_rot = slerp([alpha]).as_matrix()[0]

    # Re-orthogonalize to avoid pyquaternion ValueError
    interp_rot = _orthonormalize(interp_rot)

    # --- Compose back ---
    result = torch.eye(4, dtype=dtype, device=device)
    result[:3, :3] = torch.tensor(interp_rot, dtype=dtype, device=device)
    result[:3, 3] = torch.tensor(translation, dtype=dtype, device=device)

    return result.double()


def parse_args():
    parser = argparse.ArgumentParser(description="Render 3D objects in a scene.")
    parser.add_argument("--scene-idx", type=int, default=0, help="Index of the scene to render.")
    parser.add_argument(
        "--smpl-lora", type=str, default="lora_ckpts/smpl.pt", help="Path to the smpl LoRA model."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--hz",
        type=int,
        default=2,
        help="Frame rate for the output video.",
        choices=[2 * i for i in range(1, 61)],
    )
    return parser.parse_args()

def main():
    nusc = NuScenes(version="v1.0-mini", dataroot="/data/nuscenes", verbose=True)
    print("Multiple GPUs not yet supported...")
    args = parse_args()
    seed_everything(args.seed)
    scene = nusc.scene[args.scene_idx]
    sample = None
    objects = {}
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

    # Initialize shared models once
    shared_models = initialize_shared_models(args.smpl_lora)

    # Collect new objects to create in parallel
    humans_to_create = []
    existing_humans = []
    annotations_2hz = {}
    for t in range(scene["nbr_samples"]):
        sample_token = scene["first_sample_token"] if sample is None else sample["next"]
        sample = nusc.get("sample", sample_token)
        cam_tokens = {cam: sample["data"][cam] for cam in cams}
        cam_data = {cam: nusc.get("sample_data", cam_tokens[cam]) for cam in cam_tokens}
        ego_pose = nusc.get("ego_pose", cam_data["CAM_FRONT"]["ego_pose_token"])
        if t == 0:
            cam_calib_tokens = {
                cam: cam_data[cam]["calibrated_sensor_token"] for cam in cam_tokens
            }
            intrinsics = {
                cam: np.array(
                    nusc.get("calibrated_sensor", cam_calib_tokens[cam])[
                        "camera_intrinsic"
                    ]
                )
                for cam in cam_tokens
            }
            extrinsics, cam_front_to_ego = all_to_camera_front(nusc, cam_calib_tokens)
        ego_to_world = np.eye(4)
        ego_to_world[:3, :3] = Quaternion(ego_pose["rotation"]).rotation_matrix
        ego_to_world[:3, 3] = ego_pose["translation"]
        cam_front_to_world = ego_to_world @ cam_front_to_ego
        for _, ann_token in enumerate(sample["anns"]):
            ann = nusc.get("sample_annotation", ann_token)
            if not ann["category_name"].startswith("human."):
                continue
            size = ann["size"]
            inst_token = ann["instance_token"]

            if inst_token not in existing_humans:
                # Queue for parallel creation
                humans_to_create.append(
                    {
                        "inst_token": inst_token,
                        "size": size,
                        "prompt": generate_human_prompt()
                    }
                )
                existing_humans.append(inst_token)
            obj_to_cam_front = get_obj_to_cam_front(
                ann["rotation"],
                ann["translation"],
                cam_front_to_world,
            )
            if inst_token in annotations_2hz:
                annotations_2hz[inst_token][t] = obj_to_cam_front
            else:
                annotations_2hz[inst_token] = {t: obj_to_cam_front}
    del existing_humans

    # Create interpolated annotations with higher frame rate
    interpolation_factor = args.hz // 2  # Convert from 2Hz to desired Hz
    annotations_required = {}

    for inst_token in annotations_2hz:
        annotations_required[inst_token] = {}
        time_keys = sorted(annotations_2hz[inst_token].keys())

        # For each pair of consecutive keyframes, interpolate
        for i in range(len(time_keys) - 1):
            t1, t2 = time_keys[i], time_keys[i + 1]
            mat1 = annotations_2hz[inst_token][t1]
            mat2 = annotations_2hz[inst_token][t2]

            # Add the first keyframe
            for interp_step in range(interpolation_factor):
                interp_t = t1 * interpolation_factor + interp_step
                if interp_step == 0:
                    # Use exact keyframe
                    annotations_required[inst_token][interp_t] = mat1
                else:
                    # Interpolate
                    alpha = interp_step / interpolation_factor
                    interp_mat = interpolate_transform_matrix(mat1, mat2, alpha)
                    annotations_required[inst_token][interp_t] = interp_mat

        # Add the last keyframe
        if time_keys:
            last_t = time_keys[-1]
            for interp_step in range(interpolation_factor):
                interp_t = last_t * interpolation_factor + interp_step
                if interp_step == 0:
                    annotations_required[inst_token][interp_t] = annotations_2hz[
                        inst_token
                    ][last_t]
                else:
                    # For the last frame, just repeat the last pose
                    annotations_required[inst_token][interp_t] = annotations_2hz[
                        inst_token
                    ][last_t]

    created_humans = create_humans_batched(humans_to_create, shared_models)
    # Store created objects and transform them
    for obj, obj_info in zip(created_humans, humans_to_create):
        objects[obj_info["inst_token"]] = obj

    for t in trange(scene["nbr_samples"] * interpolation_factor):
        gs = None
        current_smpl_gs = []
        for inst_token in objects.keys():
            if t not in annotations_required[inst_token]:
                continue
            obj = objects[inst_token]
            obj_to_cam_front = annotations_required[inst_token][t]
            obj_gs = obj.transform_gs(
                obj_to_cam_front, (t * 120) // args.hz
            )  # smpl poses are at 120hz
            current_smpl_gs.append(obj_gs)

        if current_smpl_gs:
            gs = current_smpl_gs[0]
            for obj_gs in current_smpl_gs[1:]:
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
    with imageio.get_writer(
        f"videos/{args.scene_idx}/objects.mp4", fps=args.hz
    ) as video_writer:
        for frame_path in frame_paths:
            frame = imageio.imread(frame_path)
            video_writer.append_data(frame)

    # Clean up the frame images and remove the folder
    for frame_path in frame_paths:
        os.remove(frame_path)
    os.rmdir(f"videos/{args.scene_idx}/rendered_images")

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
