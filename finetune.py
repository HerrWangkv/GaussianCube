"""
LoRA fine-tuning for GaussianCube using SDS loss from Stable Diffusion.

This script implements LoRA-based fine-tuning of pretrained GaussianCube models
using Score Distillation Sampling (SDS) from Stable Diffusion 2.1.
"""

import argparse
from tqdm import trange
import copy
import os
import time
import glob
import imageio
import contextlib
import numpy as np
import torch as th
import torch.nn as nn
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from dataset.dataset_render import orbit_camera, load_cam
from utils import dist_util, logger
from utils.fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)

from utils.script_util import build_single_viewpoint_cam, init_volume_grid, create_gaussian_diffusion
from model.nn import update_ema
from model.resample import UniformSampler
from model.gaussian_diffusion import ModelMeanType
from model.lora_unet import convert_unet_to_lora, LoRAUNetModel
from model.clip import FrozenCLIPEmbedder
from model.dpmsolver import NoiseScheduleVP, model_wrapper, DPM_Solver
from model.unet import UNetModel
from utils.refiner_utils import StableDiffusionXLRefiner
from gaussian_renderer import render

# Initial loss scale for FP16 training
INITIAL_LOG_LOSS_SCALE = 20.0

# Model repository mapping (from inference.py)
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


def ignore_stderr(func):
    def wrapper(*args, **kwargs):
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            return func(*args, **kwargs)

    return wrapper


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


class LoRAFinetuneLoop:
    """
    LoRA fine-tuning loop for GaussianCube using SDS loss.
    
    This class handles the fine-tuning of pretrained GaussianCube models
    with LoRA adapters, guided by SDS loss from Stable Diffusion.
    """

    def __init__(
        self,
        model,
        diffusion,
        # Training configuration
        batch_size=1,
        lr=5e-5,
        max_steps=50000,
        use_fp16=True,
        resume_checkpoint=None,
        prompt_file=None,
        use_tensorboard=True,
        ema_rate=0.9999,
        weight_decay=0.0,
        max_grad_norm=1.0,
        fp16_scale_growth=1e-3,
        # Data configuration
        mean_file=None,
        std_file=None,
        bound=0.5,
        num_pts_each_axis=32,
        # Logging configuration
        log_interval=100,
        save_interval=1000,
        image_save_interval=100,
        # LoRA configuration
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_target_modules=None,
        lora_exclude_modules=None,
        # Guidance configuration
        timestep_range=(0.3, 0.7),
        strength=0.5,
        guidance_scale=10.0,
        vram_O=True,
        render_views=4,
        elevation_range=(-10, 10),
        fovx_range=(0.8, 1.2),
        cam_radius_range=(1.0, 3.0),
        active_sh_degree=0,
        white_background=True,
        **kwargs,
    ):
        """
        Initialize the LoRA fine-tuning loop.

        Args:
            model: Pretrained GaussianCube model
            diffusion: Gaussian diffusion process
            batch_size: Training batch size (must be 1)
            lr: Learning rate for LoRA parameters
            ema_rate: EMA rate(s) for model averaging
            log_interval: Logging interval
            save_interval: Model saving interval
            resume_checkpoint: Path to resume checkpoint
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha scaling
            lora_dropout: LoRA dropout rate
            lora_target_modules: Target modules for LoRA adaptation
            sd_version: Stable Diffusion version
            guidance_scale: guidance scale
            prompt_file: File containing training prompts
            timestep_range: Range (min, max) for random intermediate timestep selection
            image_save_interval: Interval for saving training images
            ... (other parameters similar to TrainLoop)
        """
        # Convert model to LoRA-enabled version
        print("Converting model to LoRA-enabled version...")
        self.lora_model = convert_unet_to_lora(
            model,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            lora_exclude_modules=lora_exclude_modules,
            **kwargs,
        )

        # Store original model reference
        self.original_model = model
        self.model = self.lora_model
        self.diffusion = diffusion

        # Move LoRA model to device (important for distributed training)
        print(f"Moving LoRA model to device: {dist_util.dev()}")
        self.model.to(dist_util.dev())

        # Training configuration
        if batch_size != 1:
            raise NotImplementedError("Batch size greater than 1 is not supported.")
        self.batch_size = batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint  # Store original resume checkpoint path
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = UniformSampler(diffusion.num_timesteps)
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.max_grad_norm = max_grad_norm
        self.noise_schedule = NoiseScheduleVP(
            schedule="discrete",
            betas=th.from_numpy(self.diffusion.betas).to(dist_util.dev()),
        )
        # Inference configuration
        self.image_save_interval = image_save_interval

        # LoRA configuration
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # Guidance configuration
        self.guidance_scale = guidance_scale
        self.render_views = render_views
        self.timestep_range = timestep_range
        self.render_resolution = 1024

        # Training state
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        # Rendering configuration
        self.active_sh_degree = active_sh_degree
        self.bg_color = th.tensor([1,1,1]).to(th.float32).to(dist_util.dev()) if white_background else th.tensor([0,0,0]).to(th.float32).to(dist_util.dev())
        self.std_volume = th.tensor(init_volume_grid(bound=bound, num_pts_each_axis=num_pts_each_axis)).to(th.float32).to(dist_util.dev()).contiguous()
        self.min_elevation, self.max_elevation = elevation_range
        self.min_fovx, self.max_fovx = fovx_range
        self.min_cam_radius, self.max_cam_radius = cam_radius_range

        # Initialize Stable Diffusion for SDS
        print(f"Initializing Stable Diffusion XL Refiner for guidance...")
        self.refiner = StableDiffusionXLRefiner(
            device=dist_util.dev(),
            fp16=use_fp16,
            vram_O=vram_O,
            refiner_strength=strength,
        )

        # Initialize CLIP text encoder for GaussianCube diffusion conditioning
        print("Initializing CLIP text encoder for GaussianCube conditioning...")
        self.clip_text_encoder = FrozenCLIPEmbedder()
        self.clip_text_encoder = self.clip_text_encoder.eval().to(dist_util.dev())

        # Initialize dpm_solver
        # Setup noise schedule

        # Load prompts for SDS guidance
        self.prompts = self._load_prompts(prompt_file)

        # Data normalization (same as original)
        if mean_file is not None and std_file is not None:
            self.mean = th.load(mean_file, weights_only=True).to(th.float32).to(dist_util.dev())
            self.std = th.load(std_file, weights_only=True).to(th.float32).to(dist_util.dev())
            if len(self.mean.shape) == 1:
                print("Using mean shape: ", self.mean.shape)
                self.mean = self.mean.reshape(1, -1, 1, 1, 1).requires_grad_(False).contiguous()
                self.std = self.std.reshape(1, -1, 1, 1, 1).requires_grad_(False).contiguous()
            else:
                print("Using mean shape: ", self.mean.shape)
                self.mean = self.mean.permute(3, 0, 1, 2).requires_grad_(False).unsqueeze(0).contiguous()
                self.std = self.std.permute(3, 0, 1, 2).requires_grad_(False).unsqueeze(0).contiguous()
        else:
            self.mean = th.tensor([0]).to(th.float32).to(dist_util.dev())
            self.std = th.tensor([1]).to(th.float32).to(dist_util.dev())

        # Setup optimizer for LoRA parameters only
        self.optimize_model = self.lora_model

        # Get only LoRA parameters for optimization
        self.lora_params = self.lora_model.get_lora_parameters()
        self.model_params = list(self.lora_params)
        self.master_params = self.model_params

        print(f"LoRA parameters: {sum(p.numel() for p in self.lora_params):,}")
        print(f"Total model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"LoRA ratio: {sum(p.numel() for p in self.lora_params) / sum(p.numel() for p in self.model.parameters()) * 100:.2f}%")

        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        # Load and sync parameters
        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        # Setup optimizer and scheduler
        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)

        # Warmup scheduler
        num_warmup_steps = 1000
        def warmup_lr_schedule(steps):  
            if steps < num_warmup_steps:  
                return float(steps) / float(max(1, num_warmup_steps))  
            return 1.0  

        self.warmup_scheduler = th.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=warmup_lr_schedule)

        # Load optimizer state if resuming
        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        # Setup DDP
        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
            self.original_ddp_model = DDP(
                self.original_model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
            self.original_ddp_model = self.original_model

        # Tensorboard setup
        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard and dist.get_rank() == 0:
            self.writer = logger.Visualizer(os.path.join(logger.get_dir(), 'tf_events'))

    def _load_prompts(self, prompt_file):
        """Load training prompts from file or use default prompts."""
        if prompt_file and os.path.exists(prompt_file):
            with open(prompt_file, 'r') as f:
                prompts = [line.strip() for line in f.readlines() if line.strip()]
            print(f"Loaded {len(prompts)} prompts from {prompt_file}")
        else:
            raise FileNotFoundError("Prompt file not found, and no default prompts provided.")

        return prompts

    def _load_and_sync_parameters(self):
        """Load checkpoint and sync parameters across processes."""
        # Process resume checkpoint path (only for LoRA checkpoints, not base model)
        processed_resume_checkpoint = find_resume_checkpoint(self.resume_checkpoint)

        if processed_resume_checkpoint:
            print("resume checkpoint: ", processed_resume_checkpoint)
            self.resume_step = parse_resume_step_from_filename(processed_resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading LoRA model from checkpoint: {processed_resume_checkpoint}...")

                # Load state dict
                checkpoint = th.load(processed_resume_checkpoint, map_location="cpu", weights_only=True)

                # Check if this is a LoRA checkpoint
                if any('lora' in key for key in checkpoint.keys()):
                    # This is a LoRA checkpoint, load LoRA weights
                    self.lora_model.load_lora_weights(processed_resume_checkpoint)
                    print("Loaded LoRA checkpoint")
                else:
                    # This should not happen in SDS fine-tuning, but handle gracefully
                    logger.warn("Resume checkpoint does not contain LoRA weights. Skipping...")

        # Sync parameters across processes
        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        """Load EMA parameters from checkpoint."""
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = th.load(ema_checkpoint, map_location=dist_util.dev(), weights_only=True)
                ema_params = self._state_dict_to_master_params(state_dict)

        return ema_params

    def _load_optimizer_state(self):
        """Load optimizer state from checkpoint."""
        main_checkpoint = self.resume_checkpoint
        opt_checkpoint = os.path.join(
            os.path.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if os.path.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            try:
                # Try weights_only first for security
                state_dict = th.load(opt_checkpoint, map_location="cpu", weights_only=True)
            except:
                # Fall back to full load if weights_only fails (optimizer states may need it)
                state_dict = th.load(opt_checkpoint, map_location="cpu", weights_only=False)
            try:
                self.opt.load_state_dict(state_dict)
            except:
                pass

    def _setup_fp16(self):
        """Setup FP16 training."""
        self.master_params = make_master_params(self.model_params)

    def run_loop(self):
        """Main training loop."""
        print(f"Starting LoRA fine-tuning with SDXL guidance...")
        print(f"LoRA rank: {self.lora_rank}, alpha: {self.lora_alpha}")
        print(f"Guidance scale: {self.guidance_scale}")

        if dist.get_rank() == 0:
            iterator = trange(
                self.step, self.max_steps + 1, desc="Training", dynamic_ncols=True
            )
        else:
            iterator = range(self.step, self.max_steps + 1)
        for _ in iterator:
            self.run_step()
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
            self.step += 1

        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self):
        """Run one training step."""
        start_time = time.time()
        self.forward_backward()
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        step_time = time.time() - start_time
        logger.logkv_mean("step_time", step_time)
        self.log_step()

    def get_pred_x0(self, output, t):
        """Extract predicted x0 from model output."""
        if self.diffusion.model_mean_type == ModelMeanType.START_X:
            pred_x0 = output['model_output'] 
        elif self.diffusion.model_mean_type == ModelMeanType.V:
            pred_x0 = self.diffusion._predict_start_from_z_and_v(x_t=output['x_t'], t=t, v=output['model_output'])
        else:
            pred_x0 = self.diffusion._predict_xstart_from_eps(output['x_t'], t, output['model_output'])
        return pred_x0

    def load_random_cams(self, cam_num):
        offset = np.random.uniform(0, 360)
        azimuth = (np.linspace(0, 360, cam_num, endpoint=False) + offset) % 360
        azimuth = azimuth.astype(np.int32)
        elevation = np.random.randint(self.min_elevation, self.max_elevation, cam_num).astype(np.int32)
        cam_radius = np.random.uniform(self.min_cam_radius, self.max_cam_radius, cam_num).astype(np.float32)
        fovx = np.random.uniform(self.min_fovx, self.max_fovx, cam_num).astype(np.float32)
        cams = None
        convert_mat = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).astype(np.float32)
        for i in range(cam_num):
            cam_poses = orbit_camera(elevation[i], azimuth[i], radius=cam_radius[i], opengl=True)
            cam_poses = convert_mat @ cam_poses
            cam = load_cam(c2w=cam_poses, fovx=fovx[i])
            if cams is None:
                for k, v in cam.items():
                    if not isinstance(v, th.Tensor):
                        if isinstance(v, np.ndarray):
                            cam[k] = th.from_numpy(np.array([v]))
                        else:
                            cam[k] = th.tensor([v])
                    else:
                        cam[k] = v.unsqueeze(0)
                cams = cam
            else:
                for k, v in cam.items():
                    if not isinstance(v, th.Tensor):
                        if isinstance(v, np.ndarray):
                            new_tensor = th.from_numpy(np.array([v]))
                        else:
                            new_tensor = th.tensor([v])
                        cams[k] = th.cat([cams[k], new_tensor], dim=0)
                    else:
                        cams[k] = th.cat([cams[k], v.unsqueeze(0)], dim=0)
        return cams

    def generate_sds_camera_views(self, pred_x0_denorm, denoised_denorm):
        """Generate camera views for SDS loss computation."""
        predicted_rendered_images = []
        denoised_rendered_images = []

        # Generate multiple camera viewpoints
        cams = self.load_random_cams(self.render_views)
        for view_idx in range(self.render_views):
            cam = build_single_viewpoint_cam(cams, view_idx)
            res = render(cam, pred_x0_denorm[0], self.std_volume, self.bg_color, self.active_sh_degree)
            predicted_rendered_images.append(res["render"])
            res = render(
                cam,
                denoised_denorm[0],
                self.std_volume,
                self.bg_color,
                self.active_sh_degree,
            )
            denoised_rendered_images.append(res["render"])

        return th.stack(predicted_rendered_images, dim=0), th.stack(
            denoised_rendered_images, dim=0
        )  # [views, 3, H, W]

    @ignore_stderr
    def compute_loss(
        self, pred_x0_denorm, denoised_denorm, prompt, save_guidance_path=None
    ):
        """Compute SDS loss for the predicted x0, with optional LPIPS regularization.
        Also computes reference output using the original (non-LoRA) model without gradients.
        """
        # Generate multiple camera views of the 3D object (LoRA model)
        predicted_rendered_views, denoised_rendered_views = (
            self.generate_sds_camera_views(pred_x0_denorm, denoised_denorm)
        )

        # Refine the rendered views using the refiner (LoRA model)
        refined_output = self.refiner.refine_images(
            images=denoised_rendered_views,
            prompt=[prompt] * denoised_rendered_views.shape[0],
            negative_prompt=[""] * denoised_rendered_views.shape[0],
            guidance_scale=self.guidance_scale,
            output_type="pt",
        )
        if isinstance(refined_output, dict):
            refined_tensors = refined_output["images"]
        else:
            refined_tensors = refined_output
        if save_guidance_path:
            # Save rendered views and refined images side by side for visualization
            # Each row: [rendered_view | refined_image]
            predicted_rendered_np = predicted_rendered_views.detach().cpu().numpy()
            denoised_rendered_np = denoised_rendered_views.detach().cpu().numpy()
            refined_np = refined_tensors.detach().cpu().numpy()
            rows = []
            for i in range(predicted_rendered_np.shape[0]):
                # Convert to uint8 images
                predicted_rendered_img = (
                    np.clip(predicted_rendered_np[i].transpose(1, 2, 0), 0, 1) * 255
                ).astype(np.uint8)
                denoised_rendered_img = (
                    np.clip(denoised_rendered_np[i].transpose(1, 2, 0), 0, 1) * 255
                ).astype(np.uint8)
                refined_img = (
                    np.clip(refined_np[i].transpose(1, 2, 0), 0, 1) * 255
                ).astype(np.uint8)
                row = np.concatenate(
                    [predicted_rendered_img, denoised_rendered_img, refined_img], axis=1
                )
                rows.append(row)
            # Stack all rows vertically
            out_img = np.concatenate(rows, axis=0)
            Image.fromarray(out_img).save(save_guidance_path)
        # Calculate MSE loss (LoRA)
        mse_loss = nn.functional.mse_loss(refined_tensors, predicted_rendered_views)

        return mse_loss

    def forward_backward(self):
        """Forward and backward pass with SDS loss only."""
        zero_grad(self.model_params)
        shape = [
            self.batch_size,
            self.model.in_channels,
            self.model.image_size,
            self.model.image_size,
            self.model.image_size
        ]

        # Start from pure noise
        noise = th.randn(shape, device=dist_util.dev(), dtype=th.float32)

        # Select random prompt
        prompt = np.random.choice(self.prompts)

        # Get text embeddings for GaussianCube diffusion conditioning
        clip_text_embeds = self.clip_text_encoder.encode(
            prompt
        )  # Shape: [1, seq_len, embed_dim]

        # Create model wrapper for inference (put conditioning in model_kwargs)
        model_fn = model_wrapper(
            self.original_ddp_model,  # Use DDP model consistently
            self.noise_schedule,
            model_type="x_start",
            model_kwargs={"cond_text": clip_text_embeds},
        )

        # Step 1: Run partial inference WITHOUT gradients to get partially denoised sample
        with th.no_grad():
            dpm_solver = DPM_Solver(
                model_fn, self.noise_schedule, algorithm_type="dpmsolver++"
            )

            # Run inference from t=1.0 to some intermediate point
            # This gives us a partially denoised sample
            # Lower values = more denoised (less noise), Higher values = more noisy
            intermediate_t = np.random.uniform(
                *self.timestep_range
            )  # Random intermediate timestep
            denoised, intermediates = dpm_solver.sample(
                x=noise,
                steps=100,
                t_start=1.0,
                t_end=1.0 / self.diffusion.num_timesteps,
                order=2,
                skip_type="time_uniform",
                method="multistep",
                return_intermediate=True,
            )
            partially_denoised = intermediates[int((1 - intermediate_t) * 100 + 1)]

        # Step 2: Now predict x0 from the partially denoised sample WITH gradients
        # Convert intermediate_t back to discrete timestep
        final_timestep = th.tensor([int(intermediate_t * self.diffusion.num_timesteps)], device=dist_util.dev())

        # This time we need gradients for the SDS loss
        model_output = self.ddp_model(
            partially_denoised, final_timestep, cond_text=clip_text_embeds
        )

        # Handle case where model outputs both mean and variance
        # Check if model output has twice the channels of input (mean + variance)
        if model_output.shape[1] == partially_denoised.shape[1] * 2:
            # Split model output to get just the mean prediction for x0 computation
            C = partially_denoised.shape[1]
            model_output_mean, _ = th.split(model_output, C, dim=1)
        else:
            model_output_mean = model_output

        # Create output dict for get_pred_x0
        output = {
            "model_output": model_output_mean,
            "x_t": partially_denoised,
        }

        # Get predicted x0 for SDS loss computation
        pred_x0 = self.get_pred_x0(output, final_timestep)
        pred_x0_denorm = pred_x0 * self.std + self.mean
        denoised_denorm = denoised * self.std + self.mean

        # Compute loss using the predicted x0
        if self.step % self.image_save_interval == 0 and dist.get_rank() == 0:
            s_path = os.path.join(logger.get_dir(), "images")
            os.makedirs(s_path, exist_ok=True)
            guidance_path = os.path.join(s_path, f"{self.step:08d}.png")
            total_loss = self.compute_loss(
                pred_x0_denorm,
                denoised_denorm,
                prompt,
                save_guidance_path=guidance_path,
            )
        else:
            total_loss = self.compute_loss(pred_x0_denorm, denoised_denorm, prompt)

        # Log losses (skip timestep-based logging since we don't have batch structure)
        logger.logkv_mean("total_loss", total_loss.item())
        logger.logkv(
            "prompt", str(prompt)
        )  # Convert to regular string to avoid numpy.str_ formatting error

        # Tensorboard logging
        if self.use_tensorboard and self.step % self.log_interval == 0 and dist.get_rank() == 0:
            log_dict = {
                "total_loss": total_loss.item(),
                "intermediate_t": intermediate_t,
                "final_timestep": int(final_timestep[0]),
            }
            self.writer.write_dict(log_dict, self.step)

        # Backward pass
        if self.use_fp16:
            loss_scale = 2 ** self.lg_loss_scale
            (total_loss * loss_scale).backward()
        else:
            total_loss.backward()

    def optimize_fp16(self):
        """Optimize with FP16."""
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()

        # Gradient clipping
        if self.max_grad_norm > 0:
            th.nn.utils.clip_grad_norm_(self.master_params, self.max_grad_norm)

        self._anneal_lr()
        self.opt.step()
        self.warmup_scheduler.step()
        logger.logkv_mean("lr", self.opt.param_groups[0]["lr"])

        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        """Optimize with normal precision."""
        # Gradient clipping
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model_params, self.max_grad_norm)

        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        self.warmup_scheduler.step()
        logger.logkv_mean("lr", self.opt.param_groups[0]["lr"])

        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        """Log gradient norms."""
        sqsum = 0.0
        for p in self.master_params:
            if p.grad is not None:
                sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        """Learning rate annealing (placeholder)."""
        return

    def log_step(self):
        """Log step information."""
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        """Save model checkpoints."""
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"lora_model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"lora_ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with open(os.path.join(get_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        # Save main model
        save_checkpoint(0, self.master_params)

        # Save EMA models
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        # Save LoRA weights separately
        if dist.get_rank() == 0:
            lora_path = os.path.join(get_logdir(), f"lora_weights_{(self.step+self.resume_step):06d}.pt")
            self.lora_model.save_lora_weights(lora_path)
            logger.log(f"saved LoRA weights to {lora_path}")

        # Save optimizer state
        if dist.get_rank() == 0:
            with open(
                os.path.join(get_logdir(), f"lora_opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        """Convert master parameters to state dict."""
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model_params,
                master_params,  # Use actual LoRA parameters, not all model parameters
            )

        # Create a state dict with only LoRA parameters
        state_dict = {}
        param_idx = 0
        for name, param in self.optimize_model.named_parameters():
            # Check if this parameter is in our LoRA parameters
            if any(param is lora_param for lora_param in self.lora_params):
                if param_idx < len(master_params):
                    state_dict[name] = master_params[param_idx]
                    param_idx += 1
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        """Convert state dict to master parameters."""
        # Only get parameters that are in our LoRA parameter list
        params = []
        for name, param in self.optimize_model.named_parameters():
            if any(param is lora_param for lora_param in self.lora_params):
                if name in state_dict:
                    params.append(state_dict[name])

        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


# Utility functions (same as train.py)

def parse_resume_step_from_filename(filename):
    """Parse filenames to extract step number."""
    filename=filename.split('/')[-1]
    assert(filename.endswith(".pt"))
    filename=filename[:-3]
    if filename.startswith("model") or filename.startswith("lora_model"):
        split = filename.split("_")[-1] if "lora_model" in filename else filename[5:]
    elif filename.startswith("ema") or filename.startswith("lora_ema"):
        split = filename.split("_")[-1]
    else:
        return 0
    try:
        return int(split)
    except ValueError:
        return 0


def get_logdir():
    """Get log directory for checkpoints."""
    p = os.path.join(logger.get_dir(),"checkpoints")
    os.makedirs(p,exist_ok=True)
    return p


def find_resume_checkpoint(resume_checkpoint):
    """Find resume checkpoint."""
    if not resume_checkpoint:
        return None
    if "ROOT" in resume_checkpoint:
        maybe_root=os.environ.get("AMLT_MAP_INPUT_DIR")
        maybe_root="OUTPUT/log" if not maybe_root else maybe_root
        root=os.path.join(maybe_root,"checkpoints")
        resume_checkpoint=resume_checkpoint.replace("ROOT",root)
    if "LATEST" in resume_checkpoint:
        files=glob.glob(resume_checkpoint.replace("LATEST","*.pt"))
        if not files:
            return None
        return max(files,key=parse_resume_step_from_filename)
    return resume_checkpoint


def build_single_viewpoint_cam(cam_dict, idx):
    """Build single viewpoint camera."""
    cam = {k: v[idx].to(dist_util.dev()).contiguous() for k, v in cam_dict.items()}
    return cam


def find_ema_checkpoint(main_checkpoint, step, rate):
    """Find EMA checkpoint."""
    if main_checkpoint is None:
        return None
    filename = f"lora_ema_{rate}_{(step):06d}.pt"
    path = os.path.join(os.path.dirname(main_checkpoint), filename)
    if os.path.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    """Log loss dictionary."""
    for key, values in losses.items():
        if key == "sds_prompt":
            continue  # Skip logging prompt as it's a string
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


def create_argparser():
    """Create argument Parser for SDS LoRA fine-tuning."""
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for GaussianCube using SDS")

    # Experiment configuration
    parser.add_argument("--exp_name", type=str, required=True,
                        help="Experiment name for logging")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file (e.g., configs/finetune.yml)")
    parser.add_argument("--model_name", type=str, choices=list(MODEL_REPOS.keys()),
                           help="Predefined model name to download from HuggingFace")

    # Training configuration
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Training batch size (must be 1)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate for LoRA parameters")
    parser.add_argument("--max_steps", type=int, default=50000,
                        help="Maximum training steps")
    parser.add_argument("--use_fp16", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to resume LoRA checkpoint")
    parser.add_argument("--prompt_file", type=str, required=True,
                    help="Path to file containing training prompts")
    parser.add_argument("--use_tensorboard", action="store_true",
                        help="Use tensorboard for logging")

    # Logging configuration
    parser.add_argument("--log_interval", type=int, default=1, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=1000,
                        help="Model saving interval")
    parser.add_argument("--image_save_interval", type=int, default=100,
                        help="Training image saving interval")

    # System configuration
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")

    return parser


def main():
    """Main function for SDS LoRA fine-tuning."""
    args = create_argparser().parse_args()

    # Setup distributed training
    dist_util.setup_dist()
    th.cuda.set_device(dist_util.dev())

    # Set seed
    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load model configuration
    configs = OmegaConf.load(args.config)
    print("Config:", configs)

    # Create model
    print("Creating model...")
    model = UNetModel(**configs['model'])

    # Load model files - either from HuggingFace or local paths
    if args.model_name:
        print(f"Downloading {args.model_name} files from Hugging Face...")
        downloaded_files = download_model_files(args.model_name)
        ckpt_path = downloaded_files["ckpt"]
        mean_file = downloaded_files["mean"]
        std_file = downloaded_files["std"]
        bound = downloaded_files["bound"]
    else:
        raise ValueError("--model_name must be provided.")

    # Load pretrained weights
    print(f"Loading pretrained weights from {ckpt_path}...")
    model.load_state_dict(th.load(ckpt_path, map_location="cpu", weights_only=True))

    # Create diffusion process
    print("Creating diffusion process...")
    diffusion = create_gaussian_diffusion(**configs['diffusion'])

    # Move model to device
    model.to(dist_util.dev())
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    # Setup logging
    logger.configure(args.exp_name)
    if dist.get_rank() == 0:
        options = logger.args_to_dict(args)
        logger.save_args(options)

    # Create LoRA fine-tuning loop
    print("Creating LoRA fine-tuning loop...")
    loop = LoRAFinetuneLoop(
        model=model,
        diffusion=diffusion,
        # Training configuration
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.max_steps,
        use_fp16=args.use_fp16,
        resume_checkpoint=args.resume_checkpoint,
        prompt_file=args.prompt_file,
        use_tensorboard=args.use_tensorboard,
        mean_file=mean_file,
        std_file=std_file,
        bound=bound,
        # Logging configuration
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        image_save_interval=args.image_save_interval,
        # LoRA configuration
        **configs["lora"],
        # Guidance configuration
        **configs["guidance"],
        **configs["model"],
    )

    # Start training
    print("Starting SDS LoRA fine-tuning...")
    loop.run_loop()
    print("Training completed!")


if __name__ == "__main__":
    th.backends.cudnn.benchmark = True
    main()
