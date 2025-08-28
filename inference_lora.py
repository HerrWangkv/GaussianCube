import os
import random
import argparse
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from mpi4py import MPI
from huggingface_hub import hf_hub_download

from model.unet import UNetModel
from model.clip import FrozenCLIPEmbedder
from model.dpmsolver import NoiseScheduleVP, model_wrapper, DPM_Solver
from model.lora_unet import convert_unet_to_lora, LoRAUNetModel
from utils import dist_util, logger
from utils.script_util import create_gaussian_diffusion, init_volume_grid, build_single_viewpoint_cam
from dataset.dataset_render import load_data
from gaussian_renderer import render
import imageio
from tqdm import tqdm
import glob


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
        "bound": 0.5
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
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    print("Start inference...")
    args = create_argparser().parse_args()

    configs = OmegaConf.load(args.config)
    print("Config: ", configs)

    print(f"Downloading {args.model_name} files from Hugging Face...")
    downloaded_files = download_model_files(args.model_name)

    ckpt = downloaded_files["ckpt"]
    mean_file = downloaded_files["mean"]
    std_file = downloaded_files["std"]
    bound = downloaded_files["bound"]

    dist_util.setup_dist()
    torch.cuda.set_device(dist_util.dev())
    seed_everything(args.seed + dist.get_rank())

    configs["model"]["precision"] = "32"
    model = UNetModel(**configs["model"])

    diffusion = create_gaussian_diffusion(**configs["diffusion"])
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    print("Loaded ckpt: ", ckpt)

    # Add LoRA support
    print(f"Applying LoRA weights from {args.lora_checkpoint}")
    full_model = convert_unet_to_lora(model, **configs["lora"], **configs["model"])
    full_model.load_lora_weights(args.lora_checkpoint)
    full_model.eval()
    full_model.to(dist_util.dev())

    logger.configure(args.exp_name)
    options = logger.args_to_dict(args)
    if dist.get_rank() == 0:
        logger.save_args(options)

    model.to(dist_util.dev())
    model.eval()
    print("num of params: {} M".format(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6))

    clip_text_encoder = FrozenCLIPEmbedder()
    clip_text_encoder = clip_text_encoder.eval().to(dist_util.dev())
    if args.prompt_file is not None and os.path.exists(args.prompt_file):
        with open(args.prompt_file, "r") as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
            print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")
    else:
        prompts = None
    val_data = load_data(
        batch_size=1,
        deterministic=True,
        class_cond=False,
        text_cond=True,
    )

    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.from_numpy(diffusion.betas).to(dist_util.dev()))
    std_volume = torch.tensor(init_volume_grid(bound=bound, num_pts_each_axis=32)).to(torch.float32).to(dist_util.dev()).contiguous()
    bg_color = torch.tensor([1, 1, 1]).to(torch.float32).to(dist_util.dev())
    mean = (
        torch.load(mean_file, weights_only=True).to(torch.float32).to(dist_util.dev())
    )
    std = torch.load(std_file, weights_only=True).to(torch.float32).to(dist_util.dev())

    mean = mean.permute(3, 0, 1, 2).requires_grad_(False).contiguous()
    std = std.permute(3, 0, 1, 2).requires_grad_(False).contiguous()

    if prompts:
        num_batch_per_rank = args.num_samples * (len(prompts) // dist.get_world_size())
    else:
        num_batch_per_rank = args.num_samples // dist.get_world_size()
    for idx in tqdm(range(num_batch_per_rank)):

        model_kwargs = next(val_data)  

        image_size = configs["model"]["image_size"]
        sample_shape = (
            1,
            configs["model"]["in_channels"],
            image_size,
            image_size,
            image_size,
        )
        if prompts is not None:
            prompt = prompts[
                (idx + num_batch_per_rank * dist.get_rank()) // args.num_samples
            ][11:]
            idx_amoung_same_prompt = (
                idx + num_batch_per_rank * dist.get_rank()
            ) % args.num_samples
            text_features = clip_text_encoder.encode(prompt)
        elif args.text:
            prompt = args.text
            idx_amoung_same_prompt = idx + num_batch_per_rank * dist.get_rank()
            text_features = clip_text_encoder.encode(prompt)
        else:
            raise ValueError(
                "Either --text or --prompt_file must be provided for text conditioning."
            )
        condition =  {"cond_text": text_features}

        model_fn = model_wrapper(
            model,
            noise_schedule,
            model_type=MODEL_TYPES[configs["diffusion"]["predict_type"]],
            model_kwargs=condition,
        )
        dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type='dpmsolver++')

        with torch.no_grad():
            noise = torch.randn(sample_shape, device=dist_util.dev()) * args.temperature
            samples = dpm_solver.sample(
                x=noise,
                t_start=1.0,
                t_end=configs["guidance"]["timestep_range"][1],
                order=2,
                skip_type="time_uniform",
                method="adaptive",
            )
            final_timestep = torch.tensor(
                [configs["guidance"]["timestep_range"][1] * diffusion.num_timesteps]
            ).to(dist_util.dev())
            samples_output = full_model(samples, final_timestep, **condition)
            samples, _ = torch.split(samples_output, samples.shape[1], dim=1)

            samples_denorm = samples * std + mean

            frames = []
            for i, cam_info in enumerate(model_kwargs["cams"]):
                cam = build_single_viewpoint_cam(cam_info, 0)
                res = render(cam, samples_denorm[0], std_volume, bg_color, args.active_sh_degree)

                s_path = os.path.join(logger.get_dir(), 'render_images')
                os.makedirs(s_path,exist_ok=True)
                output_image = res["render"].clamp(0.0, 1.0)

                rgb_map = output_image.squeeze().permute(1, 2, 0).cpu() 
                rgb_map = (rgb_map.detach().numpy() * 255).astype('uint8')
                imageio.imwrite(
                    os.path.join(
                        s_path, f"{prompt}_{idx_amoung_same_prompt:03}_cam_{i:02}.png"
                    ),
                    rgb_map,
                )

                frames.append(rgb_map)

            samples = samples.permute(0, 2, 3, 4, 1).cpu()
            torch.save(
                samples[0],
                os.path.join(
                    logger.get_dir(), f"{prompt}_{idx_amoung_same_prompt:03}" + ".pt"
                ),
            )

            if args.render_video:
                s_path = os.path.join(logger.get_dir(), 'videos')
                os.makedirs(s_path,exist_ok=True)
                imageio.mimwrite(
                    os.path.join(s_path, f"{prompt}_{idx_amoung_same_prompt:03}.mp4"),
                    frames,
                    fps=30,
                )
                # Remove images after video is rendered
                for i in range(len(frames)):
                    img_path = os.path.join(
                        s_path.replace("videos", "render_images"),
                        f"{prompt}_{idx_amoung_same_prompt:03}_cam_{i:02}.png",
                    )
                    if os.path.exists(img_path):
                        os.remove(img_path)
    if dist.is_initialized():
        dist.destroy_process_group()

def create_argparser():
    parser = argparse.ArgumentParser()
    # Experiment args
    parser.add_argument("--model_name", type=str,
                       default="objaverse_v1.1",
                       help="Name of the model to use")
    parser.add_argument("--exp_name", type=str, default="tmp/car_lora/")
    parser.add_argument("--seed", type=int, default=0)
    # Model config
    parser.add_argument("--config", type=str, default="configs/finetune.yml")
    # Data args
    parser.add_argument("--active_sh_degree", type=int, default=0)
    # Inference args
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--render_video", action="store_true")
    parser.add_argument("--text", type=str, default="A car.")
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        default=None,
        required=True,
        help="Path to LoRA checkpoint to apply (optional)",
    )

    return parser


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
