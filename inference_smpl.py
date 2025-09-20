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
from model.dpmsolver import NoiseScheduleVP, model_wrapper, DPM_Solver, expand_dims
from model.smpl import SMPLinGaussianCube, smpl_to_openpose
from model.lora_unet import convert_unet_to_lora
from utils import dist_util, logger
from utils.script_util import create_gaussian_diffusion, init_volume_grid, build_single_viewpoint_cam
from dataset.dataset_render import load_data
from gaussian_renderer import render
import imageio
from tqdm import tqdm
from PIL import Image


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


def generate_human_prompt():
    races = ["An Asian", "An African", "A Caucasian", "A Mixed-race"]
    genders = ["man", "woman"]
    hair_colors = ["black", "brown", "blonde", "red", "gray", "white"]
    glasses = ["wearing glasses", "wearing no glasses"]
    cloth_colors = [
        "red",
        "blue",
        "green",
        "black",
        "white",
        "yellow",
        "purple",
        "pink",
        "orange",
        "gray",
        "brown",
    ]
    tops = [
        "t-shirt",
        "shirt",
        "jacket",
        "sweater",
        "hoodie",
        "coat",
        "dress",
        "blouse",
    ]
    pants = ["jeans", "trousers", "shorts", "leggings"]
    shoes = ["sneakers", "boots", "sandals"]

    prompt = (
        f"{random.choice(races)} {random.choice(genders)} with {random.choice(hair_colors)} hair, "
        f"{random.choice(glasses)}, wearing a {random.choice(cloth_colors)} {random.choice(tops)}, "
        f"{random.choice(cloth_colors)} {random.choice(pants)}, and {random.choice(cloth_colors)} {random.choice(shoes)}"
    )

    return prompt


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
    print("Model and Diffusion config: ", configs)

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
    if args.lora_checkpoint is not None:
        print(f"Applying LoRA weights from {args.lora_checkpoint}")
        model = convert_unet_to_lora(model, **configs["lora"], **configs["model"])
        model.load_lora_weights(args.lora_checkpoint)

    logger.configure(args.exp_name)
    options = logger.args_to_dict(args)
    if dist.get_rank() == 0:
        logger.save_args(options)

    model.to(dist_util.dev())
    model.eval()
    print("num of params: {} M".format(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6))

    clip_text_encoder = FrozenCLIPEmbedder()
    clip_text_encoder = clip_text_encoder.eval().to(dist_util.dev())
    prompt = generate_human_prompt()
    print("Text prompt: ", prompt)
    text_features = clip_text_encoder.encode(prompt)

    val_data = load_data(
        batch_size=1,
        deterministic=True,
        class_cond=False,
        text_cond=True,
    )

    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.from_numpy(diffusion.betas).to(dist_util.dev()))
    std_volume = torch.tensor(init_volume_grid(bound=bound, num_pts_each_axis=32)).to(torch.float32).to(dist_util.dev()).contiguous()
    bg_color = torch.tensor([0, 0, 0]).to(torch.float32).to(dist_util.dev())
    mean = torch.load(mean_file, weights_only=True).to(torch.float32).to(dist_util.dev())
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
    if "woman" in prompt:
        actual_human_model = SMPLinGaussianCube(
            "smpl/SMPL_FEMALE.pkl",
            std_volume=std_volume,
            gc_mean=mean,
            gc_std=std,
            device=dist_util.dev(),
            betas=torch.randn([1, 10]),
        )
    else:
        actual_human_model = SMPLinGaussianCube(
            "smpl/SMPL_MALE.pkl",
            std_volume=std_volume,
            gc_mean=mean,
            gc_std=std,
            device=dist_util.dev(),
            betas=torch.randn([1, 10]),
        )
    fixed_x0 = neutral_human_model.fixed_x0
    # initial_x0 = neutral_human_model.initial_x0
    def correcting_xt_fn(xt, t, step, factor=1.0):
        alpha_t, sigma_t = noise_schedule.marginal_alpha(
            t
        ), noise_schedule.marginal_std(t)
        noise = torch.randn_like(xt)
        noisy_fixed_x0 = (
            expand_dims(alpha_t, xt.dim()) * fixed_x0
            + expand_dims(sigma_t, xt.dim()) * noise
        )
        # noisy_initial_x0 = (
        #     expand_dims(alpha_t, xt.dim()) * initial_x0
        #     + expand_dims(sigma_t, xt.dim()) * noise
        # )
        xt_new = xt.clone()
        xt_new[~torch.isnan(noisy_fixed_x0)] = noisy_fixed_x0[~torch.isnan(noisy_fixed_x0)]
        # xt_new[~torch.isnan(noisy_initial_x0)] = factor * noisy_initial_x0[~torch.isnan(noisy_initial_x0)] + (1 - factor) * xt[~torch.isnan(noisy_initial_x0)]
        return xt_new

    if args.poses_file:
        poses = np.load(args.poses_file)["poses"][:,:66] # https://github.com/nghorbani/amass/issues/3#issuecomment-565714925
    img_id = 0
    num_batch_per_rank = args.num_samples // dist.get_world_size()
    for _ in range(num_batch_per_rank):

        model_kwargs = next(val_data)  
        image_size = configs["model"]["image_size"]
        sample_shape = (
            1,
            configs["model"]["in_channels"],
            image_size,
            image_size,
            image_size,
        )

        condition =  {"cond_text": text_features}
        model_fn = model_wrapper(
            model,
            noise_schedule,
            model_type=MODEL_TYPES[configs["diffusion"]["predict_type"]],
            model_kwargs=condition,
        )
        dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type='dpmsolver++', correcting_xt_fn=correcting_xt_fn)

        with torch.no_grad():
            noise = torch.randn(sample_shape, device=dist_util.dev()) * args.temperature

            samples = dpm_solver.sample(
                x=noise,
                steps=args.rescale_timesteps,
                t_start=1.0,
                t_end=1/1000,
                order=2,
                skip_type='time_uniform',
                method='multistep',
            )
            samples_denorm = samples * std + mean
            actual_human_model.update_rest_attributes(
                samples_denorm[0], assignments=neutral_human_model.assignments
            )
            frames = []
            for pose_id, pose in enumerate(tqdm(poses)):
                new_global_orient = torch.zeros([1, 3], device=dist_util.dev())
                new_global_orient[0] = torch.from_numpy(pose[:3]).to(dist_util.dev())
                new_body_pose = torch.zeros([1, 69], device=dist_util.dev())  # 23*3 axis-angle
                new_body_pose[0, :63] = torch.from_numpy(pose[3:66]).to(dist_util.dev())
                actual_human_model.apply_pose(
                    body_pose=new_body_pose, global_orient=new_global_orient
                )
                new_samples_denorm = actual_human_model.to_x0_denorm()
                for i, cam_info in enumerate(model_kwargs["cams"]):
                    # if pose_id % len(model_kwargs["cams"]) != i:
                    #     continue
                    cam = build_single_viewpoint_cam(cam_info, 0)
                    # openpose_img, _ = smpl_to_openpose(
                    #     neutral_human_model.splats["joints"],
                    #     cam_info["full_proj_transform"].squeeze(),
                    #     int(cam_info["image_width"]),
                    #     int(cam_info["image_height"]),
                    # )

                    # # Save OpenPose image
                    # Image.fromarray(openpose_img).save(
                    #     os.path.join(
                    #         s_path,
                    #         "rank_{:02}_render_{:06}_pose_{:06}_cam_{:02}_openpose.png".format(
                    #             dist.get_rank(), img_id, pose_id, i
                    #         ),
                    #     )
                    # )
                    res = render(cam, new_samples_denorm, std_volume, bg_color, args.active_sh_degree)

                    s_path = os.path.join(logger.get_dir(), 'render_images')
                    os.makedirs(s_path,exist_ok=True)
                    output_image = res["render"].clamp(0.0, 1.0)

                    rgb_map = output_image.squeeze().permute(1, 2, 0).cpu() 
                    rgb_map = (rgb_map.detach().numpy() * 255).astype('uint8')
                    imageio.imwrite(os.path.join(s_path, "rank_{:02}_render_{:06}_pose_{:06}_cam_{:02}.png".format(dist.get_rank(), img_id, pose_id, i)), rgb_map)

                    frames.append(rgb_map)
                    break
            if args.render_video:
                s_path = os.path.join(logger.get_dir(), 'videos')
                os.makedirs(s_path,exist_ok=True)
                imageio.mimwrite(
                    os.path.join(
                        s_path,
                        "rank_{:02}_render_{:06}.mp4".format(dist.get_rank(), img_id),
                    ),
                    frames,
                    fps=120,
                )

        img_id += 1
    if dist.is_initialized():
        dist.destroy_process_group()

def create_argparser():
    parser = argparse.ArgumentParser()
    # Experiment args
    parser.add_argument("--model_name", type=str, 
                        default="objaverse_v1.1",
                       help="Name of the model to use")
    parser.add_argument("--exp_name", type=str, default="tmp/smpl_lora")
    parser.add_argument("--seed", type=int, default=0)
    # Model config
    parser.add_argument("--config", type=str, default="configs/finetune_smpl.yml")
    # Data args
    parser.add_argument("--active_sh_degree", type=int, default=0)
    # Inference args
    parser.add_argument("--poses_file", type=str, default="smpl/B1 - stand to walk_poses.npz")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--rescale_timesteps", type=int, default=100)
    parser.add_argument("--render_video", action="store_true")
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        default=None,
        help="Path to LoRA checkpoint to apply (optional)",
    )

    return parser


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
