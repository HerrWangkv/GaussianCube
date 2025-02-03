import argparse
import torch
import torch.distributed as dist
import torch.utils.cpp_extension
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download

from model.unet import ControlledUNetModel, ControlNet
from model.resample import UniformSampler
from utils import dist_util, logger
from utils.script_util import create_gaussian_diffusion
from train import FinetuneLoop

def download_model_files():
    """Download text-conditioned pretrained model files from Hugging Face Hub."""
    model_info = {
        "repo_id": "BwZhang/GaussianCube-Objaverse",
        "revision": "main",
        "model_path": "v1.1/objaverse_ckpt.pt",
        "mean_path": "v1.1/mean.pt",
        "std_path": "v1.1/std.pt",
        "bound": 0.5
    }
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
        print(f"Error downloading files: {e}")
        raise
    
    return downloaded_files

def main():
    args = create_argparser().parse_args()

    model_and_diffusion_config = OmegaConf.load(args.config)
    print("Model and Diffusion config: ", model_and_diffusion_config)

    dist_util.setup_dist()
    torch.cuda.set_device(dist_util.dev())

    model = ControlledUNetModel(**model_and_diffusion_config['model'])
    controlnet = ControlNet(**model_and_diffusion_config['controlnet'])
    diffusion = create_gaussian_diffusion(**model_and_diffusion_config['diffusion'])
    has_pretrain_weight = True
    downloaded_files = download_model_files()
    ckpt = downloaded_files["ckpt"]
    mean_file = downloaded_files["mean"]
    std_file = downloaded_files["std"]
    bound = downloaded_files["bound"]
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    for param in model.parameters():
        param.requires_grad_(False)
    if args.ckpt is not None:
        controlnet.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    else:
        controlnet.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)

    logger.configure(args.exp_name)
    options = logger.args_to_dict(args)
    if dist.get_rank() == 0:
        logger.save_args(options)

    model.to(dist_util.dev())
    controlnet.to(dist_util.dev())
    print("num of params: {} M".format(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6))

    schedule_sampler = UniformSampler(model_and_diffusion_config['diffusion']['steps'])

    logger.log("creating data loader...")

    #TODO: manage uncond_p in TrainLoop
    logger.log("training...")
    FinetuneLoop(
        model,
        controlnet,
        diffusion,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        uncond_p=args.uncond_p,
        use_fp16=args.use_fp16,
        weight_decay=args.weight_decay,
        schedule_sampler=schedule_sampler,
        use_vgg=args.use_vgg,
        use_tensorboard=args.use_tensorboard,
        render_l1_weight=args.render_l1_weight,
        render_lpips_weight=args.render_lpips_weight,
        mean_file=mean_file,#args.mean_file,
        std_file=std_file,#args.std_file,
        bound=bound,#args.bound,
        has_pretrain_weight=has_pretrain_weight,
        num_pts_each_axis=args.num_pts_each_axis,
    ).run_loop()

 
def create_argparser():
    def none_or_str(value):  
        if value.lower() == 'none':  
            return None  
        return value
    parser = argparse.ArgumentParser()
    # Experiment args
    parser.add_argument("--exp_name", type=str, default="/tmp/output/")
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--use_tensorboard", action="store_true")
    # Model config
    parser.add_argument("--config", type=str, default="configs/finetune.yml")
    # Train args
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--microbatch", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--use_vgg", action="store_true")
    parser.add_argument("--ema_rate", type=float, default=0.9999)
    parser.add_argument("--uncond_p", type=float, default=0.2)
    parser.add_argument("--render_l1_weight", type=float, default=1.0)
    parser.add_argument("--render_lpips_weight", type=float, default=1.0)
    # Data args
    parser.add_argument("--num_pts_each_axis", type=int, default=32)
 
    return parser


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
