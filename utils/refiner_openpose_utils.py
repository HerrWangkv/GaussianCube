"""
Stable Diffusion XL Refiner utilities for enhancing generated images.

This module provides utilities for refining images using the Stable Diffusion XL Refiner model
to improve the quality and details of generated images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import (
    StableDiffusionXLImg2ImgPipeline, 
    DiffusionPipeline,
    StableDiffusionXLControlNetPipeline, 
    ControlNetModel,
    DDIMScheduler,
    AutoencoderKL,
    UNet2DConditionModel
)
from diffusers.utils.import_utils import is_xformers_available
from pathlib import Path
from torchvision.utils import save_image
import torchvision.transforms as transforms
from PIL import Image
import warnings

# Suppress partial model loading warnings
logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)


def seed_everything(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class StableDiffusionXLOpenposeRefiner(nn.Module):
    """
    Stable Diffusion XL Refiner model wrapper for image refinement.
    
    This class provides utilities for refining images using the Stable Diffusion XL Refiner
    to enhance details and improve overall quality.
    """

    def __init__(
        self,
        device,
        fp16=True,
        vram_O=False,
        base_model_key=None,
        controlnet_key=None,
    ):
        """
        Initialize Stable Diffusion XL Refiner model.

        Args:
            device: Device to run the model on
            fp16: Whether to use FP16 precision
            vram_O: Whether to optimize for VRAM usage
            base_model_key: Base SDXL model key
            controlnet_key: ControlNet model key
        """
        super().__init__()

        self.device = device

        print(f'[INFO] Loading Stable Diffusion XL Refiner...')

        # Determine refiner model key
        if base_model_key is not None:
            print(f'[INFO] Using custom base model key: {base_model_key}')
            base_model_key = base_model_key
        else:
            base_model_key = "stabilityai/stable-diffusion-xl-base-1.0"
        if controlnet_key is not None:
            print(f'[INFO] Using custom ControlNet model: {controlnet_key}')
            controlnet_key = controlnet_key
        else:
            controlnet_key = "thibaud/controlnet-openpose-sdxl-1.0"

        # Set precision
        self.precision_t = torch.float16 if fp16 else torch.float32

        # Load the refiner pipeline
        print(f"[INFO] Loading refiner pipeline from {base_model_key} and {controlnet_key}...")
        controlnet = ControlNetModel.from_pretrained(controlnet_key, torch_dtype=self.precision_t)
        self.refiner = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_model_key, controlnet=controlnet, torch_dtype=self.precision_t
        )

        # Memory optimization
        if vram_O:
            print('[INFO] Enabling VRAM optimizations for refiner...')
            self.refiner.enable_sequential_cpu_offload()
            self.refiner.enable_vae_slicing()
            if hasattr(self.refiner, 'enable_model_cpu_offload'):
                self.refiner.enable_model_cpu_offload()
            if hasattr(self.refiner, 'enable_attention_slicing'):
                self.refiner.enable_attention_slicing(1)

        else:
            self.refiner.to(device)

        # Enable xformers for memory efficiency if available
        if is_xformers_available():
            try:
                self.refiner.enable_xformers_memory_efficient_attention()
                print('[INFO] xformers enabled for memory efficiency.')
            except Exception as e:
                print(f'[WARNING] Failed to enable xformers: {e}')

        print(f'[INFO] Stable Diffusion XL Refiner loaded on device {device}!')

    @torch.no_grad()
    def refine_images(
        self,
        cond_images,
        prompts,
        negative_prompts="",
        num_inference_steps=25,
        guidance_scale=10,
        seed=None,
        **kwargs,
    ):
        """
        Refine an image using diffusion denoising with OpenPose ControlNet.

        Args:
            cond_images: OpenPose conditioning image tensor [B, 3, H, W]
            prompts: Text prompts
            negative_prompts: Negative text prompts
            strength: Refinement strength (0.1=minimal, 0.9=major changes)
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            controlnet_conditioning_scale: ControlNet conditioning scale
            seed: Random seed for reproducibility

        Returns:
            Refined image tensor [B, 3, H, W]
        """
        assert isinstance(cond_images, torch.Tensor), "cond_images must be a torch.Tensor"
        assert cond_images.ndim == 4, f"Expected [B,C,H,W], got {cond_images.shape}"
        if seed is not None:
            seed_everything(seed)

        # Prepare prompts
        if isinstance(prompts, str):
            prompts = [prompts] * len(cond_images)
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts] * len(cond_images)

        # Decode to image
        refined_output = self.refiner(
            prompts,
            negative_prompt=negative_prompts,
            num_inference_steps=num_inference_steps,
            image=cond_images.to(self.device, dtype=self.precision_t),
            guidance_scale=guidance_scale,
            **kwargs,
        )
        if hasattr(refined_output, "images"):
            return refined_output.images
        elif isinstance(refined_output, dict) and "images" in refined_output:
            return refined_output["images"]
        else:
            return refined_output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test SDXL Refiner utilities")
    parser.add_argument(
        "--cond_image",
        type=str,
        required=True,
        help="Path to conditional image for refinement",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="refined_output.png",
        help="Output path for refined image",
    )
    parser.add_argument(
        "--prompt", type=str, default="", help="Prompt to guide refinement"
    )
    parser.add_argument(
        "--negative_prompt", type=str, default="", help="Negative prompt"
    )
    parser.add_argument(
        "--steps", type=int, default=25, help="Number of inference steps"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    parser.add_argument("--vram_O", action="store_true", help="Optimize for VRAM")
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=10,
        help="Guidance scale for classifier-free guidance",
    )
    args = parser.parse_args()

    # Set random seed
    if args.seed is not None:
        seed_everything(args.seed)
        print(f"Using seed: {args.seed}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine FP16 usage
    use_fp16 = args.fp16 and torch.cuda.is_available()
    print(f"Using FP16: {use_fp16}")

    # Initialize refiner
    refiner = StableDiffusionXLOpenposeRefiner(
        device=device,
        fp16=use_fp16,
        vram_O=args.vram_O,
    )

    # Load and refine image
    input_cond_image = Image.open(args.cond_image).convert('RGB')

    # Convert to tensor
    transform = transforms.Compose(
        [
            transforms.Resize((768, 768)),
            transforms.ToTensor(),
        ]
    )
    cond_image_tensor = transform(input_cond_image).unsqueeze(0)

    print(f"Image tensor shape: {cond_image_tensor.shape}")
    print(f"Prompt: '{args.prompt}'")
    if args.negative_prompt:
        print(f"Negative prompt: '{args.negative_prompt}'")

    # Refine the image
    with torch.no_grad():
        refined_tensor = refiner.refine_images(
            cond_images=cond_image_tensor,
            prompts=args.prompt,
            negative_prompts=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            output_type="pt",
        )

    print(f"Refined tensor shape: {refined_tensor.shape}")

    # Convert back to PIL and save
    refined_pil = transforms.ToPILImage()(refined_tensor.squeeze(0).cpu())
    refined_pil.save(args.output)
    print(f"Refined image saved to {args.output}")
    print("Image refinement completed successfully!")
