"""
Stable Diffusion utilities for SDS (Score Distillation Sampling) loss.

This module provides utilities for integrating pretrained Stable Diffusion models
with 3D generation using Score Distillation Sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from pathlib import Path
from torchvision.utils import save_image

# Suppress partial model loading warnings
logging.set_verbosity_error()


def seed_everything(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


class StableDiffusion(nn.Module):
    """
    Stable Diffusion model wrapper for SDS loss computation.
    
    This class provides utilities for computing SDS loss using pretrained
    Stable Diffusion models to guide 3D generation.
    """

    def __init__(
        self,
        device,
        fp16=True,
        vram_O=False,
        hf_key=None,
        sd_version="2.1",
        refiner_strength=0.7,
    ):
        """
        Initialize Stable Diffusion model.

        Args:
            device: Device to run the model on
            fp16: Whether to use FP16 precision
            vram_O: Whether to optimize for VRAM usage
            sd_version: Stable Diffusion version ('2.1', '2.0', '1.5')
            hf_key: Custom HuggingFace model key
            refiner_strength: Strength for image refinement (0.1=minimal, 0.9=major changes)
        """
        super().__init__()

        self.device = device
        self.sd_version = sd_version
        self.refiner_strength = refiner_strength

        print(f'[INFO] Loading Stable Diffusion {sd_version}...')

        # Determine model key
        if hf_key is not None:
            print(f'[INFO] Using custom HuggingFace model: {hf_key}')
            model_key = hf_key
        elif sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable Diffusion version {sd_version} not supported.')

        # Set precision
        self.precision_t = torch.float16 if fp16 else torch.float32

        # Load the full pipeline
        print(f'[INFO] Loading pipeline from {model_key}...')
        self.refiner = StableDiffusionPipeline.from_pretrained(
            model_key,
            torch_dtype=self.precision_t,
            safety_checker=None,
            requires_safety_checker=False,
        )

        # Memory optimization
        if vram_O:
            print('[INFO] Enabling VRAM optimizations...')
            self.refiner.enable_sequential_cpu_offload()
            self.refiner.enable_vae_slicing()
            self.refiner.unet.to(memory_format=torch.channels_last)
            self.refiner.enable_attention_slicing(1)
            if hasattr(self.refiner, "enable_model_cpu_offload"):
                self.refiner.enable_model_cpu_offload()
            if hasattr(self.refiner, "enable_attention_slicing"):
                self.refiner.enable_attention_slicing(1)
        else:
            self.refiner.to(device)

        # Enable xformers for memory efficiency if available
        if is_xformers_available():
            try:
                self.refiner.enable_xformers_memory_efficient_attention()
                print("[INFO] xformers enabled for memory efficiency.")
            except Exception as e:
                print(f"[WARNING] Failed to enable xformers: {e}")

        # Create scheduler for SDS
        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, 
            subfolder="scheduler", 
            torch_dtype=self.precision_t
        )

        print(f"[INFO] Stable Diffusion loaded on device {device}!")

    @torch.no_grad()
    def get_text_embeds(self, prompts):
        """
        Get text embeddings from prompts.
        
        Args:
            prompts: List of text prompts or single string
            
        Returns:
            Text embeddings tensor
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        # Tokenize
        inputs = self.refiner.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.refiner.tokenizer.model_max_length,
            return_tensors="pt",
        )

        # Encode
        embeddings = self.refiner.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings

    def encode_imgs(self, imgs):
        """
        Encode images to latent space using VAE.
        
        Args:
            imgs: Images tensor [B, 3, H, W] in range [0, 1]
            
        Returns:
            Latents tensor [B, 4, H//8, W//8]
        """
        # Ensure correct data type
        imgs = imgs.to(dtype=self.precision_t, device=self.device)

        # Normalize to [-1, 1]
        imgs = 2 * imgs - 1

        # Encode
        posterior = self.refiner.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.refiner.vae.config.scaling_factor

        return latents

    def decode_latents(self, latents):
        """
        Decode latents to image space using VAE.
        
        Args:
            latents: Latents tensor [B, 4, H, W]
            
        Returns:
            Images tensor [B, 3, H*8, W*8] in range [0, 1]
        """
        # Ensure correct data type
        latents = latents.to(dtype=self.precision_t, device=self.device)

        latents = 1 / self.refiner.vae.config.scaling_factor * latents

        imgs = self.refiner.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    @torch.no_grad()
    def refine_images(
        self,
        images,
        prompts,
        negative_prompts="",
        strength=None,
        num_inference_steps=50,
        guidance_scale=10,
        seed=None,
    ):
        """
        Refine an image using diffusion denoising.

        Args:
            prompts: Text prompts
            negative_prompts: Negative text prompts
            pred_rgb: Input image [B, 3, H, W]
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            strength: Refinement strength (0.1=minimal, 0.9=major changes)

        Returns:
            Refined image [B, 3, H, W]
        """

        assert isinstance(images, torch.Tensor), "images must be a torch.Tensor"
        assert images.ndim == 4, f"Expected [B,C,H,W], got {images.shape}"

        # Clamp to valid range [0,1]
        images = images.clamp(0, 1)

        if seed is not None:
            seed_everything(seed)

        # Use provided strength or default
        refine_strength = strength if strength is not None else self.refiner_strength

        if isinstance(images, torch.Tensor):
            if images.ndim == 3:
                images = images.unsqueeze(0)
            assert images.ndim == 4, "Input images should be of shape [B, 3, H, W]"
        else:
            raise ValueError("Input images should be a torch.Tensor")

        if isinstance(prompts, str):
            prompts = [prompts * len(images)]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts * len(images)]

        # Get text embeddings
        pos_embeds = self.get_text_embeds(prompts)
        neg_embeds = self.get_text_embeds(negative_prompts)
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0)

        # Encode to latents
        latents = self.encode_imgs(images)

        # Set up scheduler first
        self.scheduler.set_timesteps(num_inference_steps)

        # For refinement, we start denoising from a middle timestep (not pure noise)
        # Choose a strength between 0.1 (minimal changes) and 0.9 (major changes)
        init_timestep = min(
            int(num_inference_steps * refine_strength), num_inference_steps - 1
        )
        t_start = max(num_inference_steps - init_timestep, 0)

        # Add noise corresponding to the chosen timestep
        timestep = self.scheduler.timesteps[t_start]
        noise = torch.randn_like(latents, dtype=self.precision_t, device=self.device)
        latents_noisy = self.scheduler.add_noise(latents, noise, timestep)

        # Use timesteps from the starting point
        timesteps = self.scheduler.timesteps[t_start:]

        # Denoising loop
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.refiner.unet(
                latent_model_input, t, encoder_hidden_states=text_embeds
            ).sample

            # Apply guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # Step
            latents_noisy = self.scheduler.step(noise_pred, t, latents_noisy)['prev_sample']

        # Decode to image
        refined_imgs = self.decode_latents(latents_noisy)

        return refined_imgs


if __name__ == "__main__":
    import argparse
    import torchvision.transforms as transforms
    from PIL import Image

    parser = argparse.ArgumentParser(description='Test Stable Diffusion utilities')
    parser.add_argument('--image', type=str, required=True, help='Path to input image for refinement')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for refinement')
    parser.add_argument('--negative_prompt', type=str, default='', help='Negative text prompt')
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'],
                       help='Stable Diffusion version')
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of denoising steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=10,
        help="Classifier-free guidance scale",
    )
    parser.add_argument('--strength', type=float, default=0.7, help='Refinement strength (0.1=minimal, 0.9=major changes)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='refined_output.png', help='Output image path')
    parser.add_argument('--fp16', action='store_true', default=True, help='Use FP16 precision')
    parser.add_argument('--no_fp16', action='store_true', help='Disable FP16 precision')

    args = parser.parse_args()

    # Set random seed
    if args.seed is not None:
        seed_everything(args.seed)
        print(f"Using seed: {args.seed}")

    # Check if image exists
    if not Path(args.image).exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine FP16 usage
    use_fp16 = args.fp16 and not args.no_fp16 and torch.cuda.is_available()
    print(f"Using FP16: {use_fp16}")

    # Initialize Stable Diffusion
    print(f"Initializing Stable Diffusion {args.sd_version}...")
    sd = StableDiffusion(
        device=device,
        fp16=use_fp16,
        sd_version=args.sd_version,
        refiner_strength=args.strength,
    )

    # Load and preprocess image
    print(f"Loading image: {args.image}")
    image = Image.open(args.image).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((768, 768)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)

    # Ensure correct data type and device
    image_tensor = image_tensor.to(device=device)

    print(f"Image tensor shape: {image_tensor.shape}")
    print(f"Prompt: '{args.prompt}'")
    if args.negative_prompt:
        print(f"Negative prompt: '{args.negative_prompt}'")

    # Refine the image
    print(f"Refining image with strength {args.strength}...")
    with torch.no_grad():
        refined_image = sd.refine_images(
            images=image_tensor,
            prompts=args.prompt,
            negative_prompts=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        )

    print(f"Refined image shape: {refined_image.shape}")

    # Save refined image
    refined_pil = transforms.ToPILImage()(refined_image.squeeze(0).cpu())
    refined_pil.save(args.output)

    print(f"Refined image saved to {args.output}")
    print("Image refinement completed successfully!")
