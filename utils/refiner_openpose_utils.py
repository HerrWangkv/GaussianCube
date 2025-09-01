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
        refiner_strength=0.3,
    ):
        """
        Initialize Stable Diffusion XL Refiner model.
        
        Args:
            device: Device to run the model on
            fp16: Whether to use FP16 precision
            vram_O: Whether to optimize for VRAM usage
            base_model_key: Base SDXL model key
            controlnet_key: ControlNet model key
            refiner_strength: Strength of refinement (0.0-1.0)
        """
        super().__init__()

        self.device = device
        self.refiner_strength = refiner_strength

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

        self.scheduler = DDIMScheduler.from_pretrained(
            base_model_key, 
            subfolder="scheduler", 
            torch_dtype=self.precision_t
        )
        print(f'[INFO] Stable Diffusion XL Refiner loaded on device {device}!')

    @torch.no_grad()
    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        """
        Get time IDs for SDXL conditioning
        """
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids.to(self.device)

    def get_text_embeds(self, prompts):
        """
        Get text embeddings for both text encoders (SDXL)
        
        Args:
            prompts: List of text prompts
            
        Returns:
            text_embeds: Concatenated embeddings for encoder_hidden_states
            pooled_embeds: Pooled embeddings for added_cond_kwargs
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        # Tokenize for first text encoder
        inputs = self.refiner.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.refiner.tokenizer.model_max_length,
            return_tensors="pt",
        )

        # Tokenize for second text encoder  
        inputs_2 = self.refiner.tokenizer_2(
            prompts,
            padding="max_length", 
            max_length=self.refiner.tokenizer_2.model_max_length,
            return_tensors="pt",
        )

        # Encode with both text encoders
        embeddings_1 = self.refiner.text_encoder(inputs.input_ids.to(self.device))[0]
        pooled_embeddings_2 = self.refiner.text_encoder_2(inputs_2.input_ids.to(self.device))[0]
        
        # For encoder_hidden_states, expand pooled embeddings to match sequence dimension
        if pooled_embeddings_2.dim() == 2:
            # Add sequence dimension to match embeddings_1
            embeddings_2 = pooled_embeddings_2.unsqueeze(1).expand(-1, embeddings_1.shape[1], -1)
        else:
            embeddings_2 = pooled_embeddings_2
        
        # Concatenate embeddings along the feature dimension for encoder_hidden_states
        embeddings = torch.cat([embeddings_1, embeddings_2], dim=-1)

        return embeddings, pooled_embeddings_2

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
        cond_images,
        prompts,
        negative_prompts="",
        strength=None,
        num_inference_steps=50,
        guidance_scale=10,
        controlnet_conditioning_scale=1.0,
        seed=None,
    ):
        """
        Refine an image using diffusion denoising with OpenPose ControlNet.

        Args:
            images: Input image tensor [B, 3, H, W]
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
        assert isinstance(images, torch.Tensor), "images must be a torch.Tensor"
        assert images.ndim == 4, f"Expected [B,C,H,W], got {images.shape}"
        assert isinstance(cond_images, torch.Tensor), "cond_images must be a torch.Tensor"
        assert cond_images.ndim == 4, f"Expected [B,C,H,W], got {cond_images.shape}"
        assert cond_images.shape == images.shape, f"Expected cond_images to have same shape as images, got {cond_images.shape} vs {images.shape}"
        if seed is not None:
            seed_everything(seed)

        # Use provided strength or default
        refine_strength = strength if strength is not None else self.refiner_strength

        # Prepare prompts
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts] * len(images)
        
        # Get text embeddings
        pos_embeds, pos_pooled = self.get_text_embeds(prompts)
        neg_embeds, neg_pooled = self.get_text_embeds(negative_prompts)
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0)
        pooled_embeds = torch.cat([neg_pooled, pos_pooled], dim=0)
        
        # Get time IDs for SDXL conditioning
        original_size = (images.shape[-2], images.shape[-1])  # (height, width)
        target_size = original_size
        crops_coords_top_left = (0, 0)
        
        # Generate time IDs
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, 
            dtype=text_embeds.dtype
        )
        
        # Duplicate for CFG (negative + positive)
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

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
            down_block_res_samples, mid_block_res_sample = self.refiner.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeds,
                controlnet_cond=cond_images.to(dtype=self.precision_t, device=self.device),
                added_cond_kwargs={
                    "text_embeds": pooled_embeds.chunk(2)[1],
                    "time_ids": add_time_ids.chunk(2)[1]
                },
                conditioning_scale=controlnet_conditioning_scale,
                return_dict=False,
            )
            noise_pred = self.refiner.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeds,
                down_block_additional_residuals=[
                    sample.to(dtype=self.precision_t)
                    for sample in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample.to(
                    dtype=self.precision_t
                ),
                added_cond_kwargs={
                    "text_embeds": pooled_embeds,
                    "time_ids": add_time_ids
                },
                return_dict=False,
            )[0]

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

    parser = argparse.ArgumentParser(description='Test SDXL Refiner utilities')
    parser.add_argument('--image', type=str, required=True, help='Path to input image for refinement')
    parser.add_argument("--cond_image", type=str, required=True, help='Path to conditional image for refinement')
    parser.add_argument('--output', type=str, default='refined_output.png', help='Output path for refined image')
    parser.add_argument('--prompt', type=str, default='', help='Prompt to guide refinement')
    parser.add_argument('--negative_prompt', type=str, default='', help='Negative prompt')
    parser.add_argument('--strength', type=float, default=0.7, help='Refinement strength (0.0-1.0)')
    parser.add_argument('--steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 precision')
    parser.add_argument('--vram_O', action='store_true', help='Optimize for VRAM')
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=10.0,
        help="Guidance scale for classifier-free guidance",
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=1.0,
        help="ControlNet conditioning scale",
    )

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
    use_fp16 = args.fp16 and torch.cuda.is_available()
    print(f"Using FP16: {use_fp16}")

    # Initialize refiner
    refiner = StableDiffusionXLOpenposeRefiner(
        device=device,
        fp16=use_fp16,
        vram_O=args.vram_O,
        refiner_strength=args.strength,
    )

    # Load and refine image
    print(f"Loading image: {args.image}")
    input_image = Image.open(args.image).convert('RGB')
    input_cond_image = Image.open(args.cond_image).convert('RGB')

    # Convert to tensor
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(input_image).unsqueeze(0)
    cond_image_tensor = transform(input_cond_image).unsqueeze(0)

    print(f"Image tensor shape: {image_tensor.shape}")
    print(f"Prompt: '{args.prompt}'")
    if args.negative_prompt:
        print(f"Negative prompt: '{args.negative_prompt}'")

    # Refine the image
    print(f"Refining image with strength {args.strength}...")
    with torch.no_grad():
        refined_tensor = refiner.refine_images(
            images=image_tensor,
            cond_images=cond_image_tensor,
            prompts=args.prompt,
            negative_prompts=args.negative_prompt,
            strength=args.strength,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            seed=args.seed,
        )

    print(f"Refined tensor shape: {refined_tensor.shape}")

    # Convert back to PIL and save
    refined_pil = transforms.ToPILImage()(refined_tensor.squeeze(0).cpu())
    refined_pil.save(args.output)
    print(f"Refined image saved to {args.output}")
    print("Image refinement completed successfully!")
