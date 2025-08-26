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
    DDIMScheduler,
    AutoencoderKL,
    UNet2DConditionModel
)
from peft import PeftModel
from diffusers.utils.import_utils import is_xformers_available
from pathlib import Path
from torchvision.utils import save_image
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


class StableDiffusionXLRefiner(nn.Module):
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
        hf_key=None,
        base_model_key=None,
        refiner_strength=0.3,
        lora_path=None,
    ):
        """
        Initialize Stable Diffusion XL Refiner model.
        
        Args:
            device: Device to run the model on
            fp16: Whether to use FP16 precision
            vram_O: Whether to optimize for VRAM usage
            hf_key: Custom HuggingFace refiner model key
            base_model_key: Base SDXL model key (for ensemble refinement)
            refiner_strength: Strength of refinement (0.0-1.0)
        """
        super().__init__()

        self.device = device
        self.refiner_strength = refiner_strength

        print(f'[INFO] Loading Stable Diffusion XL Refiner...')

        # Determine refiner model key
        if hf_key is not None:
            print(f'[INFO] Using custom HuggingFace refiner model: {hf_key}')
            refiner_model_key = hf_key
        else:
            refiner_model_key = "stabilityai/stable-diffusion-xl-refiner-1.0"

        # Set precision
        self.precision_t = torch.float16 if fp16 else torch.float32

        # Load the refiner pipeline
        print(f"[INFO] Loading refiner pipeline from {refiner_model_key}...")
        try:
            self.refiner = DiffusionPipeline.from_pretrained(
                refiner_model_key,
                torch_dtype=self.precision_t,
                use_safetensors=True,
                variant="fp16" if fp16 else None
            )
        except Exception as e:
            print(f'[WARNING] Failed to load with variant, trying without: {e}')
            self.refiner = DiffusionPipeline.from_pretrained(
                refiner_model_key,
                torch_dtype=self.precision_t,
                use_safetensors=True
            )

        # Load LoRA weights if provided
        if lora_path is not None:
            print(f"[INFO] Loading LoRA weights from {lora_path}")
            try:
                self.refiner.unet = PeftModel.from_pretrained(
                    self.refiner.unet,
                    lora_path,
                )
                print(f"[INFO] LoRA weights loaded successfully.")
            except Exception as e:
                print(f"[WARNING] Failed to load LoRA weights: {e}")

        # Load base model for ensemble refinement if specified
        self.base_pipe = None
        if base_model_key is not None:
            print(f'[INFO] Loading base model for ensemble refinement: {base_model_key}')
            try:
                self.base_pipe = DiffusionPipeline.from_pretrained(
                    base_model_key,
                    torch_dtype=self.precision_t,
                    use_safetensors=True,
                    variant="fp16" if fp16 else None
                )
            except Exception as e:
                print(f'[WARNING] Failed to load base model with variant: {e}')
                self.base_pipe = DiffusionPipeline.from_pretrained(
                    base_model_key,
                    torch_dtype=self.precision_t,
                    use_safetensors=True
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

            if self.base_pipe is not None:
                self.base_pipe.enable_sequential_cpu_offload()
                self.base_pipe.enable_vae_slicing()
                if hasattr(self.base_pipe, 'enable_model_cpu_offload'):
                    self.base_pipe.enable_model_cpu_offload()
                if hasattr(self.base_pipe, 'enable_attention_slicing'):
                    self.base_pipe.enable_attention_slicing(1)
        else:
            self.refiner.to(device)
            if self.base_pipe is not None:
                self.base_pipe.to(device)

        # Enable xformers for memory efficiency if available
        if is_xformers_available():
            try:
                self.refiner.enable_xformers_memory_efficient_attention()
                if self.base_pipe is not None:
                    self.base_pipe.enable_xformers_memory_efficient_attention()
                print('[INFO] xformers enabled for memory efficiency.')
            except Exception as e:
                print(f'[WARNING] Failed to enable xformers: {e}')

        print(f'[INFO] Stable Diffusion XL Refiner loaded on device {device}!')

    @torch.no_grad()
    def refine_image(
        self,
        image,
        prompt="",
        negative_prompt="",
        strength=None,
        num_inference_steps=50,
        guidance_scale=10,
        seed=None,
        high_noise_frac=0.8,
        aesthetic_score=4.0,
        negative_aesthetic_score=2.5,
        **kwargs,
    ):
        """
        Refine an input image using the SDXL Refiner.
        
        Args:
            image: Input image (PIL Image, torch tensor, or numpy array)
            prompt: Text prompt to guide refinement
            negative_prompt: Negative text prompt
            strength: Refinement strength (overrides default if provided)
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            seed: Random seed for reproducibility
            high_noise_frac: Fraction of noise steps for base model (ensemble mode)
            
        Returns:
            Refined PIL Image
        """
        if seed is not None:
            seed_everything(seed)

        # Use provided strength or default
        refine_strength = strength if strength is not None else self.refiner_strength

        # Convert input to PIL Image if needed
        if isinstance(image, torch.Tensor):
            # Assume image is in [0, 1] range with shape [C, H, W] or [B, C, H, W]
            if len(image.shape) == 4:
                image = image[0]  # Take first image if batch
            image = image.clamp(0, 1)
            image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image)
        elif isinstance(image, np.ndarray):
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)

        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Ensemble refinement with base model
        if self.base_pipe is not None:
            print('[INFO] Using ensemble refinement with base + refiner models...')

            # Generate with base model first
            base_output = self.base_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                strength=refine_strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                denoising_end=high_noise_frac,
                output_type=kwargs.get("output_type", "latent"),
                **{k: v for k, v in kwargs.items() if k != "output_type"},
            )
            base_image = (
                base_output.images if hasattr(base_output, "images") else base_output
            )

            # Refine with refiner model
            refined_output = self.refiner(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=base_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                denoising_start=high_noise_frac,
                **kwargs,
            )
        else:
            # Direct refinement
            # print('[INFO] Using direct refinement...')
            refined_output = self.refiner(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                strength=refine_strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                aesthetic_score=aesthetic_score,
                negative_aesthetic_score=negative_aesthetic_score,
                **kwargs,
            )

        # Return only images (PIL, numpy, or torch tensor) depending on output_type
        if hasattr(refined_output, "images"):
            return refined_output.images
        elif isinstance(refined_output, dict) and "images" in refined_output:
            return refined_output["images"]
        else:
            return refined_output

    @torch.no_grad()
    def refine_images(
        self,
        images: torch.Tensor,
        prompt="",
        negative_prompt="",
        strength=None,
        num_inference_steps=50,
        guidance_scale=10,
        seed=None,
        high_noise_frac=0.8,
        aesthetic_score=4.0,
        negative_aesthetic_score=2.5,
        **kwargs,
    ):
        """
        Refine a batch of input images using the SDXL Refiner, torch-native version.

        Args:
            images: torch.Tensor in [0,1], shape [B, C, H, W]
            prompt: Text prompt (string or list of strings of length B)
            negative_prompt: Negative prompt (string or list of strings of length B)
            strength: Refinement strength
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            seed: Random seed for reproducibility
            high_noise_frac: Fraction of noise steps for base model (ensemble mode)

        Returns:
            List of refined images (PIL by default, unless output_type="pt" or "np" is set in kwargs)
        """
        assert isinstance(images, torch.Tensor), "images must be a torch.Tensor"
        assert images.ndim == 4, f"Expected [B,C,H,W], got {images.shape}"

        # Clamp to valid range [0,1]
        images = images.clamp(0, 1)

        # RNG
        generator = None
        if seed is not None:
            generator = torch.Generator(device=images.device).manual_seed(seed)

        refine_strength = strength if strength is not None else self.refiner_strength

        if self.base_pipe is not None:
            print(
                f"[INFO] Using ensemble refinement with base + refiner on {images.shape[0]} images..."
            )

            # Base model
            base_output = self.base_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=images,
                strength=refine_strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                denoising_end=high_noise_frac,
                output_type=kwargs.get("output_type", "latent"),
                generator=generator,
                **{k: v for k, v in kwargs.items() if k != "output_type"},
            )
            base_images = (
                base_output.images if hasattr(base_output, "images") else base_output
            )

            # Refiner
            refined_output = self.refiner(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=base_images,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                denoising_start=high_noise_frac,
                generator=generator,
                **kwargs,
            )
        else:
            # print(f"[INFO] Using direct refinement on {images.shape[0]} images...")
            refined_output = self.refiner(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=images,
                strength=refine_strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                aesthetic_score=aesthetic_score,
                negative_aesthetic_score=negative_aesthetic_score,
                generator=generator,
                **kwargs,
            )

        if hasattr(refined_output, "images"):
            return refined_output.images
        elif isinstance(refined_output, dict) and "images" in refined_output:
            return refined_output["images"]
        else:
            return refined_output

    def save_refined_image(self, refined_image, path):
        """Save refined image to disk."""
        if isinstance(refined_image, torch.Tensor):
            save_image(refined_image, path)
        else:
            refined_image.save(path)
        print(f'[INFO] Refined image saved to {path}')

    def set_refiner_strength(self, strength):
        """Set the default refinement strength."""
        self.refiner_strength = max(0.0, min(1.0, strength))
        print(f'[INFO] Refiner strength set to {self.refiner_strength}')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test SDXL Refiner utilities')
    parser.add_argument('--image', type=str, required=True, help='Path to input image for refinement')
    parser.add_argument('--output', type=str, default='refined_output.png', help='Output path for refined image')
    parser.add_argument('--prompt', type=str, default='', help='Prompt to guide refinement')
    parser.add_argument('--negative_prompt', type=str, default='', help='Negative prompt')
    parser.add_argument('--strength', type=float, default=0.3, help='Refinement strength (0.0-1.0)')
    parser.add_argument('--steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 precision')
    parser.add_argument('--vram_O', action='store_true', help='Optimize for VRAM')
    parser.add_argument(
        "--aesthetic_score",
        type=float,
        default=4.0,
        help="Aesthetic score for refinement",
    )
    parser.add_argument(
        "--negative_aesthetic_score",
        type=float,
        default=2.5,
        help="Negative aesthetic score for refinement",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=10,
        help="Guidance scale for classifier-free guidance",
    )
    parser.add_argument(
        "--lora_path", type=str, default=None, help="Path to LoRA weights for refiner"
    )

    args = parser.parse_args()

    # Initialize refiner
    refiner = StableDiffusionXLRefiner(
        device=args.device, fp16=args.fp16, vram_O=args.vram_O, lora_path=args.lora_path
    )

    # Load and refine image
    input_image = Image.open(args.image)

    refined_image = refiner.refine_image(
        image=input_image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        strength=args.strength,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        aesthetic_score=args.aesthetic_score,
        negative_aesthetic_score=args.negative_aesthetic_score,
    )[0]

    # Save result
    refined_image.save(args.output)
    print(f'[INFO] Refined image saved to {args.output}')
