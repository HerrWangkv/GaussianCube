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
        sd_version='2.1', 
        hf_key=None, 
        t_range=[0.02, 0.98]
    ):
        """
        Initialize Stable Diffusion model.
        
        Args:
            device: Device to run the model on
            fp16: Whether to use FP16 precision
            vram_O: Whether to optimize for VRAM usage
            sd_version: Stable Diffusion version ('2.1', '2.0', '1.5')
            hf_key: Custom HuggingFace model key
            t_range: Timestep range for SDS sampling [min, max]
        """
        super().__init__()

        self.device = device
        self.sd_version = sd_version
        self.t_range = t_range

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
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, 
            torch_dtype=self.precision_t,
            safety_checker=None,
            requires_safety_checker=False
        )

        # Memory optimization
        if vram_O:
            print('[INFO] Enabling VRAM optimizations...')
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            if hasattr(pipe, 'enable_model_cpu_offload'):
                pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        # Extract components
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        # Ensure all components are properly on the target device
        print(f"[INFO] Moving all StableDiffusion components to device {device}...")
        if not vram_O:  # Only force device placement if not using VRAM optimizations
            self.vae = self.vae.to(device)
            self.text_encoder = self.text_encoder.to(device)
            self.unet = self.unet.to(device)

            # Force all parameters to be on the correct device (important for distributed training)
            for param in self.text_encoder.parameters():
                param.data = param.data.to(device)
            for param in self.unet.parameters():
                param.data = param.data.to(device)
            for param in self.vae.parameters():
                param.data = param.data.to(device)

        # Create scheduler for SDS
        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, 
            subfolder="scheduler", 
            torch_dtype=self.precision_t
        )

        # Clean up pipeline
        del pipe

        # Set up timestep ranges
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(
            device
        )  # Use device parameter directly

        print(
            f"[INFO] Stable Diffusion loaded on device {device}! Using timestep range [{self.min_step}, {self.max_step}]"
        )

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
        inputs = self.tokenizer(
            prompts, 
            padding='max_length', 
            max_length=self.tokenizer.model_max_length, 
            return_tensors='pt'
        )

        # Encode
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

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
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

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

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def compute_sds_loss(
        self, 
        text_embeddings, 
        pred_rgb, 
        guidance_scale=100, 
        grad_scale=1,
        as_latent=False,
        save_guidance_path=None
    ):
        """
        Compute SDS (Score Distillation Sampling) loss.
        
        This is the core function that computes the SDS loss used to guide
        3D generation using a pretrained 2D diffusion model.
        
        Args:
            text_embeddings: Text embeddings [2*B, 77, 768] (uncond + cond)
            pred_rgb: Predicted RGB images [B, 3, H, W]
            guidance_scale: Classifier-free guidance scale
            grad_scale: Gradient scaling factor
            as_latent: Whether pred_rgb is already in latent space
            save_guidance_path: Path to save guidance visualization
            
        Returns:
            SDS loss scalar
        """
        batch_size = pred_rgb.shape[0]

        if as_latent:
            # Assume pred_rgb is already in latent space
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
            latents = latents.to(dtype=self.precision_t, device=self.device)
        else:
            # Encode RGB to latent space
            pred_rgb = pred_rgb.to(dtype=self.precision_t, device=self.device)
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_512)

        # Sample random timestep
        t = torch.randint(
            self.min_step, 
            self.max_step + 1, 
            (batch_size,), 
            dtype=torch.long, 
            device=self.device
        )

        # Add noise to latents
        noise = torch.randn_like(latents, dtype=self.precision_t, device=self.device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)

        # Predict noise with U-Net (no gradient)
        with torch.no_grad():
            # Duplicate for classifier-free guidance
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)

            # Predict noise
            noise_pred = self.unet(
                latent_model_input, 
                tt, 
                encoder_hidden_states=text_embeddings
            ).sample

            # Apply classifier-free guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # Compute SDS gradient
        # w(t) = (1 - alpha_t)
        w = (1 - self.alphas[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        # Save guidance visualization if requested
        if save_guidance_path:
            self._save_guidance_visualization(
                pred_rgb if not as_latent else self.decode_latents(latents),
                latents_noisy,
                noise_pred,
                t,
                save_guidance_path
            )

        # Compute SDS loss
        # The gradient flows through latents to pred_rgb
        targets = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / batch_size

        return loss

    @torch.no_grad()
    def _save_guidance_visualization(self, pred_rgb, latents_noisy, noise_pred, t, save_path):
        """Save visualization of the guidance process."""
        # Decode noisy latents
        result_noisier_image = self.decode_latents(latents_noisy)

        # Predict denoised image
        alphas = self.scheduler.alphas.to(latents_noisy)
        total_timesteps = self.max_step - self.min_step + 1
        index = total_timesteps - t.to(latents_noisy.device) - 1
        b = len(noise_pred)
        a_t = alphas[index].reshape(b, 1, 1, 1).to(self.device)
        sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
        sqrt_one_minus_at = sqrt_one_minus_alphas[index].reshape((b, 1, 1, 1)).to(self.device)

        pred_x0 = (latents_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt()
        result_denoised_image = self.decode_latents(pred_x0.to(self.precision_t))

        # Concatenate images for visualization
        viz_images = torch.cat([pred_rgb, result_noisier_image, result_denoised_image], dim=0)
        save_image(viz_images, save_path)

    @torch.no_grad() 
    def refine_image(self, prompts, negative_prompts, pred_rgb, num_steps=20, guidance_scale=7.5, strength=0.7):
        """
        Refine an image using diffusion denoising.
        
        Args:
            prompts: Text prompts
            negative_prompts: Negative text prompts  
            pred_rgb: Input image [B, 3, H, W]
            num_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            strength: Refinement strength (0.1=minimal, 0.9=major changes)
            
        Returns:
            Refined image [B, 3, H, W]
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Ensure correct data type for input
        pred_rgb = pred_rgb.to(dtype=self.precision_t, device=self.device)

        # Get text embeddings
        pos_embeds = self.get_text_embeds(prompts)
        neg_embeds = self.get_text_embeds(negative_prompts)
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0)

        # Encode to latents
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(pred_rgb_512)

        # Set up scheduler first
        self.scheduler.set_timesteps(num_steps)

        # For refinement, we start denoising from a middle timestep (not pure noise)
        # Choose a strength between 0.1 (minimal changes) and 0.9 (major changes)
        init_timestep = min(int(num_steps * strength), num_steps - 1)
        t_start = max(num_steps - init_timestep, 0)

        # Add noise corresponding to the chosen timestep
        timestep = self.scheduler.timesteps[t_start]
        noise = torch.randn_like(latents, dtype=self.precision_t, device=self.device)
        latents_noisy = self.scheduler.add_noise(latents, noise, timestep)

        # Use timesteps from the starting point
        timesteps = self.scheduler.timesteps[t_start:]

        # Denoising loop
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(
                latent_model_input, 
                t, 
                encoder_hidden_states=text_embeds
            ).sample

            # Apply guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # Step
            latents_noisy = self.scheduler.step(noise_pred, t, latents_noisy)['prev_sample']

        # Decode to image
        refined_imgs = self.decode_latents(latents_noisy)

        return refined_imgs

    @torch.no_grad()
    def generate_from_text(
        self, 
        prompts, 
        negative_prompts="", 
        height=512, 
        width=512, 
        num_steps=50, 
        guidance_scale=7.5,
        seed=None
    ):
        """
        Generate images from text prompts.
        
        Args:
            prompts: Text prompts
            negative_prompts: Negative prompts
            height: Image height
            width: Image width  
            num_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            seed: Random seed
            
        Returns:
            Generated images [B, 3, H, W]
        """
        if seed is not None:
            seed_everything(seed)

        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Get text embeddings
        pos_embeds = self.get_text_embeds(prompts)
        neg_embeds = self.get_text_embeds(negative_prompts)
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0)

        # Initialize random latents
        latents = torch.randn(
            (len(prompts), self.unet.in_channels, height // 8, width // 8),
            device=self.device,
            dtype=self.precision_t
        )

        # Set up scheduler
        self.scheduler.set_timesteps(num_steps)

        # Denoising loop
        for i, t in enumerate(self.scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeds
            ).sample

            # Apply guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # Step
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        # Decode to images
        images = self.decode_latents(latents)

        return images

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
    parser.add_argument('--num_steps', type=int, default=20, help='Number of denoising steps')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Classifier-free guidance scale')
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
    sd = StableDiffusion(device=device, fp16=use_fp16, sd_version=args.sd_version)
    
    # Load and preprocess image
    print(f"Loading image: {args.image}")
    image = Image.open(args.image).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)
    
    # Ensure correct data type and device
    image_tensor = image_tensor.to(device=device)
    
    print(f"Image tensor shape: {image_tensor.shape}")
    print(f"Image tensor dtype: {image_tensor.dtype}")
    print(f"Prompt: '{args.prompt}'")
    if args.negative_prompt:
        print(f"Negative prompt: '{args.negative_prompt}'")
    
    # Refine the image
    print(f"Refining image with strength {args.strength}...")
    with torch.no_grad():
        refined_image = sd.refine_image(
            prompts=args.prompt,
            negative_prompts=args.negative_prompt,
            pred_rgb=image_tensor,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            strength=args.strength
        )
    
    print(f"Refined image shape: {refined_image.shape}")
    
    # Save refined image
    refined_pil = transforms.ToPILImage()(refined_image.squeeze(0).cpu())
    refined_pil.save(args.output)
    
    print(f"Refined image saved to {args.output}")
    print("Image refinement completed successfully!")
