from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, FlowMatchEulerDiscreteScheduler, StableDiffusionPipeline, StableDiffusion3Pipeline
from diffusers.utils.import_utils import is_xformers_available
from os.path import isfile
import os
from pathlib import Path

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from torch.cuda.amp import custom_bwd, custom_fwd
# from .perpneg_utils import weighted_perpendicular_aggregator


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

logger = logging.get_logger(__name__)
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
class StableDiffusion(nn.Module):
    def __init__(self, device, fp16, vram_O, sd_version='3.5', hf_key=None, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading stable diffusion...')

        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '3.5':#
            model_key = "stabilityai/stable-diffusion-3.5-medium"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        self.precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = StableDiffusion3Pipeline.from_pretrained(model_key, torch_dtype=self.precision_t, token=HUGGING_FACE_TOKEN)

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.transformer.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.tokenizer_3 = pipe.tokenizer_3
        self.text_encoder = pipe.text_encoder
        self.text_encoder_2 = pipe.text_encoder_2
        self.text_encoder_3 = pipe.text_encoder_3
        self.transformer = pipe.transformer

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=self.precision_t)

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])

        print(f'[INFO] loaded stable diffusion!')

    def _get_clip_prompt_embeds(self, prompt, clip_model_index, num_images_per_prompt=1, clip_skip=None):
        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]
        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        tokenizer_max_length = 77
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = text_encoder(text_input_ids.to(self.device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=self.device)

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds
    def _get_t5_prompt_embeds(
        self,
        prompt,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 256,
    ):
        device = self.device
        dtype = self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.text_encoder_3 is None:
            return torch.zeros(
                (
                    batch_size * num_images_per_prompt,
                    self.tokenizer_max_length,
                    self.transformer.config.joint_attention_dim,
                ),
                device=device,
                dtype=dtype,
            )

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_3.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_3(text_input_ids.to(device))[0]

        dtype = self.text_encoder_3.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds
    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # prompt: [str]
        prompt_2 = [prompt] if isinstance(prompt, str) else prompt
        prompt_3 = [prompt] if isinstance(prompt, str) else prompt
        prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
            prompt=prompt,
            clip_model_index=0,
        )
        prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
            prompt=prompt_2,
            clip_model_index=1,
        )
        clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

        t5_prompt_embed = self._get_t5_prompt_embeds(
            prompt=prompt_3,
        )
        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
        )
        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)
        return prompt_embeds, pooled_prompt_embeds
    
    @torch.no_grad()
    def refine(self, prompts, negative_prompts, pred_rgb, begin_index=600):
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        pos_embeds, pooled_pos_embeds = self.get_text_embeds(prompts) 
        neg_embeds, pooled_neg_embeds = self.get_text_embeds(negative_prompts) 
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0) 
        pooled_text_embeds = torch.cat([pooled_neg_embeds, pooled_pos_embeds], dim=0)

        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        # encode image into latents with vae, requires grad!
        latents = self.encode_imgs(pred_rgb_512)
        assert 2 * latents.shape[0] == text_embeds.shape[0]
        noise = torch.randn_like(latents)
        dummy_t = torch.tensor([[begin_index]] * latents.shape[0], dtype=torch.long, device=self.device)
        self.scheduler.set_timesteps(1000)
        self.scheduler.set_begin_index(begin_index)
        latents_noisy = self.scheduler.scale_noise(latents, dummy_t, noise)

        self.scheduler.set_timesteps(40)
        # produce_latents
        inference_begin_index = -(40 * begin_index) // 1000
        for i, t in enumerate(self.scheduler.timesteps[-inference_begin_index:]):
            latent_model_input = torch.cat([latents_noisy] * 2)
            timestep = t.expand(latent_model_input.shape[0]).to(self.device)
            noise_pred = self.transformer(hidden_states=latent_model_input, 
                                          timestep=timestep, 
                                          encoder_hidden_states=text_embeds,
                                          pooled_projections=pooled_text_embeds)[0]

            # perform guidance
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 4.5 * (noise_pred_pos - noise_pred_uncond)

            latents_noisy = self.scheduler.step(noise_pred, t, latents_noisy)['prev_sample']

        imgs = self.decode_latents(latents_noisy)
        return imgs

    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents + self.vae.config.shift_factor

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = (posterior.sample() - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        return latents


if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='3.5', choices=['1.5', '2.0', '2.1','3.5'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)
    from PIL import Image
    import numpy as np
    # rgb = Image.open("sd_origin.png")
    rgb = Image.open("/home/kwang/test/GaussianCube/experiments/test/render_images/rank_00_render_000000_cam_25.png")
    rgb = rgb.resize((opt.W, opt.H))
    rgb = torch.tensor(np.array(rgb)[:,:,:3]).permute(2, 0, 1).float() / 255.0
    rgb = rgb.unsqueeze(0).to(device)
    pos_prompts = [opt.prompt]# * 2
    neg_prompts = [opt.negative]# * 2
    # rgb = rgb.expand(2, 3, opt.H, opt.W)
    imgs = sd.refine(pos_prompts, neg_prompts, rgb)
    imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
    imgs = (imgs * 255).round().astype('uint8')
    for i in range(len(imgs)):
        plt.imshow(imgs[0])
        plt.axis('off')
        plt.savefig(f'sd_{i}.png', bbox_inches='tight')
        plt.close()



