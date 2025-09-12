import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

class CLIPTextImageLoss(nn.Module):

    def __init__(self, model_name="laion/CLIP-ViT-H-14-laion2B-s32B-b79K", device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Force safetensors only -> avoids torch.load() and the torch>=2.6 restriction
        self.model = CLIPModel.from_pretrained(
            model_name,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_safetensors=True,
        ).to(self.device)

        self.processor = CLIPProcessor.from_pretrained(model_name)

    def forward(self, gen_images, texts):
        """
        gen_images: Tensor of shape (B,3,H,W), values in [0,1] or [-1,1], requires_grad=True
        texts: list of strings, length B
        """
        # Encode text
        inputs = self.processor(text=texts, images=None, return_tensors="pt", padding=True).to(self.device)
        text_features = self.model.get_text_features(**inputs)
        text_features = F.normalize(text_features, dim=-1)

        # Resize to 224 and Normalize to CLIP stats
        gen_images = F.interpolate(
            gen_images, size=(224, 224), mode="bilinear", align_corners=False
        )
        CLIP_MEAN = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073], device=gen_images.device
        )[None, :, None, None]
        CLIP_STD = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711], device=gen_images.device
        )[None, :, None, None]
        gen_images = (gen_images - CLIP_MEAN) / CLIP_STD

        # Encode generated images
        image_features = self.model.get_image_features(pixel_values=gen_images)
        image_features = F.normalize(image_features, dim=-1)

        # Compute simple similarity loss (maximize cosine similarity)
        loss = 1 - (image_features * text_features).sum(dim=-1).mean()
        return loss

# Example usage
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = CLIPTextImageLoss(device=device)

    # Example batch
    gen_images = torch.randn(2, 3, 224, 224, device=device, requires_grad=True)
    texts = ["a red sports car", "a sunny beach"]

    loss = loss_fn(gen_images, texts)
    loss.backward()

    print("Text-Image similarity loss:", loss.item())
