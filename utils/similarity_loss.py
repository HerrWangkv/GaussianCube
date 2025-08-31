import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

class CLIPTextImageLoss(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Force safetensors only -> avoids torch.load() and the torch>=2.6 restriction
        self.model = CLIPModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_safetensors=True
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
