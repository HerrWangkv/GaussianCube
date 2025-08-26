# SDXL Refiner Utils for GaussianCube

This module provides utilities for enhancing generated images using the Stable Diffusion XL Refiner model. It's designed to work seamlessly with GaussianCube's rendering pipeline to improve the quality and details of generated 3D content.

## Features

- **Single Image Refinement**: Enhance individual images with custom prompts
- **Batch Processing**: Refine multiple images efficiently
- **Ensemble Mode**: Use base SDXL + refiner for highest quality
- **Video Frame Refinement**: Process animation frames with temporal consistency
- **Memory Optimization**: VRAM-efficient processing for large images
- **Integration Ready**: Easy integration with GaussianCube workflows

## Installation

The refiner utilities use the existing dependencies in GaussianCube. Make sure you have:

```bash
# Core dependencies (already in requirements.txt)
pip install diffusers transformers torch torchvision
pip install accelerate huggingface_hub

# Optional for better performance
pip install xformers  # For memory efficiency
```

## Quick Start

### Basic Image Refinement

```python
from utils.refiner_utils import StableDiffusionXLRefiner
from PIL import Image

# Initialize refiner
refiner = StableDiffusionXLRefiner(
    device='cuda',
    fp16=True,
    vram_O=True,  # Enable VRAM optimizations
    refiner_strength=0.3
)

# Load and refine an image
image = Image.open('input.png')
refined_image = refiner.refine_image(
    image=image,
    prompt="high quality, detailed, masterpiece",
    negative_prompt="blurry, low quality",
    strength=0.3,
    num_inference_steps=50,
    seed=42
)

# Save result
refined_image.save('output_refined.png')
```

### Batch Processing

```python
# Refine multiple images at once
images = [Image.open(f'image_{i}.png') for i in range(5)]
prompts = ["detailed artwork", "photorealistic", "high quality render"]

refined_images = refiner.refine_batch(
    images=images,
    prompts=prompts,
    strength=0.25,
    num_inference_steps=30
)

# Save all results
for i, img in enumerate(refined_images):
    img.save(f'refined_{i}.png')
```

### High-Level Pipeline

```python
from utils.refiner_utils import SDXLRefinerPipeline

# Use the convenience pipeline
pipeline = SDXLRefinerPipeline(device='cuda', fp16=True)

# Enhance a directory of images
pipeline.enhance_images(
    input_images=['img1.png', 'img2.png', 'img3.png'],
    output_dir='enhanced_output/',
    prompts=['art style', 'photorealistic', 'high detail'],
    strength=0.3,
    num_steps=40
)
```

## Integration with GaussianCube

### Refine Rendered Images

```bash
# Refine GaussianCube rendered images
python scripts/gaussian_refiner_integration.py \
    --mode images \
    --input_dir output/render_images/ \
    --output_dir output/refined_images/ \
    --prompt "high quality 3D render, detailed"
```

### Refine Animation Frames

```bash
# Refine video frames with temporal consistency
python scripts/gaussian_refiner_integration.py \
    --mode frames \
    --input_dir output/video_frames/ \
    --output_dir output/refined_frames/ \
    --prompt "smooth animation, high quality" \
    --num_frames 100
```

### Create Comparisons

```bash
# Generate before/after comparison grids
python scripts/gaussian_refiner_integration.py \
    --mode comparison \
    --input_dir output/render_images/ \
    --refined_dir output/refined_images/ \
    --output_dir output/comparisons/
```

## Configuration Options

### Refiner Parameters

- **strength**: Refinement strength (0.0-1.0)
  - `0.1-0.2`: Subtle enhancement, preserves original closely
  - `0.3-0.4`: Moderate refinement, good balance
  - `0.5+`: Strong refinement, may alter content significantly

- **num_inference_steps**: Quality vs speed tradeoff
  - `20-30`: Fast, good for batch processing
  - `40-50`: Standard quality
  - `80+`: Maximum quality, slower

- **guidance_scale**: How closely to follow the prompt
  - `5-7`: Conservative, subtle changes
  - `7.5`: Default, balanced
  - `10+`: Strong prompt adherence

### Memory Optimization

For VRAM-limited systems:

```python
refiner = StableDiffusionXLRefiner(
    device='cuda',
    fp16=True,        # Use half precision
    vram_O=True,      # Enable CPU offloading
)
```

### Ensemble Mode (Best Quality)

```python
refiner = StableDiffusionXLRefiner(
    device='cuda',
    base_model_key="stabilityai/stable-diffusion-xl-base-1.0",  # Enable ensemble
    refiner_strength=0.3
)

refined = refiner.refine_image(
    image=input_image,
    high_noise_frac=0.8,  # Fraction for base model
    # ... other parameters
)
```

## Examples

Run the example scripts to see the refiner in action:

```bash
# Basic examples with synthetic images
python examples/refiner_examples.py --mode all

# Test with a real image
python examples/refiner_examples.py --mode real --input_image path/to/image.png

# Just batch processing example
python examples/refiner_examples.py --mode batch
```

## Performance Tips

1. **Batch Processing**: Process multiple images together for efficiency
2. **VRAM Optimization**: Enable `vram_O=True` for GPU memory constraints
3. **Resolution**: Start with 512x512, upscale separately if needed
4. **Caching**: The model weights are cached after first load
5. **Mixed Precision**: Use `fp16=True` for 2x memory savings

## Troubleshooting

### Common Issues

**CUDA Out of Memory**:
```python
# Enable VRAM optimizations
refiner = StableDiffusionXLRefiner(vram_O=True)

# Or reduce batch size
refined = refiner.refine_batch(images[:2])  # Process fewer at once
```

**Model Download Issues**:
```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/cache

# Or use custom model
refiner = StableDiffusionXLRefiner(hf_key="your-custom-model")
```

**Quality Issues**:
- Try different strength values (0.2-0.4 usually work well)
- Experiment with prompts - be specific about desired quality
- Use ensemble mode for highest quality
- Increase inference steps for better results

### Performance Benchmarks

Typical performance on RTX 4090:
- Single 512x512 image: ~3-5 seconds
- Batch of 4 images: ~10-12 seconds  
- Ensemble mode: ~8-10 seconds per image

## Model Information

The refiner uses these pretrained models:
- **Refiner**: `stabilityai/stable-diffusion-xl-refiner-1.0`
- **Base (optional)**: `stabilityai/stable-diffusion-xl-base-1.0`

Models are automatically downloaded and cached on first use.

## License

This utility follows the same license as the base GaussianCube project. The Stable Diffusion XL models have their own licenses - please check the HuggingFace model pages for details.

## Contributing

To contribute improvements to the refiner utilities:

1. Test your changes with the example scripts
2. Update documentation for new features
3. Consider performance impact on different hardware
4. Add error handling for edge cases

## Changelog

### v1.0.0
- Initial release with basic refinement capabilities
- Batch processing support
- Memory optimization options
- GaussianCube integration scripts
- Example usage scripts
