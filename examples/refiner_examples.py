#!/usr/bin/env python3
"""
Example script demonstrating how to use the SDXL Refiner utilities.

This script shows various ways to use the refiner for image enhancement.
"""

import torch
import numpy as np
from PIL import Image
import argparse
import os
from pathlib import Path

from utils.refiner_utils import StableDiffusionXLRefiner, SDXLRefinerPipeline


def example_single_image_refinement():
    """Example of refining a single image."""
    print("=== Single Image Refinement Example ===")
    
    # Initialize refiner
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    refiner = StableDiffusionXLRefiner(
        device=device,
        fp16=True,
        vram_O=True,  # Enable VRAM optimizations
        refiner_strength=0.3
    )
    
    # Create a sample image (you would normally load from file)
    # For demo purposes, create a simple gradient image
    sample_image = Image.new('RGB', (512, 512), color=(128, 128, 128))
    
    # Refine the image
    refined_image = refiner.refine_image(
        image=sample_image,
        prompt="high quality, detailed, sharp",
        negative_prompt="blurry, low quality",
        strength=0.3,
        num_inference_steps=50,
        guidance_scale=7.5,
        seed=42
    )
    
    # Save the result
    output_dir = Path("output/refined_examples")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    refined_image.save(output_dir / "single_image_refined.png")
    print(f"Refined image saved to {output_dir / 'single_image_refined.png'}")
    
    return refined_image


def example_batch_refinement():
    """Example of refining multiple images in batch."""
    print("\n=== Batch Refinement Example ===")
    
    # Initialize refiner
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    refiner = StableDiffusionXLRefiner(
        device=device,
        fp16=True,
        vram_O=True
    )
    
    # Create sample images (you would normally load from files)
    sample_images = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
    for i, color in enumerate(colors):
        img = Image.new('RGB', (512, 512), color=color)
        sample_images.append(img)
    
    # Define prompts for each image
    prompts = [
        "vibrant red texture, high quality",
        "lush green nature, detailed",
        "deep blue ocean, crystal clear"
    ]
    
    # Refine the batch
    refined_images = refiner.refine_batch(
        images=sample_images,
        prompts=prompts,
        negative_prompts="low quality, blurry",
        strength=0.4,
        num_inference_steps=30,
        seed=123
    )
    
    # Save results
    output_dir = Path("output/refined_examples")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for i, refined_img in enumerate(refined_images):
        output_path = output_dir / f"batch_refined_{i:02d}.png"
        refined_img.save(output_path)
        print(f"Batch refined image {i} saved to {output_path}")
    
    return refined_images


def example_pipeline_usage():
    """Example using the high-level pipeline."""
    print("\n=== Pipeline Usage Example ===")
    
    # Initialize pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = SDXLRefinerPipeline(
        device=device,
        fp16=True,
        vram_O=True
    )
    
    # Create sample images
    sample_images = []
    for i in range(3):
        # Create different gradient images
        img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        sample_images.append(img)
    
    # Enhance images using pipeline
    output_dir = "output/pipeline_refined"
    enhanced_images = pipeline.enhance_images(
        input_images=sample_images,
        output_dir=output_dir,
        prompts=[
            "artistic masterpiece, highly detailed",
            "photorealistic, sharp focus",
            "beautiful composition, professional quality"
        ],
        strength=0.35,
        num_steps=40
    )
    
    print(f"Pipeline enhanced {len(enhanced_images)} images in {output_dir}")
    return enhanced_images


def example_with_real_image(image_path):
    """Example with a real input image."""
    print(f"\n=== Real Image Refinement: {image_path} ===")
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    # Initialize refiner with ensemble mode
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    refiner = StableDiffusionXLRefiner(
        device=device,
        fp16=True,
        vram_O=True,
        base_model_key="stabilityai/stable-diffusion-xl-base-1.0",  # Enable ensemble
        refiner_strength=0.3
    )
    
    # Load and refine image
    input_image = Image.open(image_path)
    
    refined_image = refiner.refine_image(
        image=input_image,
        prompt="masterpiece, best quality, highly detailed, sharp focus",
        negative_prompt="low quality, blurry, artifacts, distorted",
        strength=0.25,
        num_inference_steps=50,
        guidance_scale=7.5,
        seed=42,
        high_noise_frac=0.8  # For ensemble mode
    )
    
    # Save result
    output_dir = Path("output/refined_examples")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    input_name = Path(image_path).stem
    output_path = output_dir / f"{input_name}_refined.png"
    
    refined_image.save(output_path)
    print(f"Refined real image saved to {output_path}")
    
    return refined_image


def main():
    parser = argparse.ArgumentParser(description='SDXL Refiner Examples')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['single', 'batch', 'pipeline', 'real', 'all'],
                       help='Which example to run')
    parser.add_argument('--input_image', type=str, 
                       help='Path to input image for real image example')
    
    args = parser.parse_args()
    
    print("Starting SDXL Refiner Examples...")
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    try:
        if args.mode in ['single', 'all']:
            example_single_image_refinement()
        
        if args.mode in ['batch', 'all']:
            example_batch_refinement()
        
        if args.mode in ['pipeline', 'all']:
            example_pipeline_usage()
        
        if args.mode == 'real' and args.input_image:
            example_with_real_image(args.input_image)
        elif args.mode == 'real':
            print("Please provide --input_image for real image example")
            
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
    
    print("Examples completed!")


if __name__ == "__main__":
    main()
