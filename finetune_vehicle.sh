#!/bin/bash
# Fine-tuning script for GaussianCube with LoRA on Objaverse dataset
# Based on the original training command from README.md

# Run LoRA fine-tuning with the same distributed setup as original training
echo "Starting LoRA fine-tuning on Objaverse dataset..."
echo "Using 4 GPUs with MPI..."

# Set NCCL environment variables for better stability in Docker
export NCCL_P2P_DISABLE=1

# Add proper GPU binding for MPI ranks
mpiexec -n 4 python finetune_vehicle.py \
    --exp_name ./output/gaussiancube_finetuning_vehicle \
    --config configs/finetune_vehicle.yml \
    --model_name objaverse_v1.1 \
    --lr 5e-5 \
    --max_steps 50000 \
    --use_fp16 \
    --use_tensorboard \
    --prompt_file vehicle_prompts.txt \

echo "LoRA fine-tuning completed!"
echo "Check results in: ./output/gaussiancube_finetuning/"