#!/bin/bash
# Fine-tuning script for GaussianCube with LoRA on Objaverse dataset
# Based on the original training command from README.md

# Run LoRA fine-tuning with the same distributed setup as original training
echo "Starting LoRA fine-tuning on Objaverse dataset..."
echo "Using 4 GPUs with MPI..."

# Set MPI environment variables to use writable temporary directories
export TMPDIR=${TMPDIR:-/tmp}
export ORTE_TMPDIR_BASE=${ORTE_TMPDIR_BASE:-/tmp}
export OMPI_MCA_orte_tmpdir_base=${OMPI_MCA_orte_tmpdir_base:-/tmp}

# Add proper GPU binding for MPI ranks
mpiexec -n 4 python finetune_smpl2.py \
    --exp_name ./output/gaussiancube_finetuning_smpl2_ema7 \
    --config configs/finetune_smpl2.yml \
    --model_name objaverse_v1.1 \
    --lr 5e-5 \
    --max_steps 15000 \
    --image_save_interval 50 \
    --use_fp16 \
    --use_tensorboard \
    --prompt_file human_prompts.txt \
    --resume_checkpoint output/gaussiancube_finetuning_smpl2_ema6/checkpoints/step_014000/lora_model014000.pt

echo "LoRA fine-tuning completed!"
echo "Check results in: ./output/gaussiancube_finetuning/"