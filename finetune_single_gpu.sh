#!/bin/bash
# Fine-tuning script for GaussianCube with LoRA on Objaverse dataset
# Based on the original training command from README.md

# Run LoRA fine-tuning with the same distributed setup as original training

# Add proper GPU binding for MPI ranks
python finetune_smpl.py \
    --exp_name ./output/finetune_smpl_tmp \
    --config configs/finetune_smpl.yml \
    --model_name objaverse_v1.1 \
    --lr 5e-5 \
    --max_steps 10000 \
    --image_save_interval 1 \
    --use_fp16 \
    --use_tensorboard \
    --resume_checkpoint output/finetune_smpl_0921_1537/checkpoints/step_005000/lora_model005000.pt \