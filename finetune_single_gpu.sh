#!/bin/bash
# Fine-tuning script for GaussianCube with LoRA on Objaverse dataset
# Based on the original training command from README.md

# Run LoRA fine-tuning with the same distributed setup as original training

# Add proper GPU binding for MPI ranks
python finetune_smpl2.py \
    --exp_name ./output/gaussiancube_finetuning_smpl2_ema_tmp \
    --config configs/finetune_smpl2.yml \
    --model_name objaverse_v1.1 \
    --lr 5e-5 \
    --max_steps 10000 \
    --image_save_interval 1 \
    --use_fp16 \
    --use_tensorboard \
    --prompt_file human_prompts.txt \
    --lora_checkpoint output/gaussiancube_finetuning_smpl/checkpoints/step_004000/lora_model004000.pt \

echo "LoRA fine-tuning completed!"
echo "Check results in: ./output/gaussiancube_finetuning/"