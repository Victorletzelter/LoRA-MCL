#!/bin/bash

#SBATCH --job-name=1head-r24-alpha96b8SEEDVal
#SBATCH -C h100
#SBATCH --account=efs@h100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=./checkpoints/1head-r24-alpha96b8SEEDVal/log%j.out
#SBATCH --error=./checkpoints/1head-r24-alpha96b8SEEDVal/log%j.err

deepspeed llava/train/val_mem.py \
    --ckpt_path ./checkpoints/2heads01-transSlr2 \
    --lora_enable True --lora_r 8 --lora_alpha 32 --mm_projector_lr 2e-5 \
    --num_hyps 2 \
    --wta_training_mode wta \
    --wta_params_epsilon 0.1 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path liuhaotian/llava-v1.6-vicuna-7b \
    --version v1 \
    --data_train_path ./playground/data/textCapsTrain.json \
    --data_val_path ./playground/data/textCapsValFr.json \
    --image_folder ./textcaps/train_images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/1head-r24-alpha96b8SEEDVal \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "1head-r24-alpha96b8SEED"
