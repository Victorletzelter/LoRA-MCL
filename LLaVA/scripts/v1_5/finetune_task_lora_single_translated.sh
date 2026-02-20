#!/bin/bash

#SBATCH --job-name=lora1hypTranslated
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=8:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=./checkpoints/1head-r16-alpha64-trans/log%j.out
#SBATCH --error=./checkpoints/1head-r16-alpha64-trans/log%j.err

deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 16 --lora_alpha 64 --mm_projector_lr 2e-5 \
    --num_hyps 1 \
    --deepspeed ./scripts/zero3.json \
    --wta_training_mode wta \
    --wta_params_epsilon 0.1 \
    --model_name_or_path liuhaotian/llava-v1.6-vicuna-7b \
    --version v1 \
    --data_train_path ./playground/data/textCapsTrainTranslatedHalf.json \
    --data_val_path ./playground/data/textCapsVal.json \
    --image_folder ./textcaps/train_images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/1head-r16-alpha64-trans \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 6 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "1head-r16-alpha64-trans"
