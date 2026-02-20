#!/bin/bash

#SBATCH --job-name=evalTrans
#SBATCH -C h100
#SBATCH --account=efs@h100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=./checkpoints/evalTrans/log%j.out
#SBATCH --error=./checkpoints/evalTrans/log%j.err

python -m llava.eval.model_vqa \
    --model-path liuhaotian/llava-v1.6-vicuna-7b \
    --question-file ./playground/data/textCapsTest_questions_translated2.jsonl \
    --image-folder ./textcaps/train_images \
    --answers-file ./playground/data/textCapsTest_answers_translated2.jsonl \
    --temperature 0.2 \
    --num_beams 1 \
    --lora_weights ${CKPT_PATH} \
    --lora_r 8 --lora_alpha 32 \
    --num_hyps 2 \
    --num_boost 1
