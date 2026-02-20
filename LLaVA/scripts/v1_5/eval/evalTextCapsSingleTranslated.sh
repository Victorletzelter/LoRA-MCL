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

python -m llava.eval.model_vqaSingle \
    --model-path liuhaotian/llava-v1.6-vicuna-7b \
    --question-file ./playground/data/textCapsTest_questions_translated.jsonl \
    --image-folder ./textcaps/train_images \
    --answers-file ./playground/data/textCapsTest_answers_translated.jsonl \
    --temperature 0.2 \
    --num_beams 2 \
    --lora_weights ${CKPT_PATH} \
    --lora_r 16 --lora_alpha 64 \
    --num_hyps 1 \
    --num_boost 1
