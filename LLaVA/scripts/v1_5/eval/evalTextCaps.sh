#!/bin/bash

#SBATCH --job-name=evalDBS
#SBATCH -C h100
#SBATCH --account=efs@h100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=./checkpoints/evalDBS/log%j.out
#SBATCH --error=./checkpoints/evalDBS/log%j.err

python -m llava.eval.model_vqa \
    --model-path liuhaotian/llava-v1.6-vicuna-7b \
    --question-file ./playground/data/textCapsTest_questions2.jsonl \
    --image-folder ./textcaps/train_images \
    --answers-file ./playground/data/textCapsTest_answers2.jsonl \
    --temperature 0.2 \
    --num_beams ${NUM_BOOST} \
    --lora_weights ${CKPT_PATH} \
    --lora_r 8 --lora_alpha 32 \
    --num_hyps 3
