#!/usr/bin/env bash
set -euo pipefail

# Activate uv environment if present
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if conda env list | grep -q '^uv\s'; then
    conda activate uv
  fi
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."
cd ../.. # now at LLaVA root

python -u llava/train/train_mem.py \
  --model_name_or_path facebook/opt-125m \
  --version v0 \
  --data_train_path ./playground/data/mini_train.json \
  --data_val_path ./playground/data/mini_val.json \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "no" \
  --learning_rate 1e-4 \
  --logging_steps 1 \
  --bf16 False \
  --fp16 False \
  --num_hyps 1 \
  --dataloader_num_workers 0 \
  --lazy_preprocess True \
  --run_name "uv-test-run" \
  --output_dir ./checkpoints/uv_test
