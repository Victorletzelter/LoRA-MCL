#!/bin/bash

training_pairs="en-de"

LORA_RANK=16
num_samples=1
num_return_sequences=3

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo $PROJECT_DIR
PROJECT_DIR="$PROJECT_DIR/../../../ALMA"
echo $PROJECT_DIR

for multiplier in 1 3; do
    lora_rank=$((LORA_RANK * multiplier))
    your_output_dir="${PROJECT_DIR}/logs/setup_1h_rank_${lora_rank}"
    bash $PROJECT_DIR/runs/parallel_ft_lora_mh_1h_wta_eval_nt.sh ${your_output_dir} ${training_pairs} ${lora_rank} ${num_samples} ${num_return_sequences}
done