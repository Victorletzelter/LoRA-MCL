#!/bin/bash

training_pairs="en-de"

LORA_RANK=16
num_samples=0
num_return_sequences=3

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for multiplier in 1 3; do
    lora_rank=$((LORA_RANK * multiplier))
    your_output_dir="${PROJECT_DIR}/logs/1h_rank_${lora_rank}"
    bash runs/parallel_ft_lora_mh_1h_wta.sh ${your_output_dir} ${training_pairs} ${lora_rank} ${num_samples} ${num_return_sequences}
done