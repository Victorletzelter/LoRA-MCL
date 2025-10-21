#!/bin/bash

training_pairs="en-de"
LORA_RANK=16
NUM_SAMPLES=0
NUM_HYPS=3

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for EPSILON in 0.05; do
    your_output_dir="${PROJECT_DIR}/logs/3h_relaxed_epsilon_${EPSILON}"
    bash runs/parallel_ft_lora_mh_nh_relaxed.sh ${your_output_dir} ${training_pairs} ${LORA_RANK} ${NUM_SAMPLES} ${NUM_HYPS} ${EPSILON}
done
