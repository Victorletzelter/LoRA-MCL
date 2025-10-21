#!/bin/bash

training_pairs="en-de"
LORA_RANK=16
NUM_SAMPLES=2
NUM_HYPS=3

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$PROJECT_DIR/../../../ALMA/"

MAX_TRAIN_STEPS=1

for EPSILON in 0.05; do
    your_output_dir="${PROJECT_DIR}/logs/setup_3h_relaxed_epsilon_${EPSILON}"
    bash $PROJECT_DIR/runs/parallel_ft_lora_mh_nh_relaxed.sh ${your_output_dir} ${training_pairs} ${LORA_RANK} ${NUM_SAMPLES} ${NUM_HYPS} ${EPSILON} ${MAX_TRAIN_STEPS}
done
