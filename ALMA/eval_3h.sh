#!/bin/bash

training_pairs="en-de"
LORA_RANK=16
NUM_SAMPLES=-1
NUM_HYPS=3
CKPT_DIR=${1:-"logs"}

echo "Evaluating 3-hypotheses model with rank 16 on ${CKPT_DIR}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for EPSILON in 0.05; do
    your_output_dir="${PROJECT_DIR}/${CKPT_DIR}/3h_relaxed_epsilon_${EPSILON}"
    bash runs/parallel_ft_lora_mh_nh_relaxed_eval_nt.sh ${your_output_dir} ${training_pairs} ${LORA_RANK} ${NUM_SAMPLES} ${NUM_HYPS} ${EPSILON}
done
