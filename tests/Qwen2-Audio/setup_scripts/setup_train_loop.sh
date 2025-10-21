#!/bin/bash

DATASET_LIST=('clotho')
NUM_HYPS=3
WTA_MODE=relaxed-wta
EPSILON_LIST=(0.0005 0.05 0.1)
RANK=8
SEED=3141
USE_SLURM=False

cd tests/Qwen2-Audio/setup_scripts

for dataset in "${DATASET_LIST[@]}"; do
    # Train relaxed WTA
    for epsilon in "${EPSILON_LIST[@]}"; do
        bash setup_train_scripts.sh $dataset $NUM_HYPS relaxed-wta $epsilon $RANK $SEED $USE_SLURM
    done

    # Train vanilla WTA
    bash setup_train_scripts.sh $dataset $NUM_HYPS wta 0 $RANK $SEED $USE_SLURM

    # Train random WTA
    bash setup_train_scripts.sh $dataset $NUM_HYPS random-wta 0 $RANK $SEED $USE_SLURM

    # Train random relaxed WTA
    for epsilon in "${EPSILON_LIST[@]}"; do
        bash setup_train_scripts.sh $dataset $NUM_HYPS random-wta-relaxed $epsilon $RANK $SEED $USE_SLURM
    done

    # # Train 1 hypothesis MLE baseline with rank $RANK
    bash setup_train_scripts.sh $dataset 1 wta 0 $RANK $SEED $USE_SLURM

    # # Train 1 hypothesis MLE baseline with rank $NUM_HYPS*$RANK
    NEW_RANK=$((NUM_HYPS*RANK))
    bash setup_train_scripts.sh $dataset 1 wta 0 $NEW_RANK $SEED $USE_SLURM
done