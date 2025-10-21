#!/bin/bash

DATASET_LIST=('clotho' 'audiocaps')
NUM_HYPS=5
EPSILON_LIST=(0.0005 0.05)
RANK=8
SEED=3141
USE_SLURM=False

for dataset in "${DATASET_LIST[@]}"; do
    GROUP_LORA_ENABLED=True
    # Train relaxed WTA
    for epsilon in "${EPSILON_LIST[@]}"; do
        echo "Running: bash train_scripts.sh $dataset $NUM_HYPS relaxed-wta $epsilon $RANK $SEED $USE_SLURM"
        bash train_scripts.sh $dataset $NUM_HYPS relaxed-wta $epsilon $RANK $SEED $USE_SLURM $GROUP_LORA_ENABLED
    done

    # Train vanilla WTA
    echo "Running: bash train_scripts.sh $dataset $NUM_HYPS wta 0 $RANK $SEED $USE_SLURM"
    bash train_scripts.sh $dataset $NUM_HYPS wta 0 $RANK $SEED $USE_SLURM $GROUP_LORA_ENABLED

    # Train MoE
    GROUP_LORA_ENABLED=False
    echo "Running: bash train_scripts.sh $dataset $NUM_HYPS moe 0 $RANK $SEED $USE_SLURM"
    bash train_scripts.sh $dataset $NUM_HYPS moe 0 $RANK $SEED $USE_SLURM $GROUP_LORA_ENABLED

    # Train annealed WTA
    GROUP_LORA_ENABLED=True
    echo "Running: bash train_scripts.sh $dataset $NUM_HYPS annealed-wta 0 $RANK $SEED $USE_SLURM"
    bash train_scripts.sh $dataset $NUM_HYPS annealed-wta 0 $RANK $SEED $USE_SLURM $GROUP_LORA_ENABLED

    # Train 1 hypothesis MLE baseline with rank $RANK
    GROUP_LORA_ENABLED=True
    echo "Running: bash train_scripts.sh $dataset 1 wta 0 $RANK $SEED $USE_SLURM"
    bash train_scripts.sh $dataset 1 wta 0 $RANK $SEED $USE_SLURM $GROUP_LORA_ENABLED

    # Train 1 hypothesis MLE baseline with rank $NUM_HYPS*$RANK
    NEW_RANK=$((NUM_HYPS*RANK))
    echo "Running: bash train_scripts.sh $dataset 1 wta 0 $NEW_RANK $SEED $USE_SLURM"
    bash train_scripts.sh $dataset 1 wta 0 $NEW_RANK $SEED $USE_SLURM $GROUP_LORA_ENABLED
done