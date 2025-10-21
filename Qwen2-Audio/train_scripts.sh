# Extract the number of hyps from the command line

ERROR_MSG="Usage: $0 <dataset_name> <num_hyps> <wta_mode> <epsilon> <rank> <seed> <use_slurm>"

if [ -z "$1" ]; then
    echo "Usage: $0 "
    exit 1
else
    DATASET_NAME=$1
fi

if [ -z "$2" ]; then
    echo $ERROR_MSG
    exit 1
else
    NUM_HYPS=$2
fi

if [ -z "$3" ]; then
    echo $ERROR_MSG
    exit 1
else
    WTA_MODE=$3
fi

if [ -z "$4" ]; then
    echo $ERROR_MSG
    exit 1
else
    EPSILON=$4
fi

if [ -z "$5" ]; then
    echo $ERROR_MSG
    exit 1
else
    RANK=$5
fi

if [ -z "$6" ]; then
    echo $ERROR_MSG
    exit 1
else
    SEED=$6
fi

if [ -z "$7" ]; then
    echo $ERROR_MSG
    exit 1
else
    USE_SLURM=$7
fi

if [ -z "$8" ]; then
    echo $ERROR_MSG
    exit 1
else
    GROUP_LORA_ENABLED=$8
fi

# if dataset is Clotho, set MAX_LENGTH to 480000 (30s), else set to 160000 (10s)
# number of epochs is set to 10 for clotho and 1 for audiocaps
if [ "$DATASET_NAME" == "clotho" ]; then
    MAX_LENGTH=480000
    NUM_EPOCHS=10
elif [ "$DATASET_NAME" == "audiocaps" ]; then
    MAX_LENGTH=160000
    NUM_EPOCHS=1
else
    echo "Invalid dataset name"
    exit 1
fi

TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=2
EVAL_BATCH_SIZE=1

cd scripts

if [ "$USE_SLURM" == "True" ]; then
    sbatch --export=ALL,N_HYPS=$NUM_HYPS,NUM_EPOCHS=$NUM_EPOCHS,WTA_MODE=$WTA_MODE,TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE,EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE,DATASET_NAME=$DATASET_NAME,SEED=$SEED,EPSILON=$EPSILON,MAX_LENGTH=$MAX_LENGTH,RANK=$RANK,GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS,GROUP_LORA_ENABLED=$GROUP_LORA_ENABLED qwen_train.slurm
else
    export NUM_HYPS
    export NUM_EPOCHS
    export WTA_MODE
    export EPSILON
    export RANK
    export SEED
    export MAX_LENGTH
    export GRADIENT_ACCUMULATION_STEPS
    export TRAIN_BATCH_SIZE
    export EVAL_BATCH_SIZE
    export DATASET_NAME
    export USE_SLURM
    export GROUP_LORA_ENABLED
    bash qwen_train.sh
fi
