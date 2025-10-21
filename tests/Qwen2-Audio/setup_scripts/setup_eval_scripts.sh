#!/bin/bash

#===============================================================================
# SCRIPT SETUP & ARGUMENT VALIDATION
#===============================================================================

DIR_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR=$DIR_SCRIPT

export PROJECT_DIR

if [ -z "$1" ]; then
    echo "Error: No argument provided"
    echo "Usage: $0 <number_of_hypotheses> <dataset_name>"
    exit 1
fi

ARGUMENT=$1 # number of hypotheses
DATASET_NAME=$2 # clotho or audiocaps

#===============================================================================
# EPSILON RENDERING CONFIGURATION
#===============================================================================

declare -A EPSILON_TO_RENDER
EPSILON_TO_RENDER[0.1]=0_1
EPSILON_TO_RENDER[0.05]=0_05
EPSILON_TO_RENDER[0.0005]=0_0005

#===============================================================================
# DATASET-SPECIFIC CONFIGURATION
#===============================================================================

if [[ "${DATASET_NAME}" = "clotho" ]]; then
    NUM_EPOCHS=10
    MAX_LENGTH=480000
elif [[ "${DATASET_NAME}" = "audiocaps" ]]; then
    NUM_EPOCHS=1
    MAX_LENGTH=160000
fi

SEED=1234

#===============================================================================
# SAMPLING METHOD PARAMETERS
#===============================================================================

# Base parameters
ALPHA0=0.
TYPICAL_P0=1.

# Vanilla sampling params (top-k)
TOP_P_VANILLA=1.0
TOP_K_VANILLA=50

# Nucleus sampling (top-p)
TOP_P_NUCLEUS=0.95
TOP_K_NUCLEUS=0

# Typical sampling
TOP_K_TYPICAL=0
TYPICAL_P=0.95

# Ancestral sampling
TOP_P_ANCESTRAL=1.0
TOP_K_ANCESTRAL=0

#===============================================================================
# SAMPLING CONFIGURATIONS LOOKUP
#===============================================================================

declare -A SAMPLING_CONFIGS
SAMPLING_CONFIGS["top_k"]="${TOP_K_VANILLA} ${TOP_P_VANILLA} ${TYPICAL_P0}"
SAMPLING_CONFIGS["top_p"]="${TOP_K_NUCLEUS} ${TOP_P_NUCLEUS} ${TYPICAL_P0}"  
SAMPLING_CONFIGS["typical"]="${TOP_K_TYPICAL} ${TOP_P_VANILLA} ${TYPICAL_P}"
SAMPLING_CONFIGS["ancestral"]="${TOP_K_ANCESTRAL} ${TOP_P_ANCESTRAL} ${TYPICAL_P0}"

#===============================================================================
# SLURM JOB STATIC PARAMETERS (exported via ALL)
#===============================================================================

TRAIN_BATCH_SIZE=2
EVAL_BATCH_SIZE=1
WTA_INI_TEMP=1.0
WTA_FIN_TEMP=1e-6
WTA_DECAY_RATE=0.999
WTA_SCHEDULE_MODE=global_step
CROSS_DATASET=False

#===============================================================================
# DEFAULT GENERATION CONFIG (can be overridden per job)
#===============================================================================

GENERATION_CONFIG_NUM_BEAM_GROUPS=1
GENERATION_CONFIG_DIVERSITY_PENALTY=0.0
GENERATION_CONFIG_ALPHA=${ALPHA0}
WTA_MODE=wta
RANK=8
EPSILON=0.0
GENERATION_CONFIG_TEMPERATURE=1.0
GENERATION_CONFIG_REPETITION_PENALTY=1.1

#===============================================================================
# TEST-TIME AUGMENTATION PARAMETERS
#===============================================================================

TIME_MASK_PERCENTAGE_VALUES=(0.2 0.4 0.6)
FREQ_MASK_PERCENTAGE_VALUES=(0.3 0.6 0.9)


#===============================================================================
# EXPERIMENT CONTROL FLAGS
#===============================================================================

DIVERSITY_PENALTY_VALUES=(0.5 0.8 1.0)
EPSILON_VALUES=(0.05)

one_hyp_baseline=False
random_wta=False
random_relaxed_wta=False
wta=False
relaxed_wta=False
annealed_wta=False
tta=False
moe_baseline=True
do_sampling=False
use_slurm=False

# Extract the path of the script
SCRIPT_PATH=$(realpath "${BASH_SOURCE[0]}")

cd $(dirname "$SCRIPT_PATH")/scripts

echo "SCRIPT_PATH: $SCRIPT_PATH"
echo "DIR_SCRIPT: $(dirname "$SCRIPT_PATH")/scripts"

launch_shell_script () {
    # Initialize associative array for parameters
    declare -A params
    
    # Set default values
    params[n_hyps]=1
    params[num_epochs]=1
    params[wta_mode]="wta"
    params[train_batch_size]=2
    params[eval_batch_size]=1
    params[dataset_name]=""
    params[ckpt_path]=""
    params[num_return_sequences]=1
    params[seed]=42
    params[name]="default"
    params[cross_dataset]="False"
    params[wta_ini_temp]=1.0
    params[wta_fin_temp]=1e-6
    params[wta_decay_rate]=0.999
    params[wta_schedule_mode]="global_step"
    params[max_length]=480000
    params[rank]=8
    params[epsilon]=0.05
    params[generation_config_beam_size]=1
    params[generation_config_do_sample]="False"
    params[generation_config_top_k]=50
    params[generation_config_top_p]=1.0
    params[generation_config_typical_p]=1.0
    params[generation_config_diversity_penalty]=0.0
    params[generation_config_num_beam_groups]=1
    params[generation_config_repetition_penalty]=1.1
    params[generation_config_temperature]=1.0
    # TTA-specific parameters (optional overrides)
    params[gaussian_noise_std]=""
    params[tta_enabled]=""
    params[tta_mode]=""
    params[time_mask_percentage]=""
    params[freq_mask_percentage]=""
    params[greedy_sh_tta]=""
    
    # Parse keyword arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --n_hyps=*)
                params[n_hyps]="${1#*=}"
                shift
                ;;
            --num_epochs=*)
                params[num_epochs]="${1#*=}"
                shift
                ;;
            --wta_mode=*)
                params[wta_mode]="${1#*=}"
                shift
                ;;
            --train_batch_size=*)
                params[train_batch_size]="${1#*=}"
                shift
                ;;
            --eval_batch_size=*)
                params[eval_batch_size]="${1#*=}"
                shift
                ;;
            --dataset_name=*)
                params[dataset_name]="${1#*=}"
                shift
                ;;
            --ckpt_path=*)
                params[ckpt_path]="${1#*=}"
                shift
                ;;
            --num_return_sequences=*)
                params[num_return_sequences]="${1#*=}"
                shift
                ;;
            --seed=*)
                params[seed]="${1#*=}"
                shift
                ;;
            --name=*)
                params[name]="${1#*=}"
                shift
                ;;
            --cross_dataset=*)
                params[cross_dataset]="${1#*=}"
                shift
                ;;
            --wta_ini_temp=*)
                params[wta_ini_temp]="${1#*=}"
                shift
                ;;
            --wta_fin_temp=*)
                params[wta_fin_temp]="${1#*=}"
                shift
                ;;
            --wta_decay_rate=*)
                params[wta_decay_rate]="${1#*=}"
                shift
                ;;
            --wta_schedule_mode=*)
                params[wta_schedule_mode]="${1#*=}"
                shift
                ;;
            --max_length=*)
                params[max_length]="${1#*=}"
                shift
                ;;
            --rank=*)
                params[rank]="${1#*=}"
                shift
                ;;
            --epsilon=*)
                params[epsilon]="${1#*=}"
                shift
                ;;
            --generation_config_beam_size=*)
                params[generation_config_beam_size]="${1#*=}"
                shift
                ;;
            --generation_config_do_sample=*)
                params[generation_config_do_sample]="${1#*=}"
                shift
                ;;
            --generation_config_top_k=*)
                params[generation_config_top_k]="${1#*=}"
                shift
                ;;
            --generation_config_top_p=*)
                params[generation_config_top_p]="${1#*=}"
                shift
                ;;
            --generation_config_typical_p=*)
                params[generation_config_typical_p]="${1#*=}"
                shift
                ;;
            --generation_config_diversity_penalty=*)
                params[generation_config_diversity_penalty]="${1#*=}"
                shift
                ;;
            --generation_config_num_beam_groups=*)
                params[generation_config_num_beam_groups]="${1#*=}"
                shift
                ;;
            --gaussian_noise_std=*)
                params[gaussian_noise_std]="${1#*=}"
                shift
                ;;
            --tta_enabled=*)
                params[tta_enabled]="${1#*=}"
                shift
                ;;
            --tta_mode=*)
                params[tta_mode]="${1#*=}"
                shift
                ;;
            --time_mask_percentage=*)
                params[time_mask_percentage]="${1#*=}"
                shift
                ;;
            --freq_mask_percentage=*)
                params[freq_mask_percentage]="${1#*=}"
                shift
                ;;
            --greedy_sh_tta=*)
                params[greedy_sh_tta]="${1#*=}"
                shift
                ;;
            *)
                echo "Unknown parameter: $1"
                return 1
                ;;
        esac
    done
    
    # Export the parameters
    export N_HYPS="${params[n_hyps]}"
    export NUM_EPOCHS="${params[num_epochs]}"
    export WTA_MODE="${params[wta_mode]}"
    export TRAIN_BATCH_SIZE="${params[train_batch_size]}"
    export EVAL_BATCH_SIZE="${params[eval_batch_size]}"
    export DATASET_NAME="${params[dataset_name]}"
    export CKPT_PATH="${params[ckpt_path]}"
    export NUM_RETURN_SEQUENCES="${params[num_return_sequences]}"
    export SEED="${params[seed]}"
    export NAME="${params[name]}"
    export CROSS_DATASET="${params[cross_dataset]}"
    export WTA_INI_TEMP="${params[wta_ini_temp]}"
    export WTA_FIN_TEMP="${params[wta_fin_temp]}"
    export WTA_DECAY_RATE="${params[wta_decay_rate]}"
    export WTA_SCHEDULE_MODE="${params[wta_schedule_mode]}"
    export MAX_LENGTH="${params[max_length]}"
    export RANK="${params[rank]}"
    export EPSILON="${params[epsilon]}"
    
    # Export generation config parameters (override the global ones if provided)
    export GENERATION_CONFIG_BEAM_SIZE="${params[generation_config_beam_size]}"
    export GENERATION_CONFIG_DO_SAMPLE="${params[generation_config_do_sample]}"
    export GENERATION_CONFIG_TOP_K="${params[generation_config_top_k]}"
    export GENERATION_CONFIG_TOP_P="${params[generation_config_top_p]}"
    export GENERATION_CONFIG_TYPICAL_P="${params[generation_config_typical_p]}"
    export GENERATION_CONFIG_DIVERSITY_PENALTY="${params[generation_config_diversity_penalty]}"
    export GENERATION_CONFIG_NUM_BEAM_GROUPS="${params[generation_config_num_beam_groups]}"
    
        # Export TTA parameters (override global values if provided)
    if [[ -n "${params[gaussian_noise_std]}" ]]; then
        export GAUSSIAN_NOISE_STD="${params[gaussian_noise_std]}"
    elif [[ -n "${GAUSSIAN_NOISE_STD:-}" ]]; then
        export GAUSSIAN_NOISE_STD="$GAUSSIAN_NOISE_STD"
    fi
    
    if [[ -n "${params[tta_enabled]}" ]]; then
        export TTA_ENABLED="${params[tta_enabled]}"
    elif [[ -n "${TTA_ENABLED:-}" ]]; then
        export TTA_ENABLED="$TTA_ENABLED"
    fi
    
    if [[ -n "${params[tta_mode]}" ]]; then
        export TTA_MODE="${params[tta_mode]}"
    elif [[ -n "${TTA_MODE:-}" ]]; then
        export TTA_MODE="$TTA_MODE"
    fi
    
    if [[ -n "${params[time_mask_percentage]}" ]]; then
        export TIME_MASK_PERCENTAGE="${params[time_mask_percentage]}"
    elif [[ -n "${TIME_MASK_PERCENTAGE:-}" ]]; then
        export TIME_MASK_PERCENTAGE="$TIME_MASK_PERCENTAGE"
    fi
    
    if [[ -n "${params[freq_mask_percentage]}" ]]; then
        export FREQ_MASK_PERCENTAGE="${params[freq_mask_percentage]}"
    elif [[ -n "${FREQ_MASK_PERCENTAGE:-}" ]]; then
        export FREQ_MASK_PERCENTAGE="$FREQ_MASK_PERCENTAGE"
    fi
    
    if [[ -n "${params[greedy_sh_tta]}" ]]; then
        export GREEDY_SH_TTA="${params[greedy_sh_tta]}"
    elif [[ -n "${GREEDY_SH_TTA:-}" ]]; then
        export GREEDY_SH_TTA="$GREEDY_SH_TTA"
    fi
    export GENERATION_CONFIG_TEMPERATURE
    export GENERATION_CONFIG_REPETITION_PENALTY
    export GENERATION_CONFIG_ALPHA

    echo "CKPT_PATH: $CKPT_PATH"
    echo "N_HYPS: $N_HYPS"

    bash setup_qwen_eval.sh
}

#===============================================================================
# MAIN EXPERIMENT LOOP
#===============================================================================

for i in ${ARGUMENT}; do

    #===========================================================================
    # BASELINE EXPERIMENTS (Single Hypothesis)
    #===========================================================================
    
    if [ "${one_hyp_baseline}" = "True" ]; then

        echo "BASELINE EXPERIMENTS (Single Hypothesis)"

        for rank in 8 $((8*i)); do
            var_name="CKPT_1_HYP_RANK${rank}_${DATASET_NAME^^}"
            echo "CKPT_PATH: ${!var_name}"
            
            # MAP Decoding
            for multiplier in 1 2 ${i}; do
                # Diverse Beam Search	
                for value in "${DIVERSITY_PENALTY_VALUES[@]}"; do
                    if [ "${use_slurm}" = "True" ]; then
                        sbatch --export=ALL,N_HYPS=1,NUM_EPOCHS=${NUM_EPOCHS},DATASET_NAME=${DATASET_NAME},CKPT_PATH=${!var_name},NUM_RETURN_SEQUENCES=${i},GENERATION_CONFIG_TOP_K=${TOP_K_VANILLA},GENERATION_CONFIG_TOP_P=${TOP_P_VANILLA},GENERATION_CONFIG_BEAM_SIZE=$((multiplier*i)),GENERATION_CONFIG_DO_SAMPLE=False,GENERATION_CONFIG_NUM_BEAM_GROUPS=${i},GENERATION_CONFIG_DIVERSITY_PENALTY=${value},SEED=${SEED},GENERATION_CONFIG_TYPICAL_P=${TYPICAL_P0},NAME=dbs_multiplier_${multiplier},MAX_LENGTH=${MAX_LENGTH},RANK=${rank} setup_qwen_eval.slurm
                    else
                        echo ${!var_name}
                        launch_shell_script \
                            --n_hyps=1 \
                            --num_epochs=${NUM_EPOCHS} \
                            --wta_mode=${WTA_MODE} \
                            --train_batch_size=${TRAIN_BATCH_SIZE} \
                            --eval_batch_size=${EVAL_BATCH_SIZE} \
                            --dataset_name=${DATASET_NAME} \
                            --ckpt_path=${!var_name} \
                            --num_return_sequences=${i} \
                            --seed=${SEED} \
                            --name="dbs_multiplier_${multiplier}" \
                            --cross_dataset=False \
                            --max_length=${MAX_LENGTH} \
                            --rank=${rank} \
                            --generation_config_beam_size=$((multiplier*i)) \
                            --generation_config_do_sample=False \
                            --generation_config_top_k=${TOP_K_VANILLA} \
                            --generation_config_top_p=${TOP_P_VANILLA} \
                            --generation_config_typical_p=${TYPICAL_P0} \
                            --generation_config_diversity_penalty=${value} \
                            --generation_config_num_beam_groups=${i}
                    fi
                done
                
                # Beam Search
                if [ "${use_slurm}" = "True" ]; then
                    sbatch --export=ALL,N_HYPS=1,NUM_EPOCHS=${NUM_EPOCHS},DATASET_NAME=${DATASET_NAME},CKPT_PATH=${!var_name},NUM_RETURN_SEQUENCES=${i},GENERATION_CONFIG_TOP_K=${TOP_K_VANILLA},GENERATION_CONFIG_TOP_P=${TOP_P_VANILLA},GENERATION_CONFIG_BEAM_SIZE=$((multiplier*i)),GENERATION_CONFIG_DO_SAMPLE=False,SEED=${SEED},GENERATION_CONFIG_TYPICAL_P=${TYPICAL_P0},NAME=bs_multiplier_${multiplier},MAX_LENGTH=${MAX_LENGTH},RANK=${rank} setup_qwen_eval.slurm
                else
                    echo ${!var_name}
                    launch_shell_script \
                        --n_hyps=1 \
                        --num_epochs=${NUM_EPOCHS} \
                        --wta_mode=${WTA_MODE} \
                        --train_batch_size=${TRAIN_BATCH_SIZE} \
                        --eval_batch_size=${EVAL_BATCH_SIZE} \
                        --dataset_name=${DATASET_NAME} \
                        --ckpt_path=${!var_name} \
                        --num_return_sequences=${i} \
                        --seed=${SEED} \
                        --name="bs_multiplier_${multiplier}" \
                        --cross_dataset=False \
                        --max_length=${MAX_LENGTH} \
                        --rank=${rank} \
                        --generation_config_beam_size=$((multiplier*i)) \
                        --generation_config_do_sample=False \
                        --generation_config_top_k=${TOP_K_VANILLA} \
                        --generation_config_top_p=${TOP_P_VANILLA} \
                        --generation_config_typical_p=${TYPICAL_P0}
                fi
            done

            if [ "${do_sampling}" = "True" ]; then
                # Sampling methods (top-k, nucleus, typical, ancestral)
                for method in "${!SAMPLING_CONFIGS[@]}"; do
                    config=(${SAMPLING_CONFIGS[$method]})
                    if [ "${use_slurm}" = "True" ]; then
                    sbatch --export=ALL,N_HYPS=1,NUM_EPOCHS=${NUM_EPOCHS},DATASET_NAME=${DATASET_NAME},CKPT_PATH=${!var_name},NUM_RETURN_SEQUENCES=${i},GENERATION_CONFIG_TOP_K=${config[0]},GENERATION_CONFIG_TOP_P=${config[1]},GENERATION_CONFIG_BEAM_SIZE=1,GENERATION_CONFIG_DO_SAMPLE=True,SEED=${SEED},GENERATION_CONFIG_TYPICAL_P=${config[2]},NAME=${method},MAX_LENGTH=${MAX_LENGTH},RANK=${rank} setup_qwen_eval.slurm
                    else
                        echo ${!var_name}
                        launch_shell_script \
                            --n_hyps=1 \
                            --num_epochs=${NUM_EPOCHS} \
                            --wta_mode=${WTA_MODE} \
                            --train_batch_size=${TRAIN_BATCH_SIZE} \
                            --eval_batch_size=${EVAL_BATCH_SIZE} \
                            --dataset_name=${DATASET_NAME} \
                            --ckpt_path=${!var_name} \
                            --num_return_sequences=${i} \
                            --seed=${SEED} \
                            --name=${method} \
                            --cross_dataset=False \
                            --max_length=${MAX_LENGTH} \
                            --rank=${rank} \
                            --generation_config_beam_size=1 \
                            --generation_config_do_sample=True \
                            --generation_config_top_k=${config[0]} \
                            --generation_config_top_p=${config[1]} \
                            --generation_config_typical_p=${config[2]}
                    fi
                done
            fi

            #===========================================================================
            # TEST-TIME AUGMENTATION (TTA) EXPERIMENTS (for each rank in the baseline model)
            #===========================================================================
            
            if [ "${tta}" = "True" ]; then
                var_name="CKPT_1_HYP_RANK${rank}_${DATASET_NAME^^}"
                
                # Spec Augment with Greedy and Beam Search
                for j in ${!TIME_MASK_PERCENTAGE_VALUES[@]}; do
                    value_time=${TIME_MASK_PERCENTAGE_VALUES[$j]}
                    value_freq=${FREQ_MASK_PERCENTAGE_VALUES[$j]}
                    
                    # Beam Search with Greedy decoding
                    if [ "${use_slurm}" = "True" ]; then
                        sbatch --export=ALL,N_HYPS=1,NUM_EPOCHS=${NUM_EPOCHS},DATASET_NAME=${DATASET_NAME},CKPT_PATH=${!var_name},NUM_RETURN_SEQUENCES=${i},GENERATION_CONFIG_TOP_K=${TOP_K_VANILLA},GENERATION_CONFIG_TOP_P=${TOP_P_VANILLA},GENERATION_CONFIG_BEAM_SIZE=${i},GENERATION_CONFIG_DO_SAMPLE=False,SEED=${SEED},GENERATION_CONFIG_TYPICAL_P=${TYPICAL_P0},NAME=greedyttaspecaugment,GAUSSIAN_NOISE_STD=0.0,TTA_ENABLED=True,TTA_MODE=spec_augment,TIME_MASK_PERCENTAGE=${value_time},FREQ_MASK_PERCENTAGE=${value_freq},GREEDY_SH_TTA=Single,MAX_LENGTH=${MAX_LENGTH},RANK=${rank} setup_qwen_eval.slurm
                    else
                        echo ${!var_name}
                        launch_shell_script \
                            --n_hyps=1 \
                            --num_epochs=${NUM_EPOCHS} \
                            --wta_mode=${WTA_MODE} \
                            --train_batch_size=${TRAIN_BATCH_SIZE} \
                            --eval_batch_size=${EVAL_BATCH_SIZE} \
                            --dataset_name=${DATASET_NAME} \
                            --ckpt_path=${!var_name} \
                            --num_return_sequences=1 \
                            --seed=${SEED} \
                            --name=greedyttaspecaugment \
                            --max_length=${MAX_LENGTH} \
                            --rank=${rank} \
                            --generation_config_beam_size=${i} \
                            --generation_config_do_sample=False \
                            --generation_config_top_k=${TOP_K_VANILLA} \
                            --generation_config_top_p=${TOP_P_VANILLA} \
                            --generation_config_typical_p=${TYPICAL_P0} \
                            --gaussian_noise_std=0.0 \
                            --tta_enabled=True \
                            --tta_mode=spec_augment \
                            --time_mask_percentage=${value_time} \
                            --freq_mask_percentage=${value_freq} \
                            --greedy_sh_tta=Single
                    fi

                    # Beam Search with BS = 2 decoding in each forward pass
                    if [ "${use_slurm}" = "True" ]; then
                        sbatch --export=ALL,N_HYPS=1,NUM_EPOCHS=${NUM_EPOCHS},DATASET_NAME=${DATASET_NAME},CKPT_PATH=${!var_name},NUM_RETURN_SEQUENCES=${i},GENERATION_CONFIG_TOP_K=${TOP_K_VANILLA},GENERATION_CONFIG_TOP_P=${TOP_P_VANILLA},GENERATION_CONFIG_BEAM_SIZE=${i},GENERATION_CONFIG_DO_SAMPLE=False,SEED=${SEED},GENERATION_CONFIG_TYPICAL_P=${TYPICAL_P0},NAME=greedyttaspecaugment,GAUSSIAN_NOISE_STD=0.0,TTA_ENABLED=True,TTA_MODE=spec_augment,TIME_MASK_PERCENTAGE=${value_time},FREQ_MASK_PERCENTAGE=${value_freq},GREEDY_SH_TTA=Single,MAX_LENGTH=${MAX_LENGTH},RANK=${rank} setup_qwen_eval.slurm
                    else
                        echo ${!var_name}
                        launch_shell_script \
                            --n_hyps=1 \
                            --num_epochs=${NUM_EPOCHS} \
                            --wta_mode=${WTA_MODE} \
                            --train_batch_size=${TRAIN_BATCH_SIZE} \
                            --eval_batch_size=${EVAL_BATCH_SIZE} \
                            --dataset_name=${DATASET_NAME} \
                            --ckpt_path=${!var_name} \
                            --num_return_sequences=1 \
                            --seed=${SEED} \
                            --name=greedyttaspecaugment \
                            --max_length=${MAX_LENGTH} \
                            --rank=${rank} \
                            --generation_config_beam_size=${i} \
                            --generation_config_do_sample=False \
                            --generation_config_top_k=${TOP_K_VANILLA} \
                            --generation_config_top_p=${TOP_P_VANILLA} \
                            --generation_config_typical_p=${TYPICAL_P0} \
                            --gaussian_noise_std=0.0 \
                            --tta_enabled=True \
                            --tta_mode=spec_augment \
                            --time_mask_percentage=${value_time} \
                            --freq_mask_percentage=${value_freq} \
                            --greedy_sh_tta=Double
                    fi
                    
                    # Beam Search with beam size = i
                    for beam_size in ${i}; do
                        if [ "${use_slurm}" = "True" ]; then
                            sbatch --export=ALL,N_HYPS=1,NUM_EPOCHS=${NUM_EPOCHS},DATASET_NAME=${DATASET_NAME},CKPT_PATH=${!var_name},NUM_RETURN_SEQUENCES=${i},GENERATION_CONFIG_TOP_K=${TOP_K_VANILLA},GENERATION_CONFIG_TOP_P=${TOP_P_VANILLA},GENERATION_CONFIG_BEAM_SIZE=${beam_size},GENERATION_CONFIG_DO_SAMPLE=False,SEED=${SEED},GENERATION_CONFIG_TYPICAL_P=${TYPICAL_P0},NAME=bsttaspecaugment,GAUSSIAN_NOISE_STD=0.0,TTA_ENABLED=True,TTA_MODE=spec_augment,TIME_MASK_PERCENTAGE=${value_time},FREQ_MASK_PERCENTAGE=${value_freq},MAX_LENGTH=${MAX_LENGTH},RANK=${rank} setup_qwen_eval.slurm
                        else
                            echo ${!var_name}
                            launch_shell_script \
                                --n_hyps=1 \
                                --num_epochs=${NUM_EPOCHS} \
                                --wta_mode=${WTA_MODE} \
                                --train_batch_size=${TRAIN_BATCH_SIZE} \
                                --eval_batch_size=${EVAL_BATCH_SIZE} \
                                --dataset_name=${DATASET_NAME} \
                                --ckpt_path=${!var_name} \
                                --num_return_sequences=${i} \
                                --seed=${SEED} \
                                --name=bsttaspecaugment \
                                --max_length=${MAX_LENGTH} \
                                --rank=${rank} \
                                --generation_config_beam_size=${beam_size} \
                                --generation_config_do_sample=False \
                                --generation_config_top_k=${TOP_K_VANILLA} \
                                --generation_config_top_p=${TOP_P_VANILLA} \
                                --generation_config_typical_p=${TYPICAL_P0} \
                                --gaussian_noise_std=0.0 \
                                --tta_enabled=True \
                                --tta_mode=spec_augment \
                                --time_mask_percentage=${value_time} \
                                --freq_mask_percentage=${value_freq} \
                                --greedy_sh_tta=False
                        fi
                    done
                done
            fi
        
        done
    fi

    #===========================================================================
    # STANDARD WTA EXPERIMENTS
    #===========================================================================
    
    if [ "${wta}" = "True" ]; then
        echo "STANDARD WTA EXPERIMENTS"
        epsilon=0.0
        var_name="CKPT_${i}_HYP_WTA_${DATASET_NAME^^}"
        
        # Beam Search with different beam sizes
        for beam_size in 1 2 ${i}; do     
            if [ "${use_slurm}" = "True" ]; then
                sbatch --export=ALL,N_HYPS=${i},NUM_EPOCHS=${NUM_EPOCHS},DATASET_NAME=${DATASET_NAME},CKPT_PATH=${!var_name},NUM_RETURN_SEQUENCES=1,GENERATION_CONFIG_TOP_K=${TOP_K_VANILLA},GENERATION_CONFIG_TOP_P=${TOP_P_VANILLA},GENERATION_CONFIG_BEAM_SIZE=${beam_size},GENERATION_CONFIG_DO_SAMPLE=False,EPSILON=${epsilon},SEED=${SEED},GENERATION_CONFIG_TYPICAL_P=${TYPICAL_P0},NAME=bs_${beam_size},MAX_LENGTH=${MAX_LENGTH} setup_qwen_eval.slurm
            else
                echo ${!var_name}
                launch_shell_script \
                    --n_hyps=${i} \
                    --num_epochs=${NUM_EPOCHS} \
                    --wta_mode=wta \
                    --dataset_name=${DATASET_NAME} \
                    --ckpt_path=${!var_name} \
                    --num_return_sequences=1 \
                    --seed=${SEED} \
                    --name=bs_${beam_size} \
                    --max_length=${MAX_LENGTH} \
                    --epsilon=${epsilon} \
                    --generation_config_beam_size=${beam_size} \
                    --generation_config_do_sample=False \
                    --generation_config_top_k=${TOP_K_VANILLA} \
                    --generation_config_top_p=${TOP_P_VANILLA} \
                    --generation_config_typical_p=${TYPICAL_P0}
            fi
        done
        
        # Sampling methods (top-k, nucleus, typical, ancestral)
        if [ "${do_sampling}" = "True" ]; then
            for method in "${!SAMPLING_CONFIGS[@]}"; do
                config=(${SAMPLING_CONFIGS[$method]})
                if [ "${use_slurm}" = "True" ]; then
                    sbatch --export=ALL,N_HYPS=${i},NUM_EPOCHS=${NUM_EPOCHS},DATASET_NAME=${DATASET_NAME},CKPT_PATH=${!var_name},NUM_RETURN_SEQUENCES=1,GENERATION_CONFIG_TOP_K=${config[0]},GENERATION_CONFIG_TOP_P=${config[1]},GENERATION_CONFIG_BEAM_SIZE=1,GENERATION_CONFIG_DO_SAMPLE=True,EPSILON=${epsilon},SEED=${SEED},GENERATION_CONFIG_TYPICAL_P=${config[2]},NAME=${method},MAX_LENGTH=${MAX_LENGTH} setup_qwen_eval.slurm
                else
                    echo ${!var_name}
                    launch_shell_script \
                        --n_hyps=${i} \
                        --num_epochs=${NUM_EPOCHS} \
                        --wta_mode=wta \
                        --dataset_name=${DATASET_NAME} \
                        --ckpt_path=${!var_name} \
                        --num_return_sequences=1 \
                        --seed=${SEED} \
                        --name=${method} \
                        --max_length=${MAX_LENGTH} \
                        --epsilon=${epsilon} \
                        --generation_config_beam_size=1 \
                        --generation_config_do_sample=True \
                        --generation_config_top_k=${config[0]} \
                        --generation_config_top_p=${config[1]} \
                        --generation_config_typical_p=${config[2]}
                fi
            done
        fi
    fi

    #===========================================================================
    # RELAXED WTA EXPERIMENTS (with epsilon variations)
    #===========================================================================
    
    if [ "${relaxed_wta}" = "True" ]; then
        WTA_MODE=relaxed-wta
        echo "RELAXED WTA EXPERIMENTS"
        for epsilon in "${EPSILON_VALUES[@]}"; do
            var_name="CKPT_${i}_HYP_EPSILON_${EPSILON_TO_RENDER[${epsilon}]}_${DATASET_NAME^^}"
            
            # Beam Search with different beam sizes
            for beam_size in 1 2 ${i}; do
                if [ "${use_slurm}" = "True" ]; then
                    sbatch --export=ALL,N_HYPS=${i},NUM_EPOCHS=${NUM_EPOCHS},DATASET_NAME=${DATASET_NAME},CKPT_PATH=${!var_name},NUM_RETURN_SEQUENCES=1,GENERATION_CONFIG_TOP_K=${TOP_K_VANILLA},GENERATION_CONFIG_TOP_P=${TOP_P_VANILLA},GENERATION_CONFIG_BEAM_SIZE=${beam_size},GENERATION_CONFIG_DO_SAMPLE=False,EPSILON=${epsilon},SEED=${SEED},GENERATION_CONFIG_TYPICAL_P=${TYPICAL_P0},NAME=greedy,MAX_LENGTH=${MAX_LENGTH} setup_qwen_eval.slurm
                else
                    echo ${!var_name}
                    launch_shell_script \
                        --n_hyps=${i} \
                        --num_epochs=${NUM_EPOCHS} \
                        --wta_mode=${WTA_MODE} \
                        --dataset_name=${DATASET_NAME} \
                        --ckpt_path=${!var_name} \
                        --num_return_sequences=1 \
                        --seed=${SEED} \
                        --name=greedy \
                        --max_length=${MAX_LENGTH} \
                        --epsilon=${epsilon} \
                        --generation_config_beam_size=${beam_size} \
                        --generation_config_do_sample=False \
                        --generation_config_top_k=${TOP_K_VANILLA} \
                        --generation_config_top_p=${TOP_P_VANILLA} \
                        --generation_config_typical_p=${TYPICAL_P0}
                fi
            done
            
            if [ "${do_sampling}" = "True" ]; then
                # Sampling methods (top-k, nucleus, typical, ancestral)
                for method in "${!SAMPLING_CONFIGS[@]}"; do
                    config=(${SAMPLING_CONFIGS[$method]})
                    if [ "${use_slurm}" = "True" ]; then
                        sbatch --export=ALL,N_HYPS=${i},NUM_EPOCHS=${NUM_EPOCHS},DATASET_NAME=${DATASET_NAME},CKPT_PATH=${!var_name},NUM_RETURN_SEQUENCES=1,GENERATION_CONFIG_TOP_K=${config[0]},GENERATION_CONFIG_TOP_P=${config[1]},GENERATION_CONFIG_BEAM_SIZE=1,GENERATION_CONFIG_DO_SAMPLE=True,EPSILON=${epsilon},SEED=${SEED},GENERATION_CONFIG_TYPICAL_P=${config[2]},NAME=${method},MAX_LENGTH=${MAX_LENGTH} setup_qwen_eval.slurm
                    else
                        echo ${!var_name}
                        launch_shell_script \
                            --n_hyps=${i} \
                            --num_epochs=${NUM_EPOCHS} \
                            --wta_mode=${WTA_MODE} \
                            --dataset_name=${DATASET_NAME} \
                            --ckpt_path=${!var_name} \
                            --num_return_sequences=1 \
                            --seed=${SEED} \
                            --name=${method} \
                            --max_length=${MAX_LENGTH} \
                            --epsilon=${epsilon} \
                            --generation_config_beam_size=1 \
                            --generation_config_do_sample=True \
                            --generation_config_top_k=${config[0]} \
                            --generation_config_top_p=${config[1]} \
                            --generation_config_typical_p=${config[2]}
                    fi
                done
            fi
        done
    fi

    #===========================================================================
    # ANNEALED WTA EXPERIMENTS
    #===========================================================================
    if [ "${annealed_wta}" == "True" ]; then
        echo "ANNEALED WTA EXPERIMENTS"
        WTA_MODE=annealed-wta
        var_name="CKPT_${i}_HYP_ANNEALED_WTA_${DATASET_NAME^^}"
        echo "CKPT_PATH: ${!var_name}"
        # Beam Search with different beam sizes
        for beam_size in 1 2 ${i}; do
            if [ "${use_slurm}" = "True" ]; then
                sbatch --export=ALL,WTA_MODE=${WTA_MODE},N_HYPS=${i},NUM_EPOCHS=${NUM_EPOCHS},DATASET_NAME=${!DATASET_NAME},CKPT_PATH=${!var_name},NUM_RETURN_SEQUENCES=1,GENERATION_CONFIG_TOP_K=${TOP_K_VANILLA},GENERATION_CONFIG_TOP_P=${TOP_P_VANILLA},GENERATION_CONFIG_BEAM_SIZE=${beam_size},GENERATION_CONFIG_DO_SAMPLE=False,EPSILON=${epsilon},SEED=${SEED},GENERATION_CONFIG_TYPICAL_P=${TYPICAL_P0},NAME=bs_${beam_size},MAX_LENGTH=${MAX_LENGTH} qwen_eval.slurm
            else
                echo ${!var_name}
                launch_shell_script \
                    --n_hyps=${i} \
                    --num_epochs=${NUM_EPOCHS} \
                    --wta_mode=${WTA_MODE} \
                    --dataset_name=${DATASET_NAME} \
                    --ckpt_path=${!var_name} \
                    --num_return_sequences=1 \
                    --seed=${SEED} \
                    --name=bs_${beam_size} \
                    --max_length=${MAX_LENGTH} \
                    --generation_config_beam_size=${beam_size} \
                    --generation_config_do_sample=False \
                    --generation_config_top_k=${TOP_K_VANILLA} \
                    --generation_config_top_p=${TOP_P_VANILLA} \
                    --generation_config_typical_p=${TYPICAL_P0}
            fi
        done

        if [ "${do_sampling}" = "True" ]; then
            # Sampling methods (top-k, nucleus, typical, ancestral)
            for method in "${!SAMPLING_CONFIGS[@]}"; do
                config=(${SAMPLING_CONFIGS[$method]})
                sbatch --export=ALL,WTA_MODE=${WTA_MODE},N_HYPS=${i},NUM_EPOCHS=${NUM_EPOCHS},DATASET_NAME=${!DATASET_NAME},CKPT_PATH=${!var_name},NUM_RETURN_SEQUENCES=1,GENERATION_CONFIG_TOP_K=${config[0]},GENERATION_CONFIG_TOP_P=${config[1]},GENERATION_CONFIG_BEAM_SIZE=1,GENERATION_CONFIG_DO_SAMPLE=True,EPSILON=${epsilon},SEED=${SEED},GENERATION_CONFIG_TYPICAL_P=${config[2]},NAME=${method},MAX_LENGTH=${MAX_LENGTH} qwen_eval.slurm
            done
        fi
    fi

    #===========================================================================
    # MOE BASELINE EXPERIMENTS
    #===========================================================================
    if [ "${moe_baseline}" == "True" ]; then
        echo "MOE BASELINE EXPERIMENTS"
        WTA_MODE=moe
        var_name="CKPT_${i}_HYP_MOE_${DATASET_NAME^^}"
        # Beam Search with different beam sizes
        for beam_size in 1 2 ${i}; do  
            if [ "${use_slurm}" = "True" ]; then
                sbatch --export=ALL,WTA_MODE=${WTA_MODE},N_HYPS=${i},NUM_EPOCHS=${NUM_EPOCHS},DATASET_NAME=${!DATASET_NAME},CKPT_PATH=${!var_name},NUM_RETURN_SEQUENCES=1,GENERATION_CONFIG_TOP_K=${TOP_K_VANILLA},GENERATION_CONFIG_TOP_P=${TOP_P_VANILLA},GENERATION_CONFIG_BEAM_SIZE=${beam_size},GENERATION_CONFIG_DO_SAMPLE=False,EPSILON=${epsilon},SEED=${SEED},GENERATION_CONFIG_TYPICAL_P=${TYPICAL_P0},NAME=bs_${beam_size},MAX_LENGTH=${MAX_LENGTH} qwen_eval.slurm
            else
                echo ${!var_name}
                launch_shell_script \
                    --n_hyps=${i} \
                    --num_epochs=${NUM_EPOCHS} \
                    --wta_mode=${WTA_MODE} \
                    --dataset_name=${DATASET_NAME} \
                    --ckpt_path=${!var_name} \
                    --num_return_sequences=1 \
                    --seed=${SEED} \
                    --name=bs_${beam_size} \
                    --max_length=${MAX_LENGTH} \
                    --generation_config_beam_size=${beam_size} \
                    --generation_config_do_sample=False \
                    --generation_config_top_k=${TOP_K_VANILLA} \
                    --generation_config_top_p=${TOP_P_VANILLA} \
                    --generation_config_typical_p=${TYPICAL_P0}
            fi
        done

        if [ "${do_sampling}" = "True" ]; then
            # Sampling methods (top-k, nucleus, typical, ancestral)
            for method in "${!SAMPLING_CONFIGS[@]}"; do
                config=(${SAMPLING_CONFIGS[$method]})
                sbatch --export=ALL,WTA_MODE=${WTA_MODE},N_HYPS=${i},NUM_EPOCHS=${NUM_EPOCHS},DATASET_NAME=${!DATASET_NAME},CKPT_PATH=${!var_name},NUM_RETURN_SEQUENCES=1,GENERATION_CONFIG_TOP_K=${config[0]},GENERATION_CONFIG_TOP_P=${config[1]},GENERATION_CONFIG_BEAM_SIZE=1,GENERATION_CONFIG_DO_SAMPLE=True,EPSILON=${epsilon},SEED=${SEED},GENERATION_CONFIG_TYPICAL_P=${config[2]},NAME=${method},MAX_LENGTH=${MAX_LENGTH} qwen_eval.slurm
            done
        fi
    fi

done

#===============================================================================
# END OF SCRIPT
#===============================================================================
