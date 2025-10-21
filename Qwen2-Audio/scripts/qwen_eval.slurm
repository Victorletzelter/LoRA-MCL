#!/bin/bash

#SBATCH --job-name=sampling
#SBATCH --account=hgz@h100
#SBATCH -C h100
#SBATCH --output=$PROJECT_DIR/slurm_scripts/output_logs/qwen_eval%j.out
#SBATCH --error=$PROJECT_DIR/slurm_scripts/output_logs/qwen_eval%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread 
#SBATCH --qos=qos_gpu_h100-t3

module purge

conda init

conda activate $PROJECT_DIR/env

set -x

cd $PROJECT_DIR

n_hyps=$N_HYPS
num_train_epochs=$NUM_EPOCHS
training_wta_mode=$WTA_MODE
per_device_train_batch_size=$TRAIN_BATCH_SIZE
per_device_eval_batch_size=$EVAL_BATCH_SIZE
dataset_name=$DATASET_NAME
ckpt_path=$CKPT_PATH
num_return_sequences=$NUM_RETURN_SEQUENCES
seed=$SEED
name_run=$NAME

cross_dataset_eval=$CROSS_DATASET

wta_params_ini_temp=$WTA_INI_TEMP
wta_params_fin_temp=$WTA_FIN_TEMP
wta_params_decay_rate=$WTA_DECAY_RATE
wta_params_schedule_mode=$WTA_SCHEDULE_MODE

#Lora params
native_lora_enabled=True
native_lora_r=$RANK
native_lora_alpha=$RANK
native_lora_dropout=0.1
native_lora_bias="none"
native_group_lora_enabled=True
epsilon=$EPSILON
max_test_batches=Null
eval_strategy="epoch"
save_strategy="epoch"
learning_rate=1e-5
min_lr_rate=1e-6


gaussian_noise_std=$GAUSSIAN_NOISE_STD
tta_enabled=$TTA_ENABLED
tta_mode=$TTA_MODE
time_mask_percentage=$TIME_MASK_PERCENTAGE
freq_mask_percentage=$FREQ_MASK_PERCENTAGE

generation_config_temperature=$GENERATION_CONFIG_TEMPERATURE
generation_config_top_k=$GENERATION_CONFIG_TOP_K
generation_config_top_p=$GENERATION_CONFIG_TOP_P
generation_config_beam_size=$GENERATION_CONFIG_BEAM_SIZE
generation_config_do_sample=$GENERATION_CONFIG_DO_SAMPLE
generation_config_repetition_penalty=$GENERATION_CONFIG_REPETITION_PENALTY
generation_config_num_beam_groups=$GENERATION_CONFIG_NUM_BEAM_GROUPS
generation_config_diversity_penalty=$GENERATION_CONFIG_DIVERSITY_PENALTY
generation_config_penalty_alpha=$GENERATION_CONFIG_ALPHA
generation_config_typical_p=$GENERATION_CONFIG_TYPICAL_P
max_length=$MAX_LENGTH

greedy_sh_tta=$GREEDY_SH_TTA

if [ "$training_wta_mode" == "moe" ]; then
    native_group_lora_enabled=False
fi

if [ "$greedy_sh_tta" = "True" ]; then
    beam_size_show=1
else
    beam_size_show=$generation_config_beam_size
fi

if [ "$cross_dataset_eval" = "True" ]; then
    task_name=crossdata_eval_${dataset_name}_lora_${training_wta_mode}_${num_train_epochs}_epochs
else
    task_name=evalsampling_${num_train_epochs}_epochs_${dataset_name}_maxlength_${max_length}
fi

if [ "$tta_enabled" = "True" ]; then
    if [ "$tta_mode" = "spec_augment" ]; then
        out_dir=${name_run}_qwen_${training_wta_mode}-${num_train_epochs}-e-WTA-${n_hyps}-hyp_lora_${native_lora_r}_${native_lora_alpha}_${native_lora_dropout}_eps_${epsilon}_glora_${native_group_lora_enabled}_bs_${per_device_train_batch_size}_nr_${num_return_sequences}_temp_topk_p_bs_spl_rpen_dpen_bg_${generation_config_temperature}_${generation_config_top_k}_${generation_config_top_p}_${beam_size_show}_${generation_config_do_sample}_${generation_config_repetition_penalty}_${generation_config_diversity_penalty}_${generation_config_num_beam_groups}_alph_${generation_config_penalty_alpha}_tp_${generation_config_typical_p}_tta_${tta_enabled}_${tta_mode}_tfmask_${time_mask_percentage}_${freq_mask_percentage}_seed_${seed}
    elif [ "$tta_mode" = "gauss" ]; then
        out_dir=${name_run}_qwen_${training_wta_mode}-${num_train_epochs}-e-WTA-${n_hyps}-hyp_lora_${native_lora_r}_${native_lora_alpha}_${native_lora_dropout}_eps_${epsilon}_glora_${native_group_lora_enabled}_bs_${per_device_train_batch_size}_nr_${num_return_sequences}_temp_topk_p_bs_spl_rpen_dpen_bg_${generation_config_temperature}_${generation_config_top_k}_${generation_config_top_p}_${beam_size_show}_${generation_config_do_sample}_${generation_config_repetition_penalty}_${generation_config_diversity_penalty}_${generation_config_num_beam_groups}_alph_${generation_config_penalty_alpha}_tp_${generation_config_typical_p}_tta_${tta_enabled}_${tta_mode}_gauss_${gaussian_noise_std}_seed_${seed}
    fi
else
    out_dir=${name_run}_qwen_${training_wta_mode}-${num_train_epochs}-e-WTA-${n_hyps}-hyp_lora_${native_lora_r}_${native_lora_alpha}_${native_lora_dropout}_eps_${epsilon}_glora_${native_group_lora_enabled}_bs_${per_device_train_batch_size}_nr_${num_return_sequences}_temp_topk_p_bs_spl_rpen_dpen_bg_${generation_config_temperature}_${generation_config_top_k}_${generation_config_top_p}_${generation_config_beam_size}_${generation_config_do_sample}_${generation_config_repetition_penalty}_${generation_config_diversity_penalty}_${generation_config_num_beam_groups}_alph_${generation_config_penalty_alpha}_tp_${generation_config_typical_p}_seed_${seed}
fi

if [ "$training_wta_mode" == "moe" ]; then
    use_moe_lora=True
    training_wta_mode="wta"
    num_experts=$n_hyps
    n_hyps=1
fi

path_orig_weights_1_hyp="Qwen/Qwen2-Audio-7B-Instruct"
# path_orig_weights_1_hyp=$PROJECT_DIR/local_cache/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/0a095220c30b7b31434169c3086508ef3ea5bf0a

cache_dir=$PROJECT_DIR/local_cache

srun python main_hydra.py experiment=${dataset_name}_config run_name=${out_dir}_${dataset_name} model.pretrained_repo=${path_orig_weights_1_hyp} model.num_hyps=${n_hyps} task_name=${task_name} do_train=False do_inference=True do_compute_metrics=True model.generation_config.num_return_sequences=${num_return_sequences} model.wta_training_mode=${training_wta_mode} model.num_train_epochs=${num_train_epochs} out_dir=${out_dir}_${dataset_name} model.native_lora_enabled=${native_lora_enabled} model.native_lora_r=${native_lora_r} model.native_lora_alpha=${native_lora_alpha} model.native_lora_dropout=${native_lora_dropout} model.native_lora_bias=${native_lora_bias} model.wta_params_epsilon=${epsilon} model.native_group_lora_enabled=${native_group_lora_enabled} model.per_device_train_batch_size=${per_device_train_batch_size} model.per_device_eval_batch_size=${per_device_eval_batch_size} model.wta_params_ini_temp=${wta_params_ini_temp} model.wta_params_fin_temp=${wta_params_fin_temp} model.wta_params_decay_rate=${wta_params_decay_rate} model.wta_params_schedule_mode=${wta_params_schedule_mode} ckpt_path=${ckpt_path} model.generation_config.temperature=${generation_config_temperature} model.generation_config.top_k=${generation_config_top_k} model.generation_config.beam_size=${generation_config_beam_size} model.generation_config.do_sample=${generation_config_do_sample} model.generation_config.repetition_penalty=${generation_config_repetition_penalty} model.generation_config.num_beam_groups=${generation_config_num_beam_groups} model.generation_config.diversity_penalty=${generation_config_diversity_penalty} model.generation_config.penalty_alpha=${generation_config_penalty_alpha} model.generation_config.top_p=${generation_config_top_p} model.generation_config.typical_p=${generation_config_typical_p} model.gaussian_noise_std=${gaussian_noise_std} model.tta_enabled=${tta_enabled} model.tta_mode=${tta_mode} model.time_mask_percentage=${time_mask_percentage} model.freq_mask_percentage=${freq_mask_percentage} model.greedy_sh_tta=${greedy_sh_tta} model.max_length=${max_length} model.use_moe_lora=${use_moe_lora} model.num_experts=${num_experts} model.cache_dir=${cache_dir}
