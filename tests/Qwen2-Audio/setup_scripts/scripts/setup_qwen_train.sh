#!/bin/bash

DIR_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR=$DIR_SCRIPT/..

set -x 

n_hyps=$NUM_HYPS
num_train_epochs=$NUM_EPOCHS
training_wta_mode=$WTA_MODE
per_device_train_batch_size=$TRAIN_BATCH_SIZE
per_device_eval_batch_size=$EVAL_BATCH_SIZE
dataset_name=$DATASET_NAME
epsilon=$EPSILON
seed=$SEED
max_length=$MAX_LENGTH
rank=$RANK
gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS

#Lora params
native_lora_enabled=True
native_lora_r=$rank
native_lora_alpha=$rank
native_lora_dropout=0.1
native_lora_bias="none"
native_group_lora_enabled=True
max_test_batches=Null
eval_strategy="epoch"
save_strategy="epoch"
learning_rate=1e-5
min_lr_rate=1e-6


out_dir=${training_wta_mode}_epsilon-${epsilon}_${n_hyps}-hyp_rank-${rank}

path_orig_weights_1_hyp=$PROJECT_DIR/local_cache/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/0a095220c30b7b31434169c3086508ef3ea5bf0a

cache_dir=$PROJECT_DIR/local_cache

######### TO DELETE when the other part is deleted
PROJECT_DIR=$DIR_SCRIPT/../../../../Qwen2-Audio
#########

echo $PROJECT_DIR

cd $PROJECT_DIR

if [ "$training_wta_mode" == "moe" ]; then
    use_moe_lora=True
    training_wta_mode="wta"
    num_experts=$n_hyps
    n_hyps=1
fi

python main_hydra.py experiment=${dataset_name}_config run_name=${out_dir}_${dataset_name} model.pretrained_repo=${path_orig_weights_1_hyp} model.num_hyps=${n_hyps} task_name=setup_${dataset_name} do_train=True do_inference=True do_compute_metrics=True model.generation_config.num_return_sequences=1 model.wta_training_mode=${training_wta_mode} model.num_train_epochs=${num_train_epochs} out_dir=${out_dir}_${dataset_name} model.native_lora_enabled=${native_lora_enabled} model.native_lora_r=${native_lora_r} model.native_lora_alpha=${native_lora_alpha} model.native_lora_dropout=${native_lora_dropout} model.native_lora_bias=${native_lora_bias} model.wta_params_epsilon=${epsilon} model.native_group_lora_enabled=${native_group_lora_enabled} model.per_device_train_batch_size=${per_device_train_batch_size} model.per_device_eval_batch_size=${per_device_eval_batch_size} seed=${seed} model.max_length=${max_length} model.gradient_accumulation_steps=${gradient_accumulation_steps} model.max_steps=2 model.eval_strategy=steps model.save_strategy=steps model.eval_steps=2 max_train_batches=2 max_val_batches=2 max_test_batches=2 model.use_moe_lora=${use_moe_lora} model.num_experts=${num_experts} model.cache_dir=${cache_dir}