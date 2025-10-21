#!/bin/bash
# Shell script to train LoRA-MCL.

OUTPUT_DIR=${1:-"./alma-7b-parallel-ft-lora"}
pairs=${2:-"de-en,cs-en,is-en,zh-en,ru-en,en-de,en-cs,en-is,en-zh,en-ru"}
LORA_RANK=${3:-"16"}
NUM_SAMPLES=${4:-"16"}
NUM_HYPS=${5:-"2"}
EPSILON=${6:-"0.05"}
MAX_TRAIN_STEPS=${7:-"-1"}
# random port between 30000 and 50000
port=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$PROJECT_DIR/.."


# number of GPUs to use (defaults to all visible); override with NGPUS env var if needed
NGPUS=${NGPUS:-$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)}
echo "Using ${NGPUS} GPUs"

accelerate launch --main_process_port ${port} --num_processes ${NGPUS} --config_file $PROJECT_DIR/configs/deepspeed_train_config_bf16_mh.yaml \
     $PROJECT_DIR/run_llmmt.py \
    --model_name_or_path haoranxu/ALMA-7B-Pretrain \
    --mmt_data_path  $PROJECT_DIR/human_written_data/ \
    --use_peft \
    --lora_rank ${LORA_RANK} \
    --do_train True \
    --do_eval \
    --do_predict \
    --language_pairs ${pairs} \
    --load_best_model_at_end \
    --low_cpu_mem_usage \
    --bf16 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 64 \
    --lr_scheduler_type inverse_sqrt \
    --warmup_ratio 0.01 \
    --ignore_pad_token_for_loss \
    --ignore_prompt_token_for_loss \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --evaluation_strategy steps \
    --eval_steps 0.10 \
    --save_strategy steps \
    --save_steps 0.10 \
    --save_total_limit 1 \
    --logging_strategy steps \
    --logging_steps 2 \
    --logging_first_step True \
    --logging_nan_inf_filter false \
    --log_on_each_node false \
    --logging_dir ${OUTPUT_DIR}/tensorboard \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 2 \
    --predict_with_generate \
    --prediction_loss_only \
    --max_new_tokens 256 \
    --max_source_length 256 \
    --seed 42 \
    --overwrite_output_dir \
    --num_beams 5 \
    --ddp_timeout 999999 \
    --report_to tensorboard \
    --overwrite_cache \
    --is_mcl True \
    --is_group_lora False \
    --num_hyps ${NUM_HYPS} \
    --wta_training_mode relaxed-wta \
    --wta_params_epsilon ${EPSILON} \
    --wta_params_ini_temp 1.0 \
    --wta_params_fin_temp 1e-6 \
    --wta_params_decay_rate 0.999 \
    --wta_params_schedule_mode global_step \
    --do_compute_metrics True \
    --do_predict_all_beams True \
    --max_test_samples ${NUM_SAMPLES} \
    --max_steps ${MAX_TRAIN_STEPS} \
    --num_return_sequences 1 \