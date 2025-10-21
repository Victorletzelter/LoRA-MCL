#!/usr/bin/env python
# This file is adapted from the ALMA repository https://github.com/fe1ixxu/ALMA/blob/master/run_llmmt.py
# Under MIT License.

import json
import shutil
import traceback

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
os.environ['HF_HOME'] = f"{os.environ['PROJECT_ROOT']}/huggingface"
os.environ['TRANSFORMERS_CACHE'] = f"{os.environ['PROJECT_ROOT']}/huggingface"
os.environ['HF_DATASETS_CACHE'] = f"{os.environ['PROJECT_ROOT']}/huggingface/datasets"

import pickle
import logging
import copy
import math
import os
import sys
import json
import random
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import numpy as np

import datasets
import evaluate
import torch
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    default_data_collator,
    # is_torch_tpu_available,
    set_seed,
    LlamaTokenizer,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel, PeftConfig
from collections import defaultdict
from transformers.trainer_callback import TrainerCallback
from datasets import concatenate_datasets, interleave_datasets
from utils.trainer_llmmt import LlmmtTrainer
from utils.utils import LANG_TABLE, load_mmt_dataset, get_preprocessed_data, clean_outputstring, load_a_single_text_file, load_wmt_multi_ref_file, load_tokenizer, load_model, SavePeftModelCallback, get_key_suffix, NLLB_CODE, ISO1_ISO3_map
from utils.arguments import ModelArguments, DataTrainingArguments
from utils.ul2collator import DataCollatorForUL2

logger = logging.getLogger(__name__)

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

# Allow DeepSpeed pickled classes under torch.load(weights_only=True) in PyTorch 2.6
try:
    from torch.serialization import add_safe_globals
    from deepspeed.runtime.fp16.loss_scaler import DynamicLossScaler
    add_safe_globals([DynamicLossScaler])
except Exception:
    pass

class HypothesisTrackingCallback(TrainerCallback):
    def __init__(self, log_frequency=100):
        self.log_frequency = log_frequency
    
    def on_step_end(self, args, state, control, model=None, logs=None, **kwargs):
        if state.global_step % self.log_frequency == 0:
            if hasattr(model, 'get_hypothesis_stats'):
                stats = model.get_hypothesis_stats()
                if stats:
                    print(f"\n=== Hypothesis Usage Stats (Step {state.global_step}) ===")
                    for i in range(len(stats['hypothesis_usage_count'])):
                        count = stats['hypothesis_usage_count'][i]
                        pct = stats['hypothesis_usage_percentage'][i]
                        print(f"Hypothesis {i}: {count} times ({pct:.1f}%)")
                    print(f"Most used: Hypothesis {stats['most_used_hypothesis']}")
                    print(f"Least used: Hypothesis {stats['least_used_hypothesis']}")
                    print("=" * 50)

def load_test_labels(test_raw_data, pairs):
    """Load ground truth labels for test datasets"""
    test_labels = {}
    test_labels_src = {}
    
    for lg_pair in pairs:
        if lg_pair in test_raw_data["mmt"]:
            labels = []
            labels_src = []
            src_lang, tgt_lang = lg_pair.split("-")
            
            raw_test = test_raw_data["mmt"][lg_pair]["test"]
            for example in raw_test:
                # Prefer multi-refs when present
                if 'mrefs' in example[lg_pair] and isinstance(example[lg_pair]['mrefs'], list):
                    labels.append(example[lg_pair]['mrefs'])  # list of refs
                    # source from translation if available, else empty
                    # if example[lg_pair][tgt_lang] not in example[lg_pair]['mrefs']:
                        # labels[-1].append(example[lg_pair][tgt_lang])
                    labels_src.append(example[lg_pair][src_lang])
                    # if 'translation' in example and src_lang in example['translation']:
                        # labels_src.append(example['translation'][src_lang])
                    # else:
                        # labels_src.append("")
                elif 'translation' in example:
                    tgt_val = example['translation'][tgt_lang]
                    # normalize to list-of-strings
                    if isinstance(tgt_val, list):
                        labels.append(tgt_val)
                    else:
                        labels.append([tgt_val])
                    labels_src.append(example['translation'][src_lang])
                else:
                    # legacy dict format: example[lg_pair][tgt_lang]
                    tgt_val = example[lg_pair][tgt_lang]
                    if isinstance(tgt_val, list):
                        labels.append(tgt_val)
                    else:
                        labels.append([tgt_val])
                    labels_src.append(example[lg_pair][src_lang])
            
            test_labels[lg_pair] = labels
            # ensure sources are lists too for downstream printing [0]
            test_labels_src[lg_pair] = [[s] if not isinstance(s, list) else s for s in labels_src]
    
    return test_labels, test_labels_src

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_llmmt", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Get the datasets
    pairs = data_args.language_pairs.split(",")
    train_raw_data, valid_raw_data, test_raw_data = {}, None, {}
    if data_args.text_test_file:
        if data_args.text_test_file.endswith('.extra_refs.tok') or data_args.text_test_file.endswith('.extra_refs'):
            test_raw_data["mmt"] = load_wmt_multi_ref_file(pairs, data_args, model_args)
        else:
            test_raw_data["mmt"] = load_a_single_text_file(pairs, data_args, model_args)
    elif data_args.mmt_data_path:
        mmt_train_raw_data, valid_raw_data, mmt_test_raw_data = load_mmt_dataset(pairs, data_args, model_args, training_args, logger)
        train_raw_data["mmt"] = mmt_train_raw_data
        test_raw_data["mmt"] = mmt_test_raw_data

    load_kwargs = {
            'cache_dir': model_args.cache_dir,
            'token': True if model_args.use_auth_token else None,
            'streaming': data_args.streaming,
            "trust_remote_code": True,
        }

    if data_args.aya_datasets:
        considered_languages_ISO1 = sorted(set(lang for p in pairs for lang in p.split('-')))
        considered_languages_ISO3 = [ISO1_ISO3_map[lang] for lang in considered_languages_ISO1]
        if training_args.do_train:
            aya_train_raw_data = load_dataset(
                    data_args.aya_datasets,
                    **load_kwargs,
                )['train']
                
            train_raw_data["aya"] = aya_train_raw_data.filter(lambda x: x['language_code'] in considered_languages_ISO3)
        if training_args.do_predict:
            test_raw_data["aya"] = {}
            aya_test_raw_data = load_dataset(
                data_args.aya_datasets,
                **load_kwargs,
            )['test']
            for lg in considered_languages_ISO1:
                sub_aya_test_raw_data = aya_test_raw_data.filter(lambda x: x['language_code'] in [ISO1_ISO3_map[lg]])
                if len(sub_aya_test_raw_data) > 0:
                    test_raw_data["aya"][lg] = sub_aya_test_raw_data

    if data_args.mono_data_path:
        train_raw_data["mono"] = load_dataset(
            "json",
            data_files=data_args.mono_data_path,
            **load_kwargs,
        )

    if data_args.nllb_pretrain_data_path:
        if data_args.nllb_interleave_probs:
            interleave_probs = [float(p) for p in data_args.nllb_interleave_probs.split(",")]
        else:
            interleave_probs = [1/len(pairs)] * len(pairs)
            
        nllb_raw_data = []
        for lg_pair in pairs:
            src_lang, tgt_lang = lg_pair.split("-")
            src_lang, tgt_lang = NLLB_CODE[src_lang], NLLB_CODE[tgt_lang]
            language_key = f"{src_lang}-{tgt_lang}" if src_lang < tgt_lang else f"{tgt_lang}-{src_lang}"

            if src_lang == "tha_Thai" or tgt_lang == "tha_Thai":
                lg_dataset = load_dataset(
                    "Helsinki-NLP/opus-100",
                    "en-th",
                    **load_kwargs,
                )["train"].shuffle(seed=training_args.seed)
            else:
                lg_dataset = load_dataset(
                    data_args.nllb_pretrain_data_path,
                    language_key,
                    **load_kwargs,
                )['train'].shuffle(seed=training_args.seed)

            def normalize_example(example):
                lg1, lg2 = example["translation"].keys()
                if random.random() < 0.5:
                    combined_translation = example["translation"][lg1] + " " + example["translation"][lg2]
                else:
                    combined_translation = example["translation"][lg2] + " " + example["translation"][lg1]
                return {
                    "raw_text": combined_translation,
                }

            lg_dataset = lg_dataset.map(normalize_example, remove_columns=lg_dataset.column_names)
            nllb_raw_data.append(lg_dataset)
            
        train_raw_data["nllb_pretrain"] = interleave_datasets(nllb_raw_data, probabilities=interleave_probs, seed=training_args.seed, stopping_strategy="all_exhausted")
        
    if data_args.oscar_data_path:
        oscar_langs = data_args.oscar_data_lang.split(",")
        if data_args.interleave_probs:
            interleave_probs = [float(p) for p in data_args.interleave_probs.split(",")]
        else:
            interleave_probs = [1/len(oscar_langs)] * len(oscar_langs)
        oscar_langs = [x for x, _ in sorted(zip(oscar_langs, interleave_probs), key=lambda zippair: zippair[1])]
        interleave_probs = sorted(interleave_probs)
        oscar_train_raw_data = []

        for lg in oscar_langs:
            oscar_lg_data = load_dataset(
                data_args.oscar_data_path,
                lg,
                **load_kwargs
            )['train'].shuffle(seed=training_args.seed)

            def normalize_oscar_example(example):
                return {
                    "raw_text": example["text"],
                }
            oscar_lg_data = oscar_lg_data.map(normalize_oscar_example, remove_columns=oscar_lg_data.column_names)
            oscar_train_raw_data.append(oscar_lg_data)
        train_raw_data["oscar"] = interleave_datasets(oscar_train_raw_data, probabilities=interleave_probs, seed=training_args.seed, stopping_strategy="all_exhausted")

        if "nllb_pretrain" in train_raw_data:
        ## only for nllb pretrain and oscar:
            train_raw_data["oscar"] = interleave_datasets([train_raw_data["oscar"], train_raw_data["nllb_pretrain"]], probabilities=[0.5, 0.5], seed=training_args.seed, stopping_strategy="all_exhausted").shuffle(seed=training_args.seed)
            train_raw_data.pop("nllb_pretrain")
        
    # load tokenizer
    set_seed(training_args.seed)
    tokenizer = load_tokenizer(data_args, model_args, training_args, logger)
    if data_args.use_ul2:
        assert data_args.use_prefix_lm, "Must enable use prefix language model"

    shots_eval_dict = {}
    if data_args.few_shot_eval_path:
        for lg_pair in test_raw_data["mmt"].keys():
            pair_shot_path = os.path.join(data_args.few_shot_eval_path, f"shots.{lg_pair}.json")
            if not os.path.isfile(pair_shot_path):
                ValueError(f"Make sure the language pair {lg_pair} is in the few shot eval folder!")
            with open(pair_shot_path) as f:
                shots_eval_dict[lg_pair] = json.load(f)

    if model_args.chat_style:
        dummy_sentence = "This is a dummy sentence"
        chat_dummy_sentence = [{"role": "user", "content": dummy_sentence}] 
        dummy_sentence_with_speical_tokens = tokenizer.apply_chat_template(chat_dummy_sentence, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer.encode(dummy_sentence_with_speical_tokens, add_special_tokens=False)
        decoded_text = tokenizer.decode(encoded, skip_special_tokens=True)
        begin_prefix = decoded_text.split(dummy_sentence, 1)[0].strip()
        additional_suffix = decoded_text.split(dummy_sentence, 1)[-1]
    else:
        begin_prefix = ""
        additional_suffix = ""

    if data_args.max_test_samples is not None:
        if data_args.max_test_samples == -1:
            max_test_samples_saved = data_args.max_test_samples
            data_args.max_test_samples = None
        else:
            max_test_samples_saved = data_args.max_test_samples
    else:
        max_test_samples_saved = -1

    train_datasets, eval_datasets, test_datasets = get_preprocessed_data(train_raw_data, valid_raw_data, test_raw_data, pairs, tokenizer, shots_eval_dict, data_args, training_args, model_args)
    metric = evaluate.load("sacrebleu")

    # Load model
    model = load_model(data_args, model_args, training_args, tokenizer, logger)
    collate_fn = DataCollatorForUL2(model, tokenizer) if data_args.use_ul2 else default_data_collator
    
    # Initialize our Trainer
    trainer = LlmmtTrainer(
        model=model,
        args=training_args,
        train_dataset=train_datasets if training_args.do_train else None,
        eval_dataset=eval_datasets if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        callbacks=[SavePeftModelCallback, HypothesisTrackingCallback] if model_args.use_peft else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint

        try:
            train_result = trainer.train(resume_from_checkpoint=checkpoint)

            trainer.save_state()
            if model_args.use_peft:
                model.save_pretrained(training_args.output_dir) 
            else:
                trainer.save_model()  # Saves the tokenizer too for easy upload

        except Exception as e:
            logger.error(f"Error duringtraining: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")

        # Rename best checkpoint to best_ckpt for consistent evaluation
        ### List all the folders with name "checkpoint-*" and get the latest one
        checkpoint_folders = [f for f in os.listdir(training_args.output_dir) if f.startswith("checkpoint-")]
        if len(checkpoint_folders) > 0:
            latest_checkpoint_folder = max(checkpoint_folders, key=lambda x: int(x.split("-")[-1]))
            latest_checkpoint_path = os.path.join(training_args.output_dir, latest_checkpoint_folder)
            trainer_state_path = os.path.join(latest_checkpoint_path, "trainer_state.json")
            import json
            if os.path.exists(trainer_state_path):
                with open(trainer_state_path, 'r') as f:
                    trainer_state = json.load(f)

                best_checkpoint_path = trainer_state.get("best_model_checkpoint") if trainer_state.get("best_model_checkpoint") else latest_checkpoint_path
                if best_checkpoint_path and os.path.exists(best_checkpoint_path):
                    best_ckpt_path = os.path.join(training_args.output_dir, "best")
                    
                    # Remove existing best_ckpt if it exists
                    if os.path.exists(best_ckpt_path):
                        shutil.rmtree(best_ckpt_path)
                    
                    # Copy best checkpoint to best_ckpt
                    shutil.copytree(best_checkpoint_path, best_ckpt_path)
                    logger.info(f"Best checkpoint copied from {best_checkpoint_path} to {best_ckpt_path}")
                    
                    # Update trainer_state.json to point to new location
                    trainer_state["best_model_checkpoint"] = best_ckpt_path
                    with open(trainer_state_path, 'w') as f:
                        json.dump(trainer_state, f, indent=2)
                else:
                    logger.warning("No best checkpoint found to rename")

    dataset_name = None
    if data_args.override_test_data_path is not None:
        dataset_name = data_args.override_test_data_path
    elif data_args.text_test_file is not None:
        dataset_name = data_args.text_test_file
    elif data_args.mmt_data_path is not None:
        dataset_name = data_args.mmt_data_path

    dataset_name = dataset_name.split(os.environ['PROJECT_ROOT'])[-1]
    dataset_name = dataset_name.replace("/", "")

    # Prediction
    if training_args.do_predict is True and model_args.do_predict_all_beams is True:

        if model_args.num_hyps == 1:
            beam_size_list = [model_args.num_return_sequences, 2*model_args.num_return_sequences, model_args.num_return_sequences*model_args.num_return_sequences]
            diversity_penalty_list = [0.8, 0.0]
        elif model_args.num_hyps == 2:
            beam_size_list = [1, 2]
            diversity_penalty_list = [0.0]
        else :
            diversity_penalty_list = [0.0]
            beam_size_list = [1, 2, model_args.num_hyps]

        for beam_size in beam_size_list:
            for diversity_penalty in diversity_penalty_list:
                logger.info(f"*** Prediction for beam size {beam_size}***")
                data_args.num_beams = beam_size
                trainer.args.prediction_loss_only = False
                # if data_args.mmt_data_path:
                if "mmt" in test_datasets.keys():
                    lg_pairs = sorted(test_datasets["mmt"].keys()) # make sure each device print in the same order
                    for lg_pair in lg_pairs:
                        test_dataset = test_datasets["mmt"][lg_pair]
                        src_lang, tgt_lang = lg_pair.split("-")
                        logger.info(f"*** Prediction for {lg_pair}***")
                        if model_args.encoder_decoder_type == "nllb":
                            preds, _, _ = trainer.predict(
                            test_dataset=test_dataset, 
                            max_new_tokens=data_args.max_new_tokens, 
                            num_beams=data_args.num_beams, 
                            metric_key_prefix="test",
                            use_cache=True,
                            forced_bos_token_id=tokenizer.lang_code_to_id[NLLB_CODE[tgt_lang]],
                            num_return_sequences=model_args.num_return_sequences if model_args.num_hyps == 1 else 1,
                            diversity_penalty=diversity_penalty,
                            num_beam_groups=model_args.num_return_sequences if (model_args.num_hyps == 1 and diversity_penalty > 0.) else 1,
                            do_sample=False,
                        )
                        else:
                            preds, _, _ = trainer.predict(
                                test_dataset=test_dataset, 
                                max_new_tokens=data_args.max_new_tokens, 
                                num_beams=data_args.num_beams, 
                                metric_key_prefix="test",
                                use_cache=True,
                                num_return_sequences=model_args.num_return_sequences if model_args.num_hyps == 1 else 1,
                                diversity_penalty=diversity_penalty,
                                num_beam_groups=model_args.num_return_sequences if (model_args.num_hyps == 1 and diversity_penalty > 0.) else 1,
                                do_sample=False,
                            )

                        if preds.ndim == 3:
                            test_labels, test_labels_src = load_test_labels(test_raw_data, pairs)
                            lg_pair = list(test_labels.keys())[0]
                            decoded_preds_dict = {}
                            decoded_labels = test_labels[lg_pair]
                            decoded_labels_src = test_labels_src[lg_pair]
                            if preds.shape[1] != len(decoded_labels):
                                logger.info(f"The number of predictions {preds.shape[0]} is not equal to the number of labels {len(decoded_labels)}")
                                logger.info(f"Truncated the labels to {preds.shape[0]}")
                                decoded_labels = decoded_labels[:preds.shape[0]]
                                decoded_labels_src = decoded_labels_src[:preds.shape[0]]
                            for hyp_idx in range(preds.shape[1]):
                                decoded_preds_dict[f'hyp{hyp_idx}'] = []
                                if int(torch.cuda.current_device()) == 0:
                                    preds_hyp = np.where(preds[:,hyp_idx,:] != -100, preds[:,hyp_idx,:], tokenizer.pad_token_id)

                                    decoded_preds = tokenizer.batch_decode(preds_hyp, skip_special_tokens=True)

                                    # Some simple post-processing
                                    decoded_preds = [pred.strip() for pred in decoded_preds]

                                    for idx in range(min(data_args.display_num_translations, len(decoded_preds))):
                                        print("------------------------")
                                        print(decoded_preds[idx])

                                    output_dir = os.path.join(training_args.output_dir, f"{dataset_name}-{max_test_samples_saved}samples-NR{model_args.num_return_sequences}")
                                    if not os.path.exists(output_dir):
                                        os.makedirs(output_dir)

                                    with open(os.path.join(output_dir, f"{dataset_name}-{max_test_samples_saved}samples-{src_lang}-{tgt_lang}{data_args.suffix_eval_file}-hyp{hyp_idx}-B{beam_size}-Div{diversity_penalty}-NR{model_args.num_return_sequences}"), "w", encoding="utf-8") as f:
                                        suffix = get_key_suffix(tgt_lang, data_args, additional_suffix)
                                        if len(shots_eval_dict) != 0:
                                            split_idx = len(shots_eval_dict[lg_pair]) + 1
                                        else:
                                            split_idx = 1
                                        for pred in decoded_preds:
                                            # Output is itself if it is an encoder-decoder model, otherwise it is the prefix + output
                                            pred = clean_outputstring(pred, suffix, logger, split_idx) if not model_args.encoder_decoder_type else pred.strip()
                                            f.writelines([pred, "\n"])
                                            decoded_preds_dict[f'hyp{hyp_idx}'].append(pred)
                        
                            output_dict, pickle_file_path = create_hypotheses_pickle_file(decoded_preds_dict, decoded_labels, decoded_labels_src=decoded_labels_src, subset='test', dataset_name=dataset_name, output_dir=training_args.output_dir, beam_size=beam_size, diversity_penalty=diversity_penalty, max_test_samples=max_test_samples_saved, num_return_sequences=model_args.num_return_sequences)
                        
                            if model_args.do_compute_metrics:
                                num_hypotheses = model_args.num_return_sequences if model_args.num_hyps == 1 else model_args.num_hyps
                                try:
                                    os.system(f"python {os.environ['PROJECT_ROOT']}/compute_metrics.py --path {pickle_file_path} --num_hypotheses {num_hypotheses}")
                                except Exception as e:
                                    print(f"Warning: Could not compute metrics: {e}")

                            output_dir = os.path.join(training_args.output_dir, f"{dataset_name}-{max_test_samples_saved}samples-NR{model_args.num_return_sequences}")
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)

                            ### Write a txt file containing the target / hypotheses prediction pairs
                            with open(os.path.join(output_dir, f"{dataset_name}-{max_test_samples_saved}samples-{src_lang}-{tgt_lang}{data_args.suffix_eval_file}-hyp-B{beam_size}-Div{diversity_penalty}-NR{model_args.num_return_sequences}.txt"), "w") as f:
                                if decoded_preds_dict:  # Check if dict is not empty
                                    # Get number of examples from first hypothesis
                                    first_hyp_key = next(iter(decoded_preds_dict.keys()))
                                    n_examples = len(decoded_preds_dict[first_hyp_key])
                                    
                                    for example_idx in range(n_examples):
                                        f.writelines([decoded_labels_src[example_idx][0], "\n"])
                                        for label in decoded_labels[example_idx]:
                                            f.writelines([label, "\n"])
                                        for hyp_key in decoded_preds_dict.keys():  # Use actual keys
                                            f.writelines([f"{hyp_key}: "])
                                            f.writelines([decoded_preds_dict[hyp_key][example_idx], "\n"])
                                        f.writelines(["\n"])

                        else:
                            # Replace -100s used for padding as we can't decode them
                            if int(torch.cuda.current_device()) == 0:
                                preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

                                decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

                                # Some simple post-processing
                                decoded_preds = [pred.strip() for pred in decoded_preds]

                                for idx in range(data_args.display_num_translations):
                                    print("------------------------")
                                    print(decoded_preds[idx])

                                output_dir = os.path.join(training_args.output_dir, f"{dataset_name}-{max_test_samples_saved}samples-NR{model_args.num_return_sequences}")
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)

                                with open(os.path.join(output_dir, f"{dataset_name}-{max_test_samples_saved}samples-{src_lang}-{tgt_lang}{data_args.suffix_eval_file}-B{beam_size}"), "w", encoding="utf-8") as f:
                                    suffix = get_key_suffix(tgt_lang, data_args, additional_suffix)
                                    if len(shots_eval_dict) != 0:
                                        split_idx = len(shots_eval_dict[lg_pair]) + 1
                                    else:
                                        split_idx = 1
                                    for pred in decoded_preds:
                                        # Output is itself if it is an encoder-decoder model, otherwise it is the prefix + output
                                        pred = clean_outputstring(pred, suffix, logger, split_idx) if not model_args.encoder_decoder_type else pred.strip()
                                        f.writelines([pred, "\n"])

        output_dir = os.path.join(training_args.output_dir, f"{dataset_name}-{max_test_samples_saved}samples-NR{model_args.num_return_sequences}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        import yaml
        import json
        from dataclasses import asdict

        ### Log the full config used for this run in a yaml file
        try:
            # Convert dataclasses to dictionaries and filter out non-serializable items
            def make_serializable(obj):
                if hasattr(obj, '__dict__'):
                    result = {}
                    for key, value in obj.__dict__.items():
                        try:
                            # Test if the value can be serialized
                            json.dumps(value, default=str)
                            result[key] = value
                        except (TypeError, ValueError):
                            # Convert non-serializable objects to string representation
                            result[key] = str(value)
                    return result
                return str(obj)

            training_args_dict = make_serializable(training_args)
            model_args_dict = make_serializable(model_args)
            data_args_dict = make_serializable(data_args)

            with open(os.path.join(training_args.output_dir, "training_args.yaml"), "w") as f:
                yaml.dump(training_args_dict, f, default_flow_style=False)
            with open(os.path.join(training_args.output_dir, "model_args.yaml"), "w") as f:
                yaml.dump(model_args_dict, f, default_flow_style=False)
            with open(os.path.join(training_args.output_dir, "data_args.yaml"), "w") as f:
                yaml.dump(data_args_dict, f, default_flow_style=False)
                
        except Exception as e:
            print(f"Warning: Could not save config as YAML: {e}")
            # Fallback: save as JSON instead
            try:
                with open(os.path.join(training_args.output_dir, "training_args.json"), "w") as f:
                    json.dump(make_serializable(training_args), f, indent=2)
                with open(os.path.join(training_args.output_dir, "model_args.json"), "w") as f:
                    json.dump(make_serializable(model_args), f, indent=2)
                with open(os.path.join(training_args.output_dir, "data_args.json"), "w") as f:
                    json.dump(make_serializable(data_args), f, indent=2)
            except Exception as e2:
                print(f"Warning: Could not save config as JSON either: {e2}")
                
def create_hypotheses_pickle_file(decoded_preds, decoded_labels, decoded_labels_src=None, subset='test', dataset_name='mmt', output_dir='', beam_size=1, diversity_penalty=0.0, max_test_samples=-1, num_return_sequences=1):
    output_dict = {}
    num_hyps = len(decoded_preds)
    for hyp_idx in range(num_hyps):
        output_dict[f'hypothesis_{hyp_idx}'] = {
            'cands': [],
            'mrefs': [],
            'filenames': [],
            'fname': [],
            'subset': [],
            'index': [],
            'dataset': [],
            'preds': [],
            'losses': [],
            'src': [],
        }
        output_dict[f'hypothesis_{hyp_idx}']['src'] = [e[0] for e in decoded_labels_src]
        output_dict[f'hypothesis_{hyp_idx}']['cands'] = decoded_preds[f'hyp{hyp_idx}']
        output_dict[f'hypothesis_{hyp_idx}']['mrefs'] = decoded_labels
        output_dict[f'hypothesis_{hyp_idx}']['filenames'] = [f'example_{i}' for i in range(len(decoded_preds[f'hyp{hyp_idx}']))]
        output_dict[f'hypothesis_{hyp_idx}']['fname'] = [f'example_{i}' for i in range(len(decoded_preds[f'hyp{hyp_idx}']))]
        output_dict[f'hypothesis_{hyp_idx}']['subset'] = [subset]*len(decoded_preds[f'hyp{hyp_idx}'])
        output_dict[f'hypothesis_{hyp_idx}']['index'] = [0]*len(decoded_preds[f'hyp{hyp_idx}'])
        output_dict[f'hypothesis_{hyp_idx}']['dataset'] = [dataset_name]*len(decoded_preds[f'hyp{hyp_idx}'])
        output_dict[f'hypothesis_{hyp_idx}']['preds'] = [None]*len(decoded_preds[f'hyp{hyp_idx}'])
        output_dict[f'hypothesis_{hyp_idx}']['losses'] = [0]*len(decoded_preds[f'hyp{hyp_idx}'])

    ### Create a pickle file
    output_dir = os.path.join(output_dir, f"{dataset_name}-{max_test_samples}samples-NR{num_return_sequences}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    path_pickle = f"{dataset_name}-{max_test_samples}samples-B{beam_size}_Div{diversity_penalty}_NR{num_return_sequences}_{subset}.pkl"
    with open(os.path.join(output_dir, path_pickle), "wb") as f:
        pickle.dump(output_dict, f)

    return output_dict, os.path.join(output_dir, path_pickle)

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

