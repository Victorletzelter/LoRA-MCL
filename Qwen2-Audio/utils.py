# import sys
import time
import logging
import torch
import numpy as np
import pickle
import os.path as osp
import re
import string
import os

if os.environ.get("USE_LOCAL_TRANSFORMERS", "false").lower() == "true":
    from local_transformers import (
        AutoConfig,
        Trainer,
        BitsAndBytesConfig,
    )
    from local_transformers.configuration_utils import PretrainedConfig
else:
    from transformers import (
        AutoConfig,
        Trainer,
        BitsAndBytesConfig,
    )
    from transformers.configuration_utils import PretrainedConfig

# Make sure to have your project root set up (if needed)
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Configure the logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def save_predictions_to_pkl(outputs: dict, datasubset: str, subrun_dir: str, generation_mode: str) -> None:
    """Save predictions to a file."""
    if generation_mode is None:
        generation_mode = ""

    save_path = osp.join(subrun_dir, f'predictions_{generation_mode}_{datasubset}.pkl')
    
    # Convert any tensors to lists/native Python types
    outputs_to_save = {}
    for dataloader_idx, hypotheses in outputs.items():
        outputs_to_save[dataloader_idx] = {}
        for hyp_key, hyp_data in hypotheses.items():
            outputs_to_save[dataloader_idx][hyp_key] = {}
            for k, v in hyp_data.items():
                if isinstance(v, torch.Tensor):
                    outputs_to_save[dataloader_idx][hyp_key][k] = v.cpu().tolist()
                else:
                    outputs_to_save[dataloader_idx][hyp_key][k] = v

    with open(save_path, 'wb') as f:
        pickle.dump(outputs_to_save, f)

    save_path_txt = osp.join(subrun_dir, f'generated_captions_{generation_mode}_{datasubset}.txt')

    idx_base = list(outputs_to_save.keys())[0]
    n_hyps = len(outputs_to_save[idx_base])
    batch_size = len(outputs_to_save[idx_base][f'hypothesis_0'][0]['fname'])
    with open(save_path_txt, 'w') as f:
        for i_file in range(len(outputs_to_save[idx_base][f'hypothesis_0'])):
            for elt in range(batch_size):
                f.write('----- File {} -----\n'.format(outputs_to_save[idx_base][f'hypothesis_0'][i_file]['fname'][elt]))
                gts = outputs_to_save[idx_base][f'hypothesis_0'][i_file]['mrefs'][elt]
                f.write('GT:   '+'\n')
                for i_gt in range(len(gts)):
                    f.write('      '+gts[i_gt]+'\n')
                for hypothesis_idx in range(n_hyps) :
                    f.write('Hypothesis '+str(hypothesis_idx)+'\n')
                    f.write('Pred: '+outputs_to_save[idx_base][f'hypothesis_{hypothesis_idx}'][i_file]['cands'][elt]+'\n')
        
    return save_path

def update_output_test_losses(model, test_dataset, processor,num_hyps, dataloader_idx, key, decoded_predictions, outputs, data_module, training_args):
    # New dataloader with original batch
    trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=processor,
            data_collator=data_module.batch_processor_fn_mrefs,
            eval_dataset=test_dataset,
        )
    test_dataloader = trainer.get_test_dataloader(test_dataset)
    data_module._test_collate = data_module.batch_processor_fn_mrefs
    N_examples = training_args.per_device_eval_batch_size

    for batch_idx, output_processed in enumerate(test_dataloader):
        for hyp_idx in range(num_hyps):
            # Check that the fnames match
            assert (np.array(outputs[dataloader_idx][f'hypothesis_{hyp_idx}'][batch_idx]["fname"]) == np.array(output_processed["fname"])).all()
            ### Teachforcing for loss computation
            N_refs = len([key for key in output_processed.keys() if "ref" in key and "labels" in key])
            outputs[dataloader_idx][f'hypothesis_{hyp_idx}'][batch_idx]["losses"] = [0]*N_examples # list of tensors
            output_ids_per_ref = []
            for i in range(N_refs):
                inputs_refs_i = {}
                for key in output_processed.keys():
                    if f"ref_{i}" in key:
                        inputs_refs_i[key.split(f"_ref_{i}")[0]] = output_processed[key]
                # Set the wta training mode to wta
                inputs_refs_i['wta_training_mode'] = 'wta'
                # output_ids_per_ref.append([])
                batch_size = inputs_refs_i['input_ids'].shape[0]
                assert batch_size == 1, "Batch size is not 1, and the code is not adapted to this"
                with torch.no_grad(): ## FORWARD instead of GENERATE here to get the loss
                    # for j in range(batch_size):
                    #     output_ids_per_ref[i].append(model( 
                    #         **{'input_ids': inputs_refs_i['input_ids'][j].unsqueeze(0),
                    #         'attention_mask': inputs_refs_i['attention_mask'][j].unsqueeze(0),
                    #         'input_features': inputs_refs_i['input_features'][j].unsqueeze(0),
                    #         'feature_attention_mask': inputs_refs_i['feature_attention_mask'][j].unsqueeze(0),
                    #         'labels': inputs_refs_i['labels'][j].unsqueeze(0)},
                    #         hypothesis_idx=hyp_idx,
                    #         return_all_losses=True
                        # )) # Setting the batch size to 1 to get the loss for each example
                    output_ids_per_ref.append(model( 
                        **inputs_refs_i,
                        hypothesis_idx=hyp_idx,
                        return_all_losses=True
                    ))
            # Transform from a len of N_refs list of len(decoded_predictions) tensors to a len(decoded_predictions) list of tensors of shape (N_refs,)
            for j in range(batch_size):
                outputs[dataloader_idx][f'hypothesis_{hyp_idx}'][batch_idx]["losses"][j] = torch.stack([output_ids_per_ref[i][j] for i in range(N_refs)]).cpu()
        
    return outputs

def time_mask(spectrogram, max_mask_percentage=0.1):
    # Calculate max_mask_width as percentage of time steps
    max_mask_width = int(spectrogram.shape[2] * max_mask_percentage)
    if max_mask_width == 0:
        max_mask_width = 1
    mask_width = np.random.randint(0, max_mask_width)
    start = np.random.randint(0, spectrogram.shape[2] - mask_width)
    spectrogram[:, :, start:start + mask_width] = 0
    return spectrogram

def freq_mask(spectrogram, max_mask_percentage=0.15):
    # Calculate max_mask_width as percentage of frequency bins
    max_mask_width = int(spectrogram.shape[1] * max_mask_percentage)
    if max_mask_width == 0:
        max_mask_width = 1
    mask_width = np.random.randint(0, max_mask_width)
    start = np.random.randint(0, spectrogram.shape[1] - mask_width)
    spectrogram[:, start:start + mask_width, :] = 0
    return spectrogram

def time_warp(spectrogram, max_warp=50):
    # Randomly stretch or compress time steps
    warp_amount = np.random.randint(-max_warp, max_warp)
    if warp_amount > 0:
        # Stretch
        spectrogram = torch.nn.functional.interpolate(
            spectrogram.unsqueeze(0), 
            size=(spectrogram.shape[1], spectrogram.shape[2] + warp_amount),
            mode='bilinear'
        ).squeeze(0)
    else:
        # Compress
        spectrogram = torch.nn.functional.interpolate(
            spectrogram.unsqueeze(0), 
            size=(spectrogram.shape[1], spectrogram.shape[2] + warp_amount),
            mode='bilinear'
        ).squeeze(0)
    return spectrogram

def spec_augment(spectrogram, time_mask_percentage=0.1, freq_mask_percentage=0.15):
    # Apply both time and frequency masking
    spectrogram = time_mask(spectrogram, time_mask_percentage)
    spectrogram = freq_mask(spectrogram, freq_mask_percentage)
    return spectrogram

def apply_tta(tta_mode, original_features, noise_std, time_mask_percentage, freq_mask_percentage):
    # original features of shape (batch_size, number of frequency bins, number of time steps)
    if tta_mode == "gauss":
        noisy_features = original_features.clone()
        noise = torch.randn_like(noisy_features) * noise_std
        noisy_features = noisy_features + noise
        return noisy_features

    elif tta_mode == "spec_augment":
        # Apply SpecAugment
        noisy_features = original_features.clone()
        noisy_features = spec_augment(noisy_features, time_mask_percentage=time_mask_percentage, freq_mask_percentage=freq_mask_percentage)
        return noisy_features
    
    elif tta_mode == "time_warp":
        # Apply time warping
        noisy_features = original_features.clone()
        noisy_features = time_warp(noisy_features)
        return noisy_features
    
    else:
        raise ValueError(f"Unknown TTA mode: {tta_mode}")

def generate_predictions(model, test_dataloader, processor, device, num_hyps, dataloader_idx, key, generation_config, cfg):
    outputs = {}
    outputs[dataloader_idx] = {}

    # Check if TTA is enabled
    tta_enabled = cfg.model.tta_enabled
    if tta_enabled:
        num_tta_samples = generation_config.num_return_sequences
        tta_mode = cfg.model.tta_mode
        noise_std = cfg.model.gaussian_noise_std
        logger.info(f"TTA enabled with {num_tta_samples} samples")
        time_mask_percentage = cfg.model.time_mask_percentage
        freq_mask_percentage = cfg.model.freq_mask_percentage

    if num_hyps == 1 :
        ### Init keys
        num_hyps_considered = generation_config.num_return_sequences

        if cfg.model.use_moe_lora is True and (str(cfg.model.stochastic_router).lower() == "true" or str(cfg.model.expert_specific).lower() == "true"):
            num_hyps_considered = cfg.model.num_experts
            if generation_config.num_return_sequences != 1:
                generation_config.num_return_sequences = 1
                logger.info(f"Greedy mode detected in MoE Fixed/Stochastic Router, setting num_return_sequences to 1 with {cfg.model.num_experts} samples")

        ### Init keys
        for hyp_idx in range(num_hyps_considered):
            outputs[dataloader_idx][f'hypothesis_{hyp_idx}'] = {}
            for batch_idx, batch in enumerate(test_dataloader):
                outputs[dataloader_idx][f'hypothesis_{hyp_idx}'][batch_idx] = {}
                for key in ["cands", "losses", "mrefs", "fname", "subset", "index", "dataset"]:
                    outputs[dataloader_idx][f'hypothesis_{hyp_idx}'][batch_idx][key] = {}

        if tta_enabled:
            # Check if we are in greedy mode (this trick allow to bypass the errors in HF library when num_beam < num_return_sequences)
            # i.e., if cfg.model.greedy_sh_tta = "Single" in a config file, it means that the evaluation was performed with beam size 1 
            # in each forward pass. If, cfg.model.greedy_sh_tta = "Double", it means that the evaluation was performed with beam size 2 in each forward pass.
            if generation_config.do_sample == False and cfg.model.greedy_sh_tta == "Single":
                generation_config.num_beams = 1
                if generation_config.num_return_sequences != 1:
                    generation_config.num_return_sequences = 1
                    logger.info(f"Greedy mode detected in TTA, setting num_return_sequences to 1 with {num_tta_samples} samples")

            if generation_config.do_sample == False and cfg.model.greedy_sh_tta == "Double":
                generation_config.num_beams = 2
                if generation_config.num_return_sequences != 1:
                    generation_config.num_return_sequences = 1
                    logger.info(f"Greedy mode detected in TTA, setting num_return_sequences to 1 with {num_tta_samples} samples")

        for batch_idx, batch in enumerate(test_dataloader):
            ### Generate predictions
            with torch.no_grad():
                if cfg.model.use_moe_lora is True and (str(cfg.model.stochastic_router).lower() == "true" or str(cfg.model.expert_specific).lower() == "true"):
                    # Generate predictions for sample
                    for sample_idx in range(cfg.model.num_experts):
                        # Generate 
                        tuple_output = model.generate(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            input_features=batch["input_features"],
                            feature_attention_mask=batch["feature_attention_mask"],
                            generation_config=generation_config,
                            hypothesis_idx=sample_idx,  # We are in the single hypothesis case, only the input changes
                            output_scores=generation_config.output_scores,
                            return_dict_in_generate=generation_config.return_dict_in_generate
                        )
                        
                        if type(tuple_output) == torch.Tensor:
                            output_ids = tuple_output
                            scores = None
                        else:
                            output_ids = tuple_output.sequences
                            scores = None
                            
                        # Process predictions for this sample
                        if output_ids.shape[0] != cfg.model.per_device_eval_batch_size*generation_config.num_return_sequences:
                            actual_batch_size = output_ids.shape[0] // generation_config.num_return_sequences
                            output_ids = output_ids.view(actual_batch_size, generation_config.num_return_sequences, -1)
                        else:
                            output_ids = output_ids.view(cfg.model.per_device_eval_batch_size, generation_config.num_return_sequences, -1)

                        decoded_predictions = processor.batch_decode(
                            output_ids[:,0,:], skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )
                        # Remove newlines and backslashes, and Chinese characters
                        decoded_predictions = [
                            re.sub(r'[\u4e00-\u9fff]|\r|\v|\f|\u2028|\u2029', '', 
                                s.replace('\n', '').replace('\\', '').replace('_', '')).strip()
                            for s in decoded_predictions
                        ]
                        decoded_prompt = processor.batch_decode(
                            batch["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )
                        decoded_prompt = [e.strip() for e in decoded_prompt]

                        if len(decoded_predictions) == len(decoded_prompt):
                            for i in range(len(decoded_predictions)):
                                if decoded_prompt[i] in decoded_predictions[i]:
                                    decoded_predictions[i] = decoded_predictions[i].split(decoded_prompt[i])[-1].strip()
                        
                        # Store predictions for this TTA sample
                        outputs[dataloader_idx][f'hypothesis_{sample_idx}'][batch_idx]["cands"] = decoded_predictions
                        outputs[dataloader_idx][f'hypothesis_{sample_idx}'][batch_idx]["preds"] = output_ids[:,0,:].cpu().numpy()
                        if scores is not None:
                            outputs[dataloader_idx][f'hypothesis_{sample_idx}'][batch_idx]["lprobs"] = scores[:,0,:].cpu().numpy()
                        outputs[dataloader_idx][f'hypothesis_{sample_idx}'][batch_idx]["mrefs"] = batch["mrefs"]
                        outputs[dataloader_idx][f'hypothesis_{sample_idx}'][batch_idx]["fname"] = batch["fname"]
                        outputs[dataloader_idx][f'hypothesis_{sample_idx}'][batch_idx]["subset"] = batch['subset']
                        outputs[dataloader_idx][f'hypothesis_{sample_idx}'][batch_idx]["index"] = batch['index']
                        outputs[dataloader_idx][f'hypothesis_{sample_idx}'][batch_idx]["dataset"] = batch['dataset']
                elif tta_enabled:
                    # Store original features
                    original_features = batch["input_features"].clone()
                    
                    # Generate predictions for each TTA sample
                    for tta_idx in range(num_tta_samples):
                        # Add Gaussian noise to input features

                        noisy_features = apply_tta(tta_mode=tta_mode, original_features=original_features, noise_std=noise_std, time_mask_percentage=time_mask_percentage, freq_mask_percentage=freq_mask_percentage)

                        # Generate with noisy features
                        tuple_output = model.generate(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            input_features=noisy_features,
                            feature_attention_mask=batch["feature_attention_mask"],
                            generation_config=generation_config,
                            hypothesis_idx=0,  # We are in the single hypothesis case, only the input changes
                            output_scores=generation_config.output_scores,
                            return_dict_in_generate=generation_config.return_dict_in_generate
                        )
                        
                        if type(tuple_output) == torch.Tensor:
                            output_ids = tuple_output
                            scores = None
                        else:
                            output_ids = tuple_output.sequences
                            scores = None
                            
                        # Process predictions for this TTA sample
                        if output_ids.shape[0] != cfg.model.per_device_eval_batch_size*generation_config.num_return_sequences:
                            actual_batch_size = output_ids.shape[0] // generation_config.num_return_sequences
                            output_ids = output_ids.view(actual_batch_size, generation_config.num_return_sequences, -1)
                        else:
                            output_ids = output_ids.view(cfg.model.per_device_eval_batch_size, generation_config.num_return_sequences, -1)

                        decoded_predictions = processor.batch_decode(
                            output_ids[:,0,:], skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )
                        # Remove newlines and backslashes, and Chinese characters
                        decoded_predictions = [
                            re.sub(r'[\u4e00-\u9fff]|\r|\v|\f|\u2028|\u2029', '', 
                                s.replace('\n', '').replace('\\', '').replace('_', '')).strip()
                            for s in decoded_predictions
                        ]
                        decoded_prompt = processor.batch_decode(
                            batch["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )
                        decoded_prompt = [e.strip() for e in decoded_prompt]

                        if len(decoded_predictions) == len(decoded_prompt):
                            for i in range(len(decoded_predictions)):
                                if decoded_prompt[i] in decoded_predictions[i]:
                                    decoded_predictions[i] = decoded_predictions[i].split(decoded_prompt[i])[-1].strip()
                        
                        # Store predictions for this TTA sample
                        outputs[dataloader_idx][f'hypothesis_{tta_idx}'][batch_idx]["cands"] = decoded_predictions
                        outputs[dataloader_idx][f'hypothesis_{tta_idx}'][batch_idx]["preds"] = output_ids[:,0,:].cpu().numpy()
                        if scores is not None:
                            outputs[dataloader_idx][f'hypothesis_{tta_idx}'][batch_idx]["lprobs"] = scores[:,0,:].cpu().numpy()
                        outputs[dataloader_idx][f'hypothesis_{tta_idx}'][batch_idx]["mrefs"] = batch["mrefs"]
                        outputs[dataloader_idx][f'hypothesis_{tta_idx}'][batch_idx]["fname"] = batch["fname"]
                        outputs[dataloader_idx][f'hypothesis_{tta_idx}'][batch_idx]["subset"] = batch['subset']
                        outputs[dataloader_idx][f'hypothesis_{tta_idx}'][batch_idx]["index"] = batch['index']
                        outputs[dataloader_idx][f'hypothesis_{tta_idx}'][batch_idx]["dataset"] = batch['dataset']
                else:
                    tuple_output = model.generate(input_ids=batch["input_ids"],
                                                attention_mask=batch["attention_mask"],
                                                input_features=batch["input_features"],
                                                feature_attention_mask=batch["feature_attention_mask"],
                                                generation_config=generation_config,
                                                hypothesis_idx=0,
                                                output_scores=generation_config.output_scores,
                                                return_dict_in_generate=generation_config.return_dict_in_generate)
                    if type(tuple_output) == torch.Tensor:
                        output_ids = tuple_output
                        scores = None
                    else:
                        output_ids = tuple_output.sequences
                        # scores = tuple_output.scores.view(cfg.model.per_device_train_batch_size, generation_config.num_return_sequences, -1)
                        scores = None
                    # output_ids is a tensor of shape (num_return_sequences*batch_size, max_length)
                    if output_ids.shape[0] != cfg.model.per_device_eval_batch_size*generation_config.num_return_sequences:
                        actual_batch_size = output_ids.shape[0] // generation_config.num_return_sequences
                        output_ids = output_ids.view(actual_batch_size, generation_config.num_return_sequences, -1)
                    else:
                        output_ids = output_ids.view(cfg.model.per_device_eval_batch_size, generation_config.num_return_sequences, -1)

                    for hyp_idx in range(generation_config.num_return_sequences):                        
                        decoded_predictions = processor.batch_decode(
                            output_ids[:,hyp_idx,:], skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )
                        # Remove newlines and backslashes, and Chinese characters
                        decoded_predictions = [
                            re.sub(r'[\u4e00-\u9fff]|\r|\v|\f|\u2028|\u2029', '', 
                                s.replace('\n', '').replace('\\', '').replace('_', '')).strip()
                            for s in decoded_predictions
                        ]
                        decoded_prompt = processor.batch_decode(
                            batch["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )
                        decoded_prompt = [e.strip() for e in decoded_prompt]

                        if len(decoded_predictions) == len(decoded_prompt):
                            for i in range(len(decoded_predictions)):
                                if decoded_prompt[i] in decoded_predictions[i]:
                                    decoded_predictions[i] = decoded_predictions[i].split(decoded_prompt[i])[-1].strip()           
                
                        # Generated data
                        outputs[dataloader_idx][f'hypothesis_{hyp_idx}'][batch_idx]["cands"] = decoded_predictions # list of strings
                        outputs[dataloader_idx][f'hypothesis_{hyp_idx}'][batch_idx]["preds"] = output_ids[:,hyp_idx,:].cpu().numpy() # list of strings
                        if scores is not None:
                            outputs[dataloader_idx][f'hypothesis_{hyp_idx}'][batch_idx]["lprobs"] = scores[:,hyp_idx,:].cpu().numpy() # list of strings
                        # Labels
                        outputs[dataloader_idx][f'hypothesis_{hyp_idx}'][batch_idx]["mrefs"] = batch["mrefs"] # list of lists of strings
                        outputs[dataloader_idx][f'hypothesis_{hyp_idx}'][batch_idx]["fname"] = batch["fname"] # list of file names (.wav extensions).
                        outputs[dataloader_idx][f'hypothesis_{hyp_idx}'][batch_idx]["subset"] = batch['subset'] # list of split names, e.g., ['val', 'val', ...]
                        outputs[dataloader_idx][f'hypothesis_{hyp_idx}'][batch_idx]["index"] = batch['index'] # list of indices, e.g., [0, 1, 2, ...]
                        outputs[dataloader_idx][f'hypothesis_{hyp_idx}'][batch_idx]["dataset"] = batch['dataset'] # list of dataset names, e.g., ['clotho', 'clotho', ...]

    else:
        # Setting num return sequences to 1
        generation_config.num_return_sequences = 1
        for hyp_idx in range(num_hyps):
            outputs[dataloader_idx][f'hypothesis_{hyp_idx}'] = {}
            for batch_idx, batch in enumerate(test_dataloader):
                ### Generate predictions
                outputs[dataloader_idx][f'hypothesis_{hyp_idx}'][batch_idx] = {}
                for key in ["cands", "losses", "mrefs", "fname", "subset", "index", "dataset"]:
                    outputs[dataloader_idx][f'hypothesis_{hyp_idx}'][batch_idx][key] = {}
                with torch.no_grad():
                    tuple_output = model.generate(input_ids=batch["input_ids"],
                                                attention_mask=batch["attention_mask"],
                                                input_features=batch["input_features"],
                                                feature_attention_mask=batch["feature_attention_mask"],
                                                generation_config=generation_config,
                                                hypothesis_idx=hyp_idx,
                                                output_scores=generation_config.output_scores,
                                                return_dict_in_generate=generation_config.return_dict_in_generate)
                    if type(tuple_output) == torch.Tensor:
                        output_ids = tuple_output
                        scores = None
                    else:
                        output_ids = tuple_output.sequences
                        # scores = tuple_output.scores.view(cfg.model.per_device_train_batch_size, generation_config.num_return_sequences, -1)
                        scores = None
                    decoded_predictions = processor.batch_decode(
                        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    # decoded_predictions = [s.replace('\n', '').replace('\\', '') for s in decoded_predictions]
                    decoded_predictions = [
                        re.sub(r'[\u4e00-\u9fff]|\r|\v|\f|\u2028|\u2029', '', 
                            s.replace('\n', '').replace('\\', '').replace('_', '')).strip()
                        for s in decoded_predictions
                    ]
                    decoded_prompt = processor.batch_decode(
                        batch["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    decoded_prompt = [e.strip() for e in decoded_prompt]
                    if len(decoded_predictions) == len(decoded_prompt): # the length is the batch size
                        for i in range(len(decoded_predictions)):
                            if decoded_prompt[i] in decoded_predictions[i]:
                                decoded_predictions[i] = decoded_predictions[i].split(decoded_prompt[i])[-1].strip()
                    # Generated data
                    outputs[dataloader_idx][f'hypothesis_{hyp_idx}'][batch_idx]["cands"] = decoded_predictions # list of strings
                    outputs[dataloader_idx][f'hypothesis_{hyp_idx}'][batch_idx]["preds"] = output_ids.cpu().numpy() # shape (batch_size, max_length)
                    if scores is not None:
                        outputs[dataloader_idx][f'hypothesis_{hyp_idx}'][batch_idx]["lprobs"] = scores[:,hyp_idx,:].cpu().numpy() # list of strings
                    # Labels
                    outputs[dataloader_idx][f'hypothesis_{hyp_idx}'][batch_idx]["mrefs"] = batch["mrefs"] # list of lists of strings
                    outputs[dataloader_idx][f'hypothesis_{hyp_idx}'][batch_idx]["fname"] = batch["fname"] # list of file names (.wav extensions).
                    outputs[dataloader_idx][f'hypothesis_{hyp_idx}'][batch_idx]["subset"] = batch['subset'] # list of split names, e.g., ['val', 'val', ...]
                    outputs[dataloader_idx][f'hypothesis_{hyp_idx}'][batch_idx]["index"] = batch['index'] # list of indices, e.g., [0, 1, 2, ...]
                    outputs[dataloader_idx][f'hypothesis_{hyp_idx}'][batch_idx]["dataset"] = batch['dataset'] # list of dataset names, e.g., ['clotho', 'clotho', ...]

    return outputs, decoded_predictions

def limit_dataset_size(dataset, max_batches, per_device_eval_batch_size):
    max_samples = max_batches * per_device_eval_batch_size
    if max_samples < 0:
        max_samples = len(dataset)
    logger.info(f"Limiting dataset to {max_samples} samples ({max_batches} batches)")
    return torch.utils.data.Subset(
        dataset, 
        range(min(max_samples, len(dataset)))
    )

def convert_to_dot_access(config):
    """Convert a PretrainedConfig object to allow dot notation access"""
    # Only process dictionary values that are not already PretrainedConfig instances
    for key, value in config.__dict__.items():
        if isinstance(value, dict) and not isinstance(value, PretrainedConfig):
            # Create a new config instance for this dictionary
            new_config = PretrainedConfig.from_dict(value)
            # Only recurse if the new config has dictionary attributes
            if any(isinstance(v, dict) for v in new_config.__dict__.values()):
                convert_to_dot_access(new_config)
            setattr(config, key, new_config)
    return config

# A common dummy model creator.
def create_dummy_model(max_length, num_hyps, cache_dir, cfg):
    """
    Create a tiny version of the model (for debugging or when load_model==False).
    This version also supports optional 4-bit quantization if configured.
    """
    start_time = time.time()
    nf4_config = None
    if cfg.model.get("quantization", False):
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    # # Use the `init_model_dir` if provided, else default to pretrained_repo.
    init_path = cfg.model.get("init_model_dir", cfg.model.pretrained_repo)

    config = AutoConfig.from_pretrained(
        init_path,
        cache_dir=cache_dir,
        local_files_only=True,
        low_cpu_mem_usage=True,  # Using the global flag from above
        device_map="cuda",
        # Generation-related parameters from config (if specified)
        num_beams=cfg.model.get("generation_config", {}).get("beam_size", None),
        num_return_sequences=cfg.model.get("generation_config", {}).get("num_return_sequences", None),
        do_sample=cfg.model.get("generation_config", {}).get("do_sample", None),
        temperature=cfg.model.get("generation_config", {}).get("temperature", None),
        top_k=cfg.model.get("generation_config", {}).get("top_k", None),
        top_p=cfg.model.get("generation_config", {}).get("top_p", None),
        diversity_penalty=cfg.model.get("generation_config", {}).get("diversity_penalty", None),
        num_beam_groups=cfg.model.get("generation_config", {}).get("num_beam_groups", None),
        quantization=nf4_config if nf4_config is not None else None,
        chat_format=cfg.model.generation_config.chat_format,
        penalty_alpha=cfg.model.generation_config.penalty_alpha,
    )
    config.num_hyps = num_hyps
    config.audio_config.max_source_positions = int(1500 * max_length / (30 * 16000))
    config.wta_training_mode = cfg.model.wta_training_mode
    config.wta_params_epsilon = cfg.model.wta_params_epsilon
    config.wta_params_ini_temp = cfg.model.wta_params_ini_temp
    config.wta_params_fin_temp = cfg.model.wta_params_fin_temp
    config.wta_params_decay_rate = cfg.model.wta_params_decay_rate
    config.wta_params_schedule_mode = cfg.model.wta_params_schedule_mode
    config.tta_enabled = cfg.model.tta_enabled
    config.tta_mode = cfg.model.tta_mode
    # In debug mode, downscale network dimensions.
    if cfg.model.is_debug:
        config.audio_config.encoder_ffn_dim = 1
        config.audio_config.encoder_attention_heads = 1
        config.audio_config.num_hidden_layers = 1
        config.text_config.num_hidden_layers = 1
        config.text_config.num_attention_heads = 1
        config.text_config.hidden_size = 1
        config.text_config.intermediate_size = 1
        config.text_config.num_key_value_heads = 1

    logger.info(f"Time taken to load config: {time.time() - start_time:.2f} seconds")
    start_time = time.time()
    if cfg.model.native_lora_enabled and cfg.model.native_group_lora_enabled is False:
        from transformers import Qwen2AudioForConditionalGeneration_MH_Lora
        model = Qwen2AudioForConditionalGeneration_MH_Lora(config=config)
    elif cfg.model.native_group_lora_enabled:
        config.native_group_lora_enabled = True
        from transformers import Qwen2AudioForConditionalGeneration_MH_Group_Lora
        model = Qwen2AudioForConditionalGeneration_MH_Group_Lora(config=config)
    else:
        from transformers import Qwen2AudioForConditionalGeneration_MH
        model = Qwen2AudioForConditionalGeneration_MH(config=config)
    logger.info(f"Time taken to load model: {time.time() - start_time:.2f} seconds")
    return model
