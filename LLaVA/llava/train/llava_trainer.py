# This file is part of LLaVA (https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py)

import os
import torch
import torch.nn as nn

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.total_losses = []
        self.wta_loss = []
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            # Convert inputs to bfloat16 where applicable
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    # Skip tensors that shouldn't be converted (like labels or attention masks)
                    if key not in ['labels', 'attention_mask']:
                        inputs[key] = value.to(torch.bfloat16)
            #breakpoint()
            outputs = model(**inputs)
            logits = outputs[0]
            # Save extra_info in self for later access
            self.total_losses.append(outputs.total_loss.cpu())
   
            self.wta_loss.append(outputs.wta_loss.cpu())

        return None, logits, inputs['labels']


    def get_extra_info(self):
        # Merge all the batches together
        if not self.total_losses:
            return None
        
        import torch
        tot_losses = torch.cat(self.total_losses, dim=1)
        try:
            wta_losses = torch.cat(self.wta_loss, dim=0)
        except:
            wta_losses = torch.Tensor(self.wta_loss)
        # For distributed training, gather results from all processes
        '''if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            gathered_res = [torch.zeros_like(res) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(gathered_res, res)
            res = torch.cat(gathered_res, dim=0)'''
        # Don't reset total_losses here as it might be accessed multiple times
        # during evaluation - instead, reset it at the beginning of evaluation
        return tot_losses, wta_losses

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)

    def reset_total_losses(self):
        """Reset the total_losses list to start fresh for a new evaluation."""
        self.total_losses = []

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to reset total_losses at the beginning."""
        # Reset total_losses before starting a new evaluation
        self.reset_total_losses()
        
        # Call the parent evaluate method
        return super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )
    
    def evaluation_loopEdited(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        self.model.eval()

        '''_ = self.model.base_model(
            input_ids=torch.zeros(1, 1, dtype=torch.long).to(self.model.device),
            attention_mask=torch.ones(1, 1, dtype=torch.long).to(self.model.device),
            labels=torch.zeros(1, 1, dtype=torch.long).to(self.model.device),

        )'''
        from aac_metrics.functional import cider_d 
        import numpy as np  
        dictMetric = {}   
        sentHyp=[]
        print('number of hypotheses',self.model.config.num_hyps)  
        for hypothesis_idx in range(self.model.config.num_hyps):
            predictions = []
            labels = []
            for batch in dataloader:
                inputs = batch["input_ids"].to(self.model.device)

                # Keep only the tokens at positions where labels are not -100
                labels_mask = batch["labels"] == -100
                
                # Create a new tensor to store the filtered input tokens
                filtered_inputs = []
                
                # Process each example in the batch
                for i in range(inputs.size(0)):
                    # Get the positions where labels are not -100 for this example
                    valid_positions = labels_mask[i]
                    
                    # Extract tokens at those positions
                    valid_tokens = inputs[i][valid_positions]
                    
                    # Add to the filtered inputs list
                    filtered_inputs.append(valid_tokens)
                
                # Replace the original inputs with the filtered version
                # Note: We're keeping the original inputs for generation, but using filtered_inputs for inspection
                inputs = filtered_inputs[0] if filtered_inputs else torch.tensor([], device=inputs.device)
                inputs=inputs.unsqueeze(0)

                attention_mask = batch["attention_mask"].to(self.model.device)[:,:inputs.size(1)]
                labels.append(batch["labels"])#.cpu().numpy())
                
                with torch.inference_mode():
                    # Use the preprocessed images directly from the batch instead of reprocessing them
                    processed_images = batch["images"].to(self.model.device, dtype=self.model.dtype)
                    generated_ids = self.model.base_model.generate(
                        inputs,
                        hypothesis_idx=hypothesis_idx,
                        images=processed_images,
                        attention_mask=attention_mask,
                        pad_token_id=self.model.config.pad_token_id,
                        max_new_tokens=128,
                        bos_token_id=self.model.config.bos_token_id,
                        eos_token_id=self.model.config.eos_token_id,
                        do_sample=False,
                        num_beams=1,
                        use_cache=True,
                        repetition_penalty=1.1
                    )
                    predictions.extend(generated_ids.cpu().numpy())
            # Compute metrics
            # Detokenize predictions using the tokenizer
            # Decode both predictions and labels
            decoded_predictions = []
            decoded_labels = []
            
            # Process predictions
            for pred_ids in predictions:
                # Remove padding tokens and special tokens
                # Find the first occurrence of the pad token or EOS token
                if self.model.config.pad_token_id in pred_ids:
                    pred_ids = pred_ids[:np.where(pred_ids == self.model.config.pad_token_id)[0][0]]
                if self.model.config.eos_token_id in pred_ids:
                    pred_ids = pred_ids[:np.where(pred_ids == self.model.config.eos_token_id)[0][0] + 1]
                    
                # Convert to tensor for decoding
                pred_tensor = torch.tensor(pred_ids)
                decoded_text = self.tokenizer.decode(pred_tensor, skip_special_tokens=True)
                decoded_predictions.append(decoded_text)
            
            # Process labels - flatten the list first if needed
            flat_labels = [item for sublist in labels for item in sublist]
            
            for label_ids in flat_labels:
                # Convert to numpy if it's a tensor
                if isinstance(label_ids, torch.Tensor):
                    label_ids = label_ids.cpu().numpy()
                    
                # Remove padding tokens (-100 is typically used for ignored positions)
                label_ids = label_ids[label_ids != -100]
                
                # Find the first occurrence of the pad token or EOS token if present
                if self.model.config.pad_token_id in label_ids:
                    label_ids = label_ids[:np.where(label_ids == self.model.config.pad_token_id)[0][0]]
                if self.model.config.eos_token_id in label_ids:
                    label_ids = label_ids[:np.where(label_ids == self.model.config.eos_token_id)[0][0] + 1]
                
                # Convert to tensor for decoding
                label_tensor = torch.tensor(label_ids)
                decoded_text = self.tokenizer.decode(label_tensor, skip_special_tokens=True)
                decoded_labels.append(decoded_text)
            
            # Replace the raw token IDs with decoded text
            predictions = decoded_predictions
            labels = decoded_labels

            print('predictions',predictions)
            # Reshape labels back to a list of lists to match the original structure
            # This assumes that we know how many labels were in each original sublist
            # We need to determine the original structure from the input data
        
            # Ensure predictions is a list of strings
            if not all(isinstance(pred, str) for pred in predictions):
                predictions = [str(pred) for pred in predictions]
            
            # Ensure labels is a list of lists of strings
            # The error suggests labels should be a list[list[str]] but we have a flat list
            # We need to restructure it based on the original structure
            
            # Since we flattened the labels earlier, we need to reconstruct the original structure
            # If we don't have the original structure information, we can wrap each label in a list
            # This assumes each reference has only one ground truth
            labels = [[label] for label in labels]
            
            score,sent=cider_d(predictions, labels)
            sentHyp.append(sent['cider_d'])
            dictMetric['cider_'+str(hypothesis_idx)]=float(score['cider_d'].item())
        bests=[]
        nbUtil=torch.zeros(len(sentHyp))
        for w in range(len(sentHyp[0])):
            bests.append(max([sentHyp[h][w] for h in range(len(sentHyp))]))
            argmax=np.argmax([sentHyp[h][w] for h in range(len(sentHyp))])
            nbUtil[argmax]+=1
            
        for h in range(len(nbUtil)):
            dictMetric['nbUtils_'+str(h)]=float(nbUtil[h])
        dictMetric['cider_oracle']=float(sum(bests)/len(bests))
        # Create a proper output object that matches what the Trainer expects
        # The Trainer expects an object with a 'metrics' attribute
        class EvalPredictionOutput:
            def __init__(self, metrics):
                self.metrics = metrics
        
        # Convert our dictionary of metrics to the expected output format
        output = EvalPredictionOutput(metrics=dictMetric)
        # Add num_samples attribute to match what Trainer expects
        # This should be the number of examples that were evaluated
        # We can determine this from the length of our predictions
        output.num_samples = len(predictions)
        
        # If there are any other attributes that the Trainer expects, add them here
        # For example, if the Trainer expects a predictions attribute
        output.predictions = predictions
        output.label_ids = labels
        # Return the properly formatted output object instead of just the dictionary
        return output
    
    '''def training_step(self, model, inputs):
        # Standard processing
        model.train()
        inputs = self._prepare_inputs(inputs)
        outputs = model(**inputs)
        loss = outputs.loss

        loss.backward()
        # Print gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name} grad norm: {param.grad.norm().item()}")
                if param.grad.norm().item()>0:
                    n=name
                    p=param
        print(n,p.grad.norm().item())
        breakpoint()
        return loss.detach()'''
