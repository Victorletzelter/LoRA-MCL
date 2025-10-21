
import os
import sys
import logging
import torch
import mlflow
import sys
sys.path.append(os.path.join(os.environ["PROJECT_ROOT"], ".."))

# Make sure to have your project root set up (if needed)
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Append paths if needed
sys.path.append(os.path.join(os.environ["PROJECT_ROOT"], "conette", "src"))
from dataloading import HDFDataModule  # Make sure this is accessible

# Configure the logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from utils import limit_dataset_size
from typing import Dict
if os.environ.get("USE_LOCAL_TRANSFORMERS", "false").lower() == "true":
    from local_transformers.trainer_utils import EvalPrediction
    from local_transformers.integrations import MLflowCallback
    from local_transformers.trainer_callback import EarlyStoppingCallback
    from local_transformers import (
    Trainer,
    TrainingArguments,
)
else:
    from transformers.trainer_utils import EvalPrediction
    from transformers.integrations import MLflowCallback
    from transformers.trainer_callback import EarlyStoppingCallback
    from transformers import (
    Trainer,
    TrainingArguments,
)
    from mcl_wrapper import MCLTrainer

import numpy as np

class CustomMLflowCallback(MLflowCallback):
    def setup(self, args, state, model, **kwargs):
        """
        Override setup to use existing MLflow run instead of creating a new one,
        while still initializing necessary attributes
        """
        if state.is_world_process_zero:
            # Initialize required attributes from parent class
            self._initialized = True
            self._async_log = False  # Set to False for synchronous logging
            self._ml_flow = mlflow
            
            # Log info about the run being used
            active_run = mlflow.active_run()
            if active_run:
                logger.info(f"Using existing MLflow run: {active_run.info.run_name}")
            else:
                logger.warning("No active MLflow run found!")

def get_peft_state(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: v for k, v in to_return.items()}
    return to_return

def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Computes metrics for evaluation.
    
    Args:
        eval_pred (EvalPrediction): Contains:
            - predictions: np.ndarray or tuple of np.ndarray
            - label_ids: np.ndarray
            - inputs (optional): anything that was passed as `inputs` to `include_for_metrics`
            - losses (optional): if `loss` was passed to `include_for_metrics`
            
    Returns:
        dict: Dictionary containing the computed metrics
    """
    import torch.distributed as dist
    is_distributed = dist.is_initialized() if hasattr(dist, "is_initialized") else False
    
    if is_distributed:
        # Get sample information 
        n_samples = getattr(eval_pred, 'n_samples', 0)
        n_hyps = getattr(eval_pred, 'n_hyps', 0)
        
        # Return empty dict if no data
        if n_samples == 0 or not hasattr(eval_pred, 'all_losses') or len(eval_pred.all_losses) == 0:
            return {}
        
        try:
            # Process the local data
            all_losses = eval_pred.all_losses.reshape(n_hyps, n_samples, order="F")
            best_indexes = np.argmin(all_losses, axis=0)
            
            # Count hypothesis selections - use absolute counts not normalized
            counts = {f'nt_hypothesis_{i}': 0 for i in range(n_hyps)}
            for j in range(n_samples):
                counts[f'nt_hypothesis_{best_indexes[j]}'] += 1
                
            # Calculate local loss sum (not mean)
            eval_loss_sum = np.sum(eval_pred.losses) if hasattr(eval_pred, 'losses') and eval_pred.losses is not None else 0.0
            
            # Return values and sample count - Trainer will handle aggregation
            return {
                "eval_loss_sum": eval_loss_sum,
                "eval_sample_count": n_samples,
                **counts
            }
            
        except Exception as e:
            print(f"Error in compute_metrics: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return {}

    if not is_distributed: 
        print("We are assuming batch size is 1")
        n_samples = eval_pred.n_samples if hasattr(eval_pred, "n_samples") else eval_pred.predictions[-1].shape[0]
        n_hyps = eval_pred.n_hyps if hasattr(eval_pred, "n_hyps") else eval_pred.predictions[-1].shape[-1]
        all_losses = eval_pred.all_losses if hasattr(eval_pred, "all_losses") else eval_pred.predictions[-1]
        all_losses = all_losses.reshape(n_hyps, n_samples, order="F")

        # Best indexes
        best_indexes = np.argmin(all_losses, axis=0) # shape (n_samples)

        number_of_times = {f'nt_hypothesis_{i}': 0 for i in range(n_hyps)}

        for j in range(n_samples):
            number_of_times[f'nt_hypothesis_{best_indexes[j]}'] += 1/n_samples

        # Mean eval loss
        eval_loss = np.mean(eval_pred.losses)

        return {"eval_wta_loss": eval_loss, **number_of_times}

def run_training(cfg, model, processor):

    if not cfg.model.native_lora_enabled:
        for name, param in model.named_parameters():
            if "multi_modal_projector" not in name:
                param.requires_grad = False
    else:
        logger.info("Using native LoRA")

        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        logger.info(f"Trainable parameters: {model.print_trainable_parameters()}")
        
    # Setup the data module in "fit" mode.
    data_module = HDFDataModule(
        processor=processor,
        root=cfg.model.data_root,
        train_hdfs=cfg.model.train_hdfs,
        val_hdfs=cfg.model.val_hdfs,
        test_hdfs=cfg.model.test_hdfs,  # also passed for later inference if needed
        bsize=cfg.model.per_device_train_batch_size,
        n_workers=cfg.model.n_workers,
        pin_memory=True,
        verbose=1,
        train_tokenizer=processor.tokenizer if hasattr(processor, "tokenizer") else None,
        audio_padding=cfg.model.audio_padding,
        text_padding=cfg.model.text_padding,
        train_cols=cfg.model.train_cols,
        val_cols=cfg.model.val_cols,
        test_cols=cfg.model.test_cols,
        max_length=cfg.model.max_length,
        prompt_template_with_space=cfg.model.prompt_template_with_space,
        add_eos_token_data=cfg.model.add_eos_token_data,
        use_mcl_wrapper=cfg.model.use_mcl_wrapper
    )
    data_module.setup("fit")
    train_dataset = data_module._train_dset
    val_dataset = data_module._val_dset

    # Limit the dataset size if specified
    if cfg.get('max_train_batches', None):
        train_dataset = limit_dataset_size(train_dataset, cfg.max_train_batches, cfg.model.per_device_train_batch_size)

    if cfg.get('max_val_batches', None):
        val_dataset = limit_dataset_size(val_dataset, cfg.max_val_batches, cfg.model.per_device_eval_batch_size)

    # Build training arguments.
    training_args = TrainingArguments(
        local_rank=os.environ.get("LOCAL_RANK", -1),
        output_dir=cfg.paths.output_dir,
        logging_dir=cfg.paths.output_dir,
        logging_strategy=cfg.model.logging_strategy,
        logging_steps=cfg.model.logging_steps,
        dataloader_num_workers=cfg.model.dataloader_num_workers,
        torch_compile=cfg.model.torch_compile,
        eval_strategy=cfg.model.eval_strategy,
        save_strategy=cfg.model.save_strategy,
        eval_steps=cfg.model.eval_steps,
        gradient_accumulation_steps=cfg.model.gradient_accumulation_steps,
        eval_accumulation_steps=cfg.model.eval_accumulation_steps,
        optim=cfg.model.optim,
        adam_beta1=cfg.model.adam_beta1,
        adam_beta2=cfg.model.adam_beta2,
        adam_epsilon=cfg.model.adam_epsilon,
        learning_rate=cfg.model.learning_rate,
        lr_scheduler_type=cfg.model.lr_scheduler_type,
        lr_scheduler_kwargs={"min_lr_rate": cfg.model.min_lr_rate},
        warmup_ratio=cfg.model.warmup_ratio,
        per_device_train_batch_size=cfg.model.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.model.per_device_eval_batch_size,
        num_train_epochs=cfg.model.num_train_epochs,
        max_steps=cfg.model.max_steps,
        weight_decay=cfg.model.weight_decay,
        max_grad_norm=cfg.model.max_grad_norm,
        bf16=cfg.model.bf16,
        bf16_full_eval=cfg.model.bf16_full_eval,
        save_total_limit=cfg.model.save_total_limit,
        load_best_model_at_end=cfg.model.load_best_model_at_end,
        remove_unused_columns=cfg.model.remove_unused_columns,
        include_for_metrics=["loss", "inputs", "all_losses"],
        report_to=["tensorboard", "mlflow"] if cfg.mlflow.enabled else "all",
    )

    if cfg.model.use_mcl_wrapper is True and os.environ.get("USE_LOCAL_TRANSFORMERS", "false").lower() != "true":
        training_args.eval_do_concat_batches = False
        training_args.prediction_loss_only = True
        trainer = MCLTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=processor,
            data_collator=data_module.batch_processor_fn,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.model.early_stopping_patience, early_stopping_threshold=cfg.model.early_stopping_threshold),
                    CustomMLflowCallback()],
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=processor,
            data_collator=data_module.batch_processor_fn,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.model.early_stopping_patience, early_stopping_threshold=cfg.model.early_stopping_threshold),
                    CustomMLflowCallback()],
        )

    logger.info("Starting training ...")
    trainer.train()
    # Save the model to the provided output path.
    if cfg.model.do_save_model:
        if cfg.model.native_lora_enabled:
            peft_state_dict = get_peft_state(model.named_parameters(), cfg.model.native_lora_bias)
            model.save_pretrained(os.path.join(cfg.paths.output_dir, "adapter_model"), state_dict=peft_state_dict)
        else:
            trainer.save_model(cfg.paths.output_dir)
    logger.info("Training complete and model saved.")

    return model
