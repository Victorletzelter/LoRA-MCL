import os
import sys
import logging

if os.environ.get("USE_LOCAL_TRANSFORMERS", "false").lower() == "true":
    from local_transformers import (
        Trainer,
        TrainingArguments,
    )
else:
    from transformers import (
        Trainer,
        TrainingArguments,      
    )
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Append paths if needed
sys.path.append(os.environ["CONNETTE_PATH"])
from dataloading import HDFDataModule  # Make sure this is accessible

# Configure the logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from utils import save_predictions_to_pkl, update_output_test_losses, generate_predictions, limit_dataset_size, convert_to_dot_access, create_dummy_model

def reformat_outputs_bs1(outputs):
    new_outputs = {}
    for dataloader_idx in outputs.keys():
        new_outputs[dataloader_idx] = {}
        for hyp_key in outputs[dataloader_idx].keys():
            new_outputs[dataloader_idx][hyp_key] = {}
            absolute_batch_idx = 0
            for batch_idx in outputs[dataloader_idx][hyp_key].keys():
                for elt in range(len(outputs[dataloader_idx][hyp_key][batch_idx]['cands'])):
                    new_outputs[dataloader_idx][hyp_key][absolute_batch_idx] = {}
                    for key in outputs[dataloader_idx][hyp_key][batch_idx].keys():
                        if key == 'preds':
                            new_outputs[dataloader_idx][hyp_key][absolute_batch_idx][key] = outputs[dataloader_idx][hyp_key][batch_idx][key][elt][None,:]
                        elif key == 'index':
                            new_outputs[dataloader_idx][hyp_key][absolute_batch_idx][key] = [absolute_batch_idx]
                        elif key != 'losses':
                            new_outputs[dataloader_idx][hyp_key][absolute_batch_idx][key] = [outputs[dataloader_idx][hyp_key][batch_idx][key][elt]]
                    new_outputs[dataloader_idx][hyp_key][absolute_batch_idx]['losses'] = {}
                    absolute_batch_idx += 1
    return new_outputs

# (Optionally, reinitialize or reuse the data module.)
# Here we create a new instance if training was not done; else, reusing the one from training might be acceptable.
def run_inference(cfg, model, processor):
    data_module = HDFDataModule(
        processor=processor,
        root=cfg.model.data_root,
        train_hdfs=cfg.model.train_hdfs,
        val_hdfs=cfg.model.val_hdfs,
        test_hdfs=cfg.model.test_hdfs,
        bsize=cfg.model.per_device_eval_batch_size,
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
        use_peft_mcl=cfg.model.use_peft_mcl)

    print(f"data_module use_peft_mcl: {data_module.use_peft_mcl}")

    data_module.setup("test")
    test_datasets = data_module._test_dsets

    # Build inference arguments.
    inference_args = TrainingArguments(
        output_dir=cfg.model.output_dir,
        logging_dir=cfg.paths.output_dir,
        logging_strategy=cfg.model.logging_strategy,
        logging_steps=cfg.model.logging_steps,
        eval_strategy=cfg.model.eval_strategy,
        save_strategy=cfg.model.save_strategy,
        optim=cfg.model.optim,
        adam_beta1=cfg.model.adam_beta1,
        adam_beta2=cfg.model.adam_beta2,
        adam_epsilon=cfg.model.adam_epsilon,
        learning_rate=cfg.model.learning_rate,
        lr_scheduler_type=cfg.model.lr_scheduler_type,
        lr_scheduler_kwargs={"min_lr_rate": cfg.model.min_lr_rate, "num_warmup_steps": cfg.model.num_warmup_steps},
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
    )

    # Setting back the wta training mode for inference
    model.wta_training_mode = 'wta'

    idx_dataset = -1

    for key, test_dataset in test_datasets.items():

        if idx_dataset == -1:
            batch_size_eval = cfg.model.per_device_eval_batch_size
        else:
            cfg.model.per_device_eval_batch_size = batch_size_eval
        idx_dataset += 1
        inference_trainer = Trainer(
            model=model,
            args=inference_args,
            tokenizer=processor,
            data_collator=data_module.batch_processor_test_fn,
            eval_dataset=test_dataset,
        )

        # Limit the dataset size if specified
        if cfg.get('max_test_batches', None):
            test_dataset = limit_dataset_size(test_dataset, cfg.max_test_batches, cfg.model.per_device_eval_batch_size)

        logger.info("Running inference on the test dataset ...")

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        data_module._test_collate = data_module.batch_processor_test_fn
        test_dataloader = inference_trainer.get_test_dataloader(test_dataset)
        outputs, decoded_predictions = generate_predictions(model=model, 
                                                            test_dataloader=test_dataloader, 
                                                            processor=processor, 
                                                            device=device, 
                                                            num_hyps=cfg.model.num_hyps, 
                                                            dataloader_idx=idx_dataset, 
                                                            key=key,
                                                            generation_config=model.generation_config,
                                                            cfg=cfg)

        # Reformat output to have batch size one
        outputs = reformat_outputs_bs1(outputs)

        # Batch size 1 to the inference args
        # Save batch size
        inference_args.per_device_eval_batch_size = 1
        
        if os.environ.get("USE_LOCAL_TRANSFORMERS", "false").lower() != "true":
            inference_args.eval_do_concat_batches = False
            inference_args.prediction_loss_only = True

        # New dataloader with original batch
        inference_trainer = Trainer(
                model=model,
                args=inference_args,
                tokenizer=processor,
                data_collator=data_module.batch_processor_fn_mrefs,
                eval_dataset=test_dataset)
        test_dataloader = inference_trainer.get_test_dataloader(test_dataset)
        data_module._test_collate = data_module.batch_processor_fn_mrefs
        outputs = update_output_test_losses(model=model, 
                                            test_dataset=test_dataset, 
                                            processor=processor, 
                                            num_hyps=cfg.model.num_hyps, 
                                            dataloader_idx=idx_dataset, 
                                            key=key,
                                            decoded_predictions=decoded_predictions, 
                                            outputs=outputs, 
                                            data_module=data_module, 
                                            training_args=inference_args)

        if 'datasets_local' in key:
            datasubset = key.split('/')[-1].split('_resample')[0]
        elif 'HDF' in key:
            datasubset = key.split('/HDF/')[-1].split('_resample')[0]
        else:
            datasubset = key.split('/')[-1].split('_resample')[0]

        generation_mode = f"do_sample_{cfg.model.generation_config.do_sample}_temp_{cfg.model.generation_config.temperature}_top_k_{cfg.model.generation_config.top_k}_top_p_{cfg.model.generation_config.top_p}_beam_size_{cfg.model.generation_config.beam_size}_num_groups_{cfg.model.generation_config.num_beam_groups}_diversity_penalty_{cfg.model.generation_config.diversity_penalty}"
        save_path = save_predictions_to_pkl(outputs=outputs, datasubset=datasubset, subrun_dir=cfg.paths.output_dir, generation_mode=generation_mode)
        
        if cfg.pickle_path is None:
            logger.info(f"pickle_path is not specified, using {save_path}")
            cfg.pickle_path = [save_path]
        elif type(cfg.pickle_path) == str:
            logger.info(f"pickle_path is specified, using {cfg.pickle_path}")
            cfg.pickle_path = [cfg.pickle_path]
        else:
            # logger.info(f"pickle_path is specified, using {cfg.pickle_path}")
            logger.info(f"Adding {save_path} to pickle_path")
            cfg.pickle_path.append(save_path)
    return cfg, outputs
