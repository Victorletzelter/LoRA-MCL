"""This script sets up the model and processor for the training and inference. 
It is used in the main_hydra.py file."""
import sys
import logging
import mlflow

from omegaconf import OmegaConf

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Append paths if needed
import os
sys.path.append(os.environ["CONNETTE_PATH"])
from dataloading import HDFDataModule  # Make sure this is accessible

# Configure the logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from utils import create_dummy_model

if os.environ.get("USE_LOCAL_TRANSFORMERS", "false").lower() == "true":
    from local_transformers import (
        AutoProcessor,
        AutoConfig,
        GenerationConfig
    )
else:
    from transformers import (
        AutoProcessor,
        AutoConfig,
        GenerationConfig
    )

# --------------------------------------------------------------------------
# Common Setup: load processor and model
# ---------------------------------------------------------------------------

def flatten_dict(d, parent_key='', sep='.'):
    """Flatten a nested dictionary for MLflow logging"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def setup_mlflow(cfg):
    """Setup MLflow tracking"""
    if hasattr(cfg, 'mlflow'):
        mlflow_dir = os.path.join(cfg.paths.log_dir, "mlflow")
        if not os.path.exists(mlflow_dir):
            os.makedirs(mlflow_dir)
        if not os.path.exists(os.path.join(mlflow_dir, "mlruns")):
            os.makedirs(os.path.join(mlflow_dir, "mlruns"))
        mlflow.set_tracking_uri("file:" + os.path.join(mlflow_dir, "mlruns"))
        mlflow.set_experiment(cfg.mlflow.experiment_name)
        
        # Log the configuration
        # Start the run 
        run = mlflow.start_run(run_name=cfg.mlflow.run_name)
        
        # Log Hydra config as parameters
        flat_config = OmegaConf.to_container(cfg, resolve=True)
        mlflow.log_params(flatten_dict(flat_config))
        
        # Log the full config as an artifact
        config_path = os.path.join(cfg.paths.output_dir, "config.yaml")
        with open(config_path, "w") as f:
            OmegaConf.save(cfg, f)
        mlflow.log_artifact(config_path)

    return None

def setup_processor(cfg):
    processor = AutoProcessor.from_pretrained(
        cfg.model.pretrained_repo,
        cache_dir=cfg.model.cache_dir,
        low_cpu_mem_usage=True,
        device_map=f"cuda:{int(os.getenv('LOCAL_RANK', 0))}",
    )
    return processor

def setup_model_and_processor(cfg):

    max_length = cfg.model.max_length
    num_hyps = cfg.model.num_hyps
    cache_dir = cfg.model.cache_dir
    is_debug = cfg.model.is_debug
    load_model = cfg.model.load_model  # This flag indicates whether to load a pre-trained model.

    if cfg.model.use_mcl_wrapper:
        from processor_patch import patch_qwen2audio_processor
        patch_qwen2audio_processor()

    # Load the processor from the pretrained repo.
    processor = AutoProcessor.from_pretrained(
        cfg.model.pretrained_repo,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map=f"cuda:{int(os.getenv('LOCAL_RANK', 0))}",
    )

    # Load (or create) the model.
    if load_model:
        logger.info("Loading model for fine tuning / inference.")
        if is_debug:
            # In debug mode, create a simplified dummy model.
            model = create_dummy_model(max_length=max_length, num_hyps=num_hyps, cache_dir=cache_dir, cfg=cfg)
            # Even in debug mode, re-load the processor from the pretrained repo if desired.
            processor = AutoProcessor.from_pretrained(cfg.model.pretrained_repo, cache_dir=cache_dir)
        else:
            if cfg.ckpt_path is None or cfg.model.native_lora_enabled :
                ckpt_path_considered = cfg.model.pretrained_repo
            else :
                ckpt_path_considered = cfg.ckpt_path
            logger.info(f"Loading model from {ckpt_path_considered}")
            # Load using AutoConfig and from_pretrained.
            config_model = AutoConfig.from_pretrained(
                ckpt_path_considered,
                cache_dir=cache_dir,
                low_cpu_mem_usage=True,
                device_map=f"cuda:{int(os.getenv('LOCAL_RANK', 0))}",
                local_files_only=False,
                num_beams=cfg.model.get("generation_config", {}).get("beam_size", None),
                num_return_sequences=cfg.model.get("generation_config", {}).get("num_return_sequences", None),
                do_sample=cfg.model.get("generation_config", {}).get("do_sample", None),
                temperature=cfg.model.get("generation_config", {}).get("temperature", None),
                top_k=cfg.model.get("generation_config", {}).get("top_k", None),
                top_p=cfg.model.get("generation_config", {}).get("top_p", None),
                diversity_penalty=cfg.model.get("generation_config", {}).get("diversity_penalty", None),
                num_beam_groups=cfg.model.get("generation_config", {}).get("num_beam_groups", None),
                max_new_tokens=cfg.model.get("generation_config", {}).get("max_new_tokens", None),
                penalty_alpha=cfg.model.get("generation_config", {}).get("penalty_alpha", None),
                typical_p=cfg.model.get("generation_config", {}).get("typical_p", None))
            
            config_model.num_hyps = num_hyps
            config_model.wta_training_mode = cfg.model.wta_training_mode
            config_model.wta_params_epsilon = cfg.model.wta_params_epsilon
            config_model.audio_config.max_source_positions = int(1500 * max_length / (30 * 16000))
            config_model.wta_params_ini_temp = cfg.model.wta_params_ini_temp
            config_model.wta_params_fin_temp = cfg.model.wta_params_fin_temp
            config_model.wta_params_decay_rate = cfg.model.wta_params_decay_rate
            config_model.wta_params_schedule_mode = cfg.model.wta_params_schedule_mode
            config_model.native_group_lora_enabled = cfg.model.native_group_lora_enabled
            config_model.tta_enabled = cfg.model.tta_enabled
            config_model.tta_mode = cfg.model.tta_mode

            config_model.sparse_moe_enabled = cfg.model.sparse_moe_enabled
            config_model.router_aux_loss_coef = cfg.model.router_aux_loss_coef
            config_model.num_experts = cfg.model.num_experts
            config_model.top_k_sparse_moe = cfg.model.top_k_sparse_moe

            import torch
            if cfg.model.native_lora_enabled and cfg.model.native_group_lora_enabled is False and cfg.model.use_mcl_wrapper is False:
                from local_transformers import Qwen2AudioForConditionalGeneration_MH_Lora
                model = Qwen2AudioForConditionalGeneration_MH_Lora.from_pretrained(
                    ckpt_path_considered,
                    cache_dir=cache_dir,
                    config=config_model,
                    device_map=f"cuda:{int(os.getenv('LOCAL_RANK', 0))}",
                    torch_dtype=torch.bfloat16,
                )
            elif cfg.model.native_group_lora_enabled is True and cfg.model.use_mcl_wrapper is False:
                from local_transformers import Qwen2AudioForConditionalGeneration_MH_Group_Lora
                model = Qwen2AudioForConditionalGeneration_MH_Group_Lora.from_pretrained(
                    ckpt_path_considered,
                    cache_dir=cache_dir,
                    config=config_model,
                    device_map=f"cuda:{int(os.getenv('LOCAL_RANK', 0))}",
                    torch_dtype=torch.bfloat16,
                )
            elif cfg.model.use_mcl_wrapper is False: # If mcl wrapper is not used, one of the two above must be used.
                raise ValueError("Invalid model configuration")
    else:
        logger.info("Creating a new model instance (not loading pretrained checkpoint).")
        model = create_dummy_model(max_length=max_length, num_hyps=num_hyps, cache_dir=cache_dir, cfg=cfg)

    ### Generation config override
    # If cfg.model.generation_config is an OmegaConf instance, convert it to a dictionary.
    gen_config_dict = OmegaConf.to_container(cfg.model.generation_config)

    # Now create a GenerationConfig instance with the desired parameters.
    gen_config = GenerationConfig(
        num_beams=gen_config_dict.get("beam_size"),
        num_return_sequences=gen_config_dict.get("num_return_sequences"),
        do_sample=gen_config_dict.get("do_sample"),
        temperature=gen_config_dict.get("temperature"),
        top_k=gen_config_dict.get("top_k"),
        top_p=gen_config_dict.get("top_p"),
        diversity_penalty=gen_config_dict.get("diversity_penalty"),
        num_beam_groups=gen_config_dict.get("num_beam_groups"),
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        repetition_penalty=gen_config_dict.get("repetition_penalty"),
        chat_format="chatml",
        output_scores=gen_config_dict.get("output_scores"),
        return_dict_in_generate=gen_config_dict.get("return_dict_in_generate"),
        max_new_tokens=gen_config_dict.get("max_new_tokens", None),
        penalty_alpha=gen_config_dict.get("penalty_alpha", None),
        typical_p=gen_config_dict.get("typical_p", None),
    )
    if cfg.model.use_mcl_wrapper is False:
        model.generation_config = gen_config

    if cfg.model.native_lora_enabled:
        logger.info("Using native LoRA")
        if os.environ.get("USE_LOCAL_TRANSFORMERS", "false").lower() == "true":
            from local_peft import LoraConfig, get_peft_model, PeftModel
        else:
            from peft import LoraConfig, get_peft_model, PeftModel
        peft_config = LoraConfig(
                r=cfg.model.native_lora_r,
                lora_alpha=cfg.model.native_lora_alpha,
                lora_dropout=cfg.model.native_lora_dropout,
                target_modules=cfg.model.native_lora_target_modules,
                bias=cfg.model.native_lora_bias,
                exclude_modules=cfg.model.native_lora_exclude_modules,
            )
        logger.info("Peft config created")
        peft_config.use_group_lora = cfg.model.native_group_lora_enabled
        peft_config.use_moe_lora = cfg.model.use_moe_lora
        peft_config.num_experts = cfg.model.num_experts
        peft_config.init_zero_router = cfg.model.init_zero_router
        peft_config.stochastic_router = cfg.model.stochastic_router
        peft_config.expert_specific = cfg.model.expert_specific
        peft_config.fast_version = cfg.model.fast_version
        peft_config.sparse_moe_enabled = cfg.model.sparse_moe_enabled
        peft_config.top_k_sparse_moe = cfg.model.top_k_sparse_moe

        # Get the LOCAL_RANK
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # # Ensure distributed environment is properly initialized
        if os.environ.get("IS_DISTRIBUTED", "false") == 'true':
            if local_rank != -1 and not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl")
                torch.cuda.set_device(local_rank)
                # Add a barrier to ensure all processes are synchronized
                torch.distributed.barrier()
        
        if cfg.model.use_mcl_wrapper:
            import sys
            sys.path.append(os.path.join(os.environ["PROJECT_ROOT"], ".."))
            from mcl_wrapper import get_peft_mcl, patch_peft_for_mcl
            from peft import LoraConfig
            patch_peft_for_mcl(enable=True) # Patch the updates to the peft library
            mcl_params = { # LoRA-MCL related parameters.
            'num_hyps': cfg.model.num_hyps, # Number of hypotheses (K)
            'wta_training_mode' : cfg.model.wta_training_mode, # "wta", "relaxed-wta" or "annealed-wta"
            'use_group_lora' : cfg.model.native_group_lora_enabled, # whether to use the Group LoRA implementation (that accelerates training)
            'wta_params_epsilon' : cfg.model.wta_params_epsilon, # \\varepsilon when  wta_training_mode == 'relaxed-wta'
            # Parameters only used if 'wta_training_mode'='annealed-wta':
            'wta_params_ini_temp': cfg.model.wta_params_ini_temp, # initial temperature when 'wta_training_mode'='annealed-wta'
            'wta_params_fin_temp': cfg.model.wta_params_fin_temp, # final temperature (i.e., temperature from which the mode is switched back to wta) when 'wta_training_mode'='annealed-wta'
            'wta_params_decay_rate': cfg.model.wta_params_decay_rate, # decay rate rho, where temperature(t) = temperature(0)*rho**{t} 
            'wta_params_schedule_mode': cfg.model.wta_params_schedule_mode, # type of decay for the temperature, either "global step" if t represents the training step, and "epoch_number" if t is the epoch number
            }
            model_kwargs = {
                "max_source_positions": int(1500 * max_length / (30 * 16000)),
            }
            if cfg.ckpt_path is None :
                model_name_or_path = cfg.model.pretrained_repo
            else :
                model_name_or_path = cfg.ckpt_path
            # from monkey_patch_accelerate import patched_set_module_tensor_to_device
            # import accelerate          
            # accelerate.utils.modeling.set_module_tensor_to_device = patched_set_module_tensor_to_device
            # accelerate.utils.set_module_tensor_to_device = patched_set_module_tensor_to_device      
            model = get_peft_mcl(
                pretrained_repo=cfg.model.pretrained_repo,
                model_name_or_path=model_name_or_path, # Any HuggingFace model
                lora_config=peft_config, # LoRA config
                generation_config=gen_config,
                **mcl_params,
                **model_kwargs
            )
            logger.info("MCL model created")
        else:
            model = get_peft_model(model, peft_config, adapter_name="lora0")
        logger.info("Peft model created")

        # Add remaining adapters
        for i in range(1, num_hyps):
            if cfg.model.use_mcl_wrapper is False and cfg.model.native_group_lora_enabled is False:
                model.add_adapter(f"lora{i}", peft_config)
        if cfg.model.native_group_lora_enabled is True:
            all_adapters = [f"lora{i}" for i in range(cfg.model.num_hyps)]
            for module in model.modules():
                if hasattr(module, 'set_adapter') and hasattr(module, 'max_hyps'):
                    # This is a GroupLinear layer, enabled all adapters for training
                    module.set_adapter(all_adapters)

        logger.info("Remaining adapters added")
        if cfg.ckpt_path is not None and cfg.model.native_lora_enabled is True and cfg.model.use_mcl_wrapper is False:
            for i in range(num_hyps):
                model.load_adapter(os.path.join(cfg.ckpt_path, f"lora{i}"), adapter_name=f"lora{i}", is_trainable=cfg.do_train, torch_device=f"cuda:{int(os.getenv('LOCAL_RANK', 0))}")

        logger.info(f"Trainable parameters: {model.print_trainable_parameters()}")

    return model, processor
