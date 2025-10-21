"""
This script combines fine tuning and inference.
It uses a Hydra configuration (file "run.yaml" under the conf/ directory)
where you can set:
  run:
    do_train: true/false       # Whether to fine tune the model.
    do_inference: true/false   # Whether to run inference after (or instead of) training.
    do_compute_metrics: true/false # Whether to compute metrics after (or instead of) training.
Other model and paths settings are also read from cfg.
"""


import accelerate
from monkey_patch_accelerate import patched_set_module_tensor_to_device
accelerate.utils.modeling.set_module_tensor_to_device = patched_set_module_tensor_to_device
accelerate.utils.set_module_tensor_to_device = patched_set_module_tensor_to_device

import os
import mlflow
import torch
import random
import numpy as np

from dotenv import load_dotenv
load_dotenv()

import sys
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

# Make sure to have your project root set up 
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Set the absolute paths for the following environment variables
relative_path_vars = [
    "LOCAL_CACHE", "SENTENCE_TRANSFORMERS_HOME",
    "AAC_METRICS_CACHE", "COCO_CAPTION_PATH", "CONNETTE_PATH"
]

for var_name in relative_path_vars:
    if var_name in os.environ:
        if not os.path.isabs(os.environ[var_name]):
            os.environ[var_name] = os.path.join(os.environ["PROJECT_ROOT"], os.environ[var_name])

sys.path.append(os.environ["CONNETTE_PATH"])

# Configure the logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class OutputRedirection:
    def __init__(self, stdout_path, stderr_path):
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self.stdout_file = None
        self.stderr_file = None
        self._stdout_original = sys.stdout
        self._stderr_original = sys.stderr

    def __enter__(self):
        self.stdout_file = open(self.stdout_path, 'w')
        self.stderr_file = open(self.stderr_path, 'w')
        sys.stdout = self.stdout_file
        sys.stderr = self.stderr_file
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout_original
        sys.stderr = self._stderr_original
        if self.stdout_file:
            self.stdout_file.close()
        if self.stderr_file:
            self.stderr_file.close()

def setup_transformers_path(cfg):
    """
    Setup the transformers path based on configuration.
    If use_local_transformers is True, prepend the local transformers path to sys.path.
    """
    if hasattr(cfg.model, 'use_local_transformers') and cfg.model.use_local_transformers:
        os.environ["USE_LOCAL_TRANSFORMERS"] = "true"
        # Get the project root directory
        project_root = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
        local_transformers_path = os.path.join(project_root, "transformers")
        
        # Check if local transformers directory exists
        if os.path.exists(local_transformers_path):
            # Insert at the beginning of sys.path to prioritize local version
            sys.path.insert(0, local_transformers_path)
            logger.info(f"Using local transformers from: {local_transformers_path}")
        else:
            logger.warning(f"Local transformers directory not found at {local_transformers_path}, using installed version")
    else:
        os.environ["USE_LOCAL_TRANSFORMERS"] = "false"
        
@hydra.main(version_base=None, config_path=os.path.join(os.environ["PROJECT_ROOT"],'conf'), config_name="run")
def main(cfg: DictConfig):

    # Setup transformers path based on configuration
    setup_transformers_path(cfg)

    from model_setup import setup_model_and_processor, setup_processor, setup_mlflow
    from training import run_training
    from inference import run_inference
    from compute_metrics import run_evaluation

    logger.info(f"Output directory: {cfg.paths.output_dir}")

    if "native_lora_target_modules" in cfg.model and type(cfg.model.native_lora_target_modules) == str:
        import json
        cfg.model.native_lora_target_modules = json.loads(cfg.model.native_lora_target_modules)

    with OutputRedirection(os.path.join(cfg.paths.output_dir, "std.out"),os.path.join(cfg.paths.output_dir, "std.err")):

        # Set seed
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

        # Setup MLflow
        mlflow_run = setup_mlflow(cfg)

        try:
            if cfg.do_train is True or cfg.do_inference is True or cfg.pickle_path is None:
                model, processor = setup_model_and_processor(cfg)
                # Log the model architecture
                logger.info(f"Model architecture: {model}")
            else :
                processor = setup_processor(cfg)

            # ---------------------------------------------------------------------------
            # Fine Tuning Phase
            # ---------------------------------------------------------------------------
            
            training_completed = False
            
            if cfg.do_train:
                # Freeze parameters except for those in the multi_modal_projector
                model = run_training(cfg, model, processor)
                training_completed = True
            else:
                logger.info("Skipping training; using the loaded model for inference.")
                training_completed = True

            # ---------------------------------------------------------------------------
            # Inference Phase
            # ---------------------------------------------------------------------------
            if training_completed:
                if cfg.do_inference:
                    logger.info("Running inference")
                    logger.info(f"Setting per device eval batch size to 1 for inference.")
                    cfg.model.per_device_eval_batch_size = 1
                    cfg, outputs = run_inference(cfg, model, processor)

                if cfg.pickle_path is None and cfg.do_compute_metrics:
                    logger.error("pickle_path is not specified and do_compute_metrics is True")
                    cfg.do_compute_metrics = False

                if cfg.do_compute_metrics:
                    logger.info("Running evaluation / metrics computation")
                    run_evaluation(cfg, processor)

        finally:
            if mlflow_run:
                mlflow.end_run()

        logger.info("Run completed successfully. Logs saved at %s", cfg.paths.output_dir)

        if "out_dir" in cfg : 
            path_to_copy = os.path.join(cfg.paths.root_dir, "cleaned_outputs")
            path_to_copy = os.path.join(path_to_copy, cfg.out_dir)
            if not os.path.exists(path_to_copy):
                os.makedirs(path_to_copy)
            path_to_copy = os.path.join(path_to_copy, cfg.paths.output_dir.split('/')[-1])
            # Output dir management at the end of training
            try:
                os.symlink(cfg.paths.output_dir, path_to_copy)
                logger.info(f"Successfully created symlink: {path_to_copy} -> {cfg.paths.output_dir}")
            except Exception as e:
                logger.error(f"Failed to copy final Hydra outputs: {e}")

if __name__ == "__main__":
    main() 