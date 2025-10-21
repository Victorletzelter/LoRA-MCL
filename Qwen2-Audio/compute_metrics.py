"""This script is used to compute the metrics. 
It is adapted from the [Conette library](https://github.com/Labbeti/conette-audio-captioning/),
[AAC-Metrics](https://github.com/Labbeti/aac-metrics),
as well as the [COCO-Caption](https://github.com/tylin/coco-caption) library.
Note that adaptations of the AAC-Metrics library are done in the Qwen2-Audio/metrics/aac_metrics_custom directory. 
This was for oracle metrics computation, because the standard captioning metrics doesn't handle multiple hypotheses."""

import os
import os.path as osp
import sys
import logging
import numpy as np
import rootutils
import mlflow
import csv
import tempfile
import pickle
import torch
import yaml
from aac_metrics.utils.checks import is_mono_sents, is_mult_sents
import tempfile
from torch.utils.tensorboard import SummaryWriter
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

sys.path.append(os.environ["CONNETTE_PATH"])
from dataloading import HDFDataModule  # Make sure this is accessible
from typing import Any, Optional, Union
from aac_metrics.utils.checks import is_mono_sents, is_mult_sents
from aac_metrics.utils.collections import flat_list, unflat_list
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torchoutil.utils.collections import all_eq
from metrics.metrics.classes.mh_all_metrics import MH_AllMetrics
from metrics.tokenization.aac_tokenizer import AACTokenizer
from metrics.utils.custom_logger import CustomTensorboardLogger
from metrics.utils.dcase import export_to_dcase_task6a_csv
import sys
from pathlib import Path
sys.path.append(os.path.join(os.environ['PROJECT_ROOT'], 'metrics', 'metrics', 'classes'))
from eval_metrics import write_json, write_yaml

# Configure the logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_evaluation(cfg, processor):
    
    pylog = logging.getLogger(__name__)

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
        use_mcl_wrapper=cfg.model.use_mcl_wrapper
        )

    data_module.setup("test")

    class MH_AACEvaluator:
        """Callback which stores candidates and references during testing to produce AAC scores.

        Include metrics : BLEU1, BLEU2, BLEU3, BLEU4, METEOR, ROUGE-L, CIDEr, SPICE, SPIDEr.
        """

        CANDS_PREFIX = "cands"
        MREFS_KEY = "mrefs"

        def __init__(
            self,
            subrun_path: Optional[str],
            test_tokenizer: AACTokenizer,
            cache_path: str = "~/.cache",
            java_path: str = "java",
            tmp_path: str = tempfile.gettempdir(),
            ckpt_name: str = "unk",
            verbose: int = 1,
            debug: bool = False,
            save_to_csv: bool = True,
            save_dcase_csv_file: bool = False,
            metric_device: Union[str, torch.device, None] = None,
            cpus: Optional[int] = None,
            save_predictions: bool = False,
            load_predictions: list[str] = None,
            cfg: dict = None
        ) -> None:
            if subrun_path is not None:
                subrun_path = osp.expandvars(subrun_path)

            self._subrun_dir = subrun_path
            self._test_tokenizer = test_tokenizer
            self._cache_path = cache_path
            self._java_path = java_path
            self._tmp_path = tmp_path
            self._model_name = ckpt_name
            self._verbose = verbose
            self._debug = debug
            self._save_to_csv = save_to_csv
            self._save_dcase_csv_file = save_dcase_csv_file
            self._metric_device = metric_device
            self._cpus = cpus
            self._all_outputs: dict[int, dict[str, Any]] = {}
            self._generation_config = None
            self._save_predictions = save_predictions
            self._load_predictions = load_predictions
            self._cfg = cfg
            self._excluded_datasubsets_metrics = None
            self._all_metrics = None

        @property
        def prefix(self) -> str:
            """Dynamically determine the prefix based on current logits_hypotheses setting."""
            return "hypothesis"
        
        def save_predictions_to_file(self, outputs: dict, datasubset: str) -> None:
            """Save predictions to a file."""
            if self._subrun_dir is None:
                pylog.warning("Cannot save predictions: no subrun directory specified")
                return

            save_path = osp.join(self._subrun_dir, f'predictions_{datasubset}.pkl')
            
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
                
            if self._verbose >= 1:
                pylog.info(f"Saved predictions to {save_path}")

        def change_format(self, outputs: dict) -> dict:
            """Change the format of the outputs."""
            new_outputs = {}
            for dataloader_idx, hypotheses in outputs.items(): # Loop over dataloader indices
                new_outputs[dataloader_idx] = {}
                for hyp_key, hyp_data in hypotheses.items(): # Loop over hypothesis keys
                    new_outputs[dataloader_idx][hyp_key] = {}
                    for k, v in hyp_data.items(): # Loop over hypothesis data point indexes 0, 1, 2, ...
                        for i, j in v.items(): # Loop over named keys: ['losses', 'preds', 'lprobs', 'cands', 'mrefs', 'fname', 'index', 'dataset', 'subset
                            if i not in new_outputs[dataloader_idx][hyp_key]:
                                new_outputs[dataloader_idx][hyp_key][i] = []
                            if len(j) == 1:
                                j = j[0]
                                if type(j) == str and '\n' in j:
                                    j = j.replace('\n', '') # Remove the new line characters from the string if present
                            new_outputs[dataloader_idx][hyp_key][i].append(j)
            return new_outputs


        def load_predictions_from_file(self, datasubset: str) -> Optional[dict]:
            """Load predictions from a file."""
            if self._load_predictions is None:
                return None

            # load_path = osp.join(self._load_predictions, f'predictions_{datasubset}.pkl')

            if type(self._load_predictions) == str:
                self._load_predictions = [self._load_predictions]
            
            for i, path in enumerate(self._load_predictions):
                if not osp.exists(path):
                    pylog.warning(f"Cannot load predictions: file not found at {path}")
                    return None

            # try:
            load_path = None
            for path in self._load_predictions:
                if datasubset in path:
                    load_path = path
                    break

            if load_path is None:
                pylog.warning(f"Cannot load predictions: file not found at {self._load_predictions}")
                return None

            with open(load_path, 'rb') as f:
                outputs = pickle.load(f)
                outputs = self.change_format(outputs)
                if self._verbose >= 1:
                    pylog.info(f"Loaded predictions from {load_path}")
                return outputs

        def on_test_start(self, test_loaders, generation_config, device, num_hypotheses):

            self._generation_config = generation_config

            if self._all_metrics is None:
                if self._metric_device is not None:
                    device = self._metric_device
                else:
                    device = device
        
                settings = {
                    "model": {
                        "lm": {
                            "generation": {
                                **generation_config,
                            }
                        }
                    },
                    "evaluation": {
                        "batch_size": 1,
                        "device": "cuda" if torch.cuda.is_available() else "cpu"
                    },
                    "paths": {
                        "cache_path": os.environ["AAC_METRICS_CACHE"] if 'AAC_METRICS_CACHE' in os.environ else None,
                    }
                    }

                from types import SimpleNamespace
                generation_config = SimpleNamespace(**generation_config)

                self._all_metrics = MH_AllMetrics(
                    is_tokenized=False,
                    num_hypotheses=num_hypotheses,
                    num_return_sequences=generation_config.num_return_sequences,
                    tokenizer=None,
                    settings=settings
                )

                if self._verbose >= 2:
                    pylog.debug(f"{len(self._all_metrics)} metrics has been initialized.")

                    assert isinstance(test_loaders, list) and all(
                        isinstance(loader, DataLoader) for loader in test_loaders
                    )
                    sizes = tuple(map(len, test_loaders))
                    pylog.debug(f"Test loader sizes: {sizes}")

        def on_test_epoch_start(self, test_dataloaders) -> None:
            self._all_outputs = {}
            if self._verbose >= 1:
                pylog.debug(f"Starting TEST epoch with model_name='{self._model_name}'")

            # If loading predictions, do it at the start of the epoch
            if self._load_predictions is not None:
                if self._verbose >= 1:
                    pylog.info(f"Loading predictions from {self._load_predictions}")

                # Handle all test dataloaders
                for dataloader_idx, dataloader in enumerate(test_dataloaders):
                    # Get first batch to determine datasubset
                    batch = next(iter(dataloader))
                    dummy_outputs = {
                        "dataset": batch["dataset"],
                        "subset": batch["subset"]
                    }
                    datasubset = _get_datasubset_name(dummy_outputs)
                    loaded_outputs = self.load_predictions_from_file(datasubset)
                    if loaded_outputs is not None:
                        # Store outputs for this dataloader
                        self._all_outputs[dataloader_idx] = loaded_outputs[dataloader_idx]
                        if self._verbose >= 1:
                            pylog.info(f"Successfully loaded predictions for dataloader {dataloader_idx} ({datasubset})")
                    else:
                        pylog.warning(f"Failed to load predictions for {datasubset} from {self._load_predictions}")
                        # raise ValueError(f"Failed to load predictions for {datasubset} from {self._load_predictions}")

        def on_test_epoch_end(self, prefix, num_hypotheses, tokenizer, generation_config) -> None:

            if self._save_predictions is True:
                for dataloader_idx, outputs in self._all_outputs.items():
                    datasubset = _get_datasubset_name(outputs[f"{prefix}_0"])
                    self.save_predictions_to_file(self._all_outputs, datasubset)

            # If loading predictions is enabled, use loaded predictions instead of computed ones
            if self._load_predictions is not None:
                all_outputs_subsets = {}
                for dataloader_idx, outputs in self._all_outputs.items():
                    datasubset = _get_datasubset_name(outputs[f"{prefix}_0"])
                    loaded_outputs = self.load_predictions_from_file(datasubset)
                    if loaded_outputs is not None:
                        all_outputs_subsets[datasubset] = loaded_outputs[list(loaded_outputs.keys())[0]]
                        if self._verbose >= 1:
                            pylog.info(f"Using loaded predictions for {datasubset}")
            else:
                all_outputs_subsets = self._all_outputs

            datasubsets = []
            metrics_all_subsets = {}
            for outputs in all_outputs_subsets.values(): # Loop over dataloaders
                datasubset = _get_datasubset_name(outputs[f"{prefix}_0"])
                counter = datasubsets.count(datasubset)
                if counter > 0:
                    old_datasubset = datasubset
                    datasubset = f"{datasubset}_{counter+1}"
                    pylog.error(
                        f"Found duplicated subset '{old_datasubset}'. Renaming to '{datasubset}'."
                    )
                    assert datasubset not in datasubsets
                datasubsets.append(datasubset)

                if num_hypotheses == 1 :
                    num_predictions = generation_config.num_return_sequences
                else :
                    num_predictions = num_hypotheses

                for hypothesis_idx in range(num_predictions):
                    # Tokenize candidates and references
                    sents_keys = [
                        key
                        for key in outputs[f"{prefix}_{hypothesis_idx}"].keys()
                        if key.startswith(self.CANDS_PREFIX) or key == self.MREFS_KEY
                    ]

                    if self._verbose >= 2:
                        pylog.debug(
                            f"Process sentences with tokenizer... ({tuple(sents_keys)=}"
                        )

                    for key in sents_keys:
                        raw_sents = outputs[f"{prefix}_{hypothesis_idx}"][key]

                        if is_mono_sents(raw_sents):
                            sents = self._test_tokenizer.tokenize_batch(sentences=raw_sents)
                            sents = self._test_tokenizer.detokenize_batch(sentences=sents)

                        elif is_mult_sents(raw_sents):
                            flat_raw_sents, sizes = flat_list(raw_sents)
                            flat_sents = self._test_tokenizer.tokenize_batch(sentences=flat_raw_sents)
                            flat_sents = self._test_tokenizer.detokenize_batch(sentences=flat_sents)
                            sents = unflat_list(flat_sents, sizes)

                        else:
                            raise TypeError(f"Cannot detect sentences type. (with {key=})")

                        outputs[f"{prefix}_{hypothesis_idx}"][key] = sents

                if self._verbose >= 2:
                    pylog.debug(f"Sentences processed. ({tuple(sents_keys)=})")

                with torch.inference_mode():
                    total_metrics, all_gt_captions, all_pred_captions = self._compute_metrics(
                        outputs=outputs, datasubset=datasubset, prefix=prefix, num_hypotheses=num_hypotheses, num_return_sequences=generation_config.num_return_sequences
                    )

                # Additional metrics
                outputs['filenames'] = outputs[f"{prefix}_0"]['fname']
                outputs['datasubset'] = datasubset
                audio_paths = None
                eos_token = tokenizer.eos_token

                # preds is a tensor [tensor([   1, 1122, 1122, 1122,  185, 2144,  270,  373, 4377, 4229,  185, 2144,
                # 3734,  414, 2092, 2144,  278,  414,  373,  414,  278,  414,  278,  414,
                #  373,  414,  278,  414,  278,  414]), ...] 
                # of length num example, with each example of a given length. The length is the length of the sequence, unless the eos token is at the end of the sequence, in which case it can be shorter.
                # Note a the first token is the bos token, so the length is the length of the sequence minus 1.
                hypotheses_key = ['{}_{}'.format(prefix, idx_hyp) for idx_hyp in range(num_predictions)]
                sequence_lengths = [[] for _ in range(len(hypotheses_key))]
                for idx_hyp, key in enumerate(hypotheses_key):
                    for e in outputs[key]['preds']:
                        # Convert to list if needed
                        e_list = e.tolist() if isinstance(e, torch.Tensor) else e
                        # Find length (which is consitent with the length compacted as per the hf library, see the length penalty param)
                        if eos_token in e_list:
                            sequence_lengths[idx_hyp].append(e_list.index(eos_token)+1)
                        else:
                            sequence_lengths[idx_hyp].append(len(e_list))

                sequence_lengths = torch.tensor(sequence_lengths) # [num_examples, num_hypotheses]

                # Oracle mean NLL (single sample used if num_hypotheses = 1 and num_return_sequences > 1)
                if generation_config._compute_oracle_mean_nll is True:
                    # For each example i, oracle_mean_nll[i] = - max_k 1/T(k) * sum_t=1^T log(p_k(y_t^i | y_<t^i, theta))
                    # See the perplexity definition in https://openreview.net/pdf?id=QKRLH57ATT page 3 and apply log. 
                    oracle_mean_nll, full_scores, best_indexes = self._compute_oracle_mean_nll(outputs=outputs, datasubset=datasubset, audio_paths=audio_paths, prefix=prefix, num_hypotheses=num_hypotheses,num_return_sequences=generation_config.num_return_sequences)
                    total_metrics['oracle_mean_nll'] = oracle_mean_nll

                    for k in best_indexes.keys():
                        total_metrics[k] = best_indexes[k]

                    if generation_config.return_full_scores is True:
                        total_metrics['oracle_mean_nll_full_scores'] = full_scores

                # Write the outputs to disk
                self.mh_write_outputs(output_dir=self._subrun_dir, datasubset=datasubset, metrics=total_metrics, all_gt_captions=all_gt_captions, all_pred_captions=all_pred_captions, generation_mode='oracle', num_hypotheses=num_hypotheses,num_return_sequences=generation_config.num_return_sequences)                    

                # return total_metrics
                metrics_all_subsets[datasubset] = total_metrics

            return metrics_all_subsets

        def mh_write_outputs(self, output_dir, datasubset, metrics, all_gt_captions, all_pred_captions, generation_mode, num_hypotheses, num_return_sequences):

            if num_hypotheses == 1 :
                num_predictions = num_return_sequences
            else :
                num_predictions = num_hypotheses

            # Split metrics into full_scores and regular metrics
            full_scores_metrics = {k: v for k, v in metrics.items() if 'full_scores' in k}
            regular_metrics = {k: v for k, v in metrics.items() if 'full_scores' not in k}

            # Extract scores key from metrics bleu_1, bleu_2, bleu_3, bleu_4, meteor, rouge_l, cider, spice, spider
            list_scores_keys = ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'meteor', 'rouge_l', 'cider', 'spice', 'spider']
            for key in list_scores_keys:
                if key in regular_metrics:
                    full_scores_metrics[key] = {}
                    if 'score' in regular_metrics[key]:
                        full_scores_metrics[key]['score'] = regular_metrics[key]['score']
                    if 'scores' in regular_metrics[key]:
                        full_scores_metrics[key]['scores'] = regular_metrics[key].pop('scores')
            
            # Write regular metrics
            write_json(regular_metrics, Path(output_dir).joinpath('metrics_coco_'+generation_mode+'_'+datasubset+'.json'))
            write_yaml(regular_metrics, Path(output_dir).joinpath('metrics_coco_'+generation_mode+'_'+datasubset+'.yaml'))
            
            # Write full scores metrics separately if they exist
            if full_scores_metrics:
                write_json(full_scores_metrics, Path(output_dir).joinpath('metrics_coco_full_scores_'+generation_mode+'_'+datasubset+'.json'))
                write_yaml(full_scores_metrics, Path(output_dir).joinpath('metrics_coco_full_scores_'+generation_mode+'_'+datasubset+'.yaml'))

            # Write outputs to disk
            # write_json(metrics, Path(output_dir).joinpath('metrics_coco_'+generation_mode+'_'+datasubset+'.json'))
            # write_yaml(metrics, Path(output_dir).joinpath('metrics_coco_'+generation_mode+'_'+datasubset+'.yaml'))
            with open(Path(output_dir).joinpath('generated_captions_'+generation_mode+'_'+datasubset+'.txt'), 'w') as f:
                for i_file in range(len(all_gt_captions)):
                    f.write('----- File {} -----\n'.format(i_file))
                    f.write('GT:   '+'\n')
                    for i_gt in range(len(all_gt_captions[i_file])):
                        f.write('      '+all_gt_captions[i_file][i_gt]+'\n')
                    for hypothesis_idx in range(num_predictions) :
                        f.write('Hypothesis '+str(hypothesis_idx)+'\n')
                        f.write('Pred: '+all_pred_captions[f'{hypothesis_idx}'][i_file]+'\n')

        # AACEvaluator methods
        def set_model_name(self, model_name: str) -> None:
            self._model_name = model_name

        def reformat_outputs(self, outputs, datasubset, prefix, num_hypotheses, num_return_sequences) :
            # outputs dic hypothesis_0, ..., in each: losses, preds, lprobs, mpreds, mlprobs, cands,..
            new_outputs = {}
            new_outputs['GT'] = outputs[f"{prefix}_0"]['mrefs']
            new_outputs['decoded_predictions'] = {}
            new_outputs['filenames'] = outputs[f"{prefix}_0"]['fname']
            new_outputs['datasubset'] = datasubset

            if num_hypotheses == 1 :
                num_predictions = num_return_sequences
            else :
                num_predictions = num_hypotheses

            for hypothesis_idx in range(num_predictions):
                new_outputs['decoded_predictions'][f'{hypothesis_idx}'] = outputs[f"{prefix}_{hypothesis_idx}"]['cands']

            return new_outputs
            
        def _compute_metrics(
            self,
            outputs: dict[str, list],
            datasubset: str,
            prefix: str,
            num_hypotheses: int,
            num_return_sequences: int,
        ) -> tuple[dict[str, float], dict[str, list[float]]]:
        # outputs dic hypothesis_0, ..., in each: losses, preds, lprobs, mpreds, mlprobs, cands,..

            outputs = self.reformat_outputs(outputs, datasubset=datasubset, prefix=prefix, num_hypotheses=num_hypotheses, num_return_sequences=num_return_sequences)

            total_metrics, all_gt_captions, all_pred_captions = self._all_metrics(outputs)

            return total_metrics, all_gt_captions, all_pred_captions

        def _log_global_scores(
            self,
            corpus_scores: dict[str, float],
            datasubset: str,
            pl_module: LightningModule,
        ) -> None:
            global_scores_with_datasubset = {
                f"{datasubset}/{key}": score for key, score in corpus_scores.items()
            }
            for pl_logger in pl_module.loggers:
                if isinstance(pl_logger, CustomTensorboardLogger):
                    pl_logger.log_hyperparams(
                        params={}, metrics=global_scores_with_datasubset
                    )
                    pl_logger.update_files()

        def _print_example(
            self,
            outputs: dict[str, list],
            datasubset: str,
            pl_module: LightningModule,
            sents_scores: dict[str, list[float]],
        ) -> None:
            assert self._test_tokenizer is not None
            n_outputs = len(outputs["fname"])
            indexes = torch.randint(0, n_outputs, (1,)).tolist()

            pylog.info(
                f"Show {len(indexes)} example(s) with model_name={self._model_name} : "
            )

            for idx in indexes:
                fname = outputs["fname"][idx]
                dset_index = outputs["index"][idx]
                candidates = {
                    key: candidates_sents[idx]
                    for key, candidates_sents in outputs.items()
                    if key.startswith(self.CANDS_PREFIX)
                }
                mult_references = outputs[self.MREFS_KEY][idx]

                lines = "-" * 10
                width = 128

                local_main_metrics = {
                    k: v[idx] for k, v in sents_scores.items() if "spider" in k
                }

                infos = {
                    "datasubset": datasubset,
                    "index": dset_index,
                    "fname": fname,
                } | local_main_metrics

                pylog.info(
                    f"\n"
                    f"{lines}\nInfos\n{lines}\n{yaml.dump(infos, width=width, sort_keys=False)}"
                    f"{lines}\nCandidates\n{lines}\n{yaml.dump(candidates, width=width, sort_keys=False)}"
                    f"{lines}\nReferences\n{lines}\n{yaml.dump(mult_references, width=width, sort_keys=False)}"
                )

                # Log examples
                loggers = pl_module.loggers
                for logger in loggers:
                    if isinstance(logger, TensorBoardLogger):
                        prefix = logger.name
                        logger.experiment.add_text(
                            f"{prefix}/{datasubset}_cands_{dset_index}",
                            yaml.dump(candidates, sort_keys=False),
                        )
                        logger.experiment.add_text(
                            f"{prefix}/{datasubset}_mrefs_{dset_index}",
                            yaml.dump(mult_references, sort_keys=False),
                        )

        def _save_outputs_to_csv(
            self,
            dpath: str,
            datasubset: str,
            outs: dict[str, list],
            sents_scores: dict[str, list[float]],
        ) -> None:
            # Sanity check
            lens = list(map(len, outs.values())) + list(map(len, sents_scores.values()))
            assert all_eq(lens), f"{lens=}"

            n_items = lens[0]
            csv_fname = f"{self._model_name}_outputs_{datasubset}.csv"
            csv_fpath = osp.join(dpath, csv_fname)

            def process(key: str, value: Any) -> Any:
                if isinstance(value, Tensor):
                    return value.tolist()
                else:
                    return value

            csv_all_values = outs | sents_scores

            with open(csv_fpath, "w") as file:
                keys = list(csv_all_values.keys())
                writer = csv.DictWriter(file, fieldnames=keys)
                writer.writeheader()

                for i in range(n_items):
                    row = {key: values[i] for key, values in csv_all_values.items()}
                    row = {key: process(key, value) for key, value in row.items()}
                    writer.writerow(row)

            if self._save_dcase_csv_file:
                fnames = outs["fname"]
                mcands = {k: v for k, v in outs.items() if k.startswith(self.CANDS_PREFIX)}

                dcase_dpath = osp.join(dpath, "dcase")
                os.makedirs(dcase_dpath, exist_ok=True)

                for cands_name, cands in mcands.items():
                    if len(mcands) == 1:
                        dcase_fname = (
                            f"submission_output_{self._model_name}_{datasubset}.csv"
                        )
                    else:
                        dcase_fname = f"submission_output_{self._model_name}_{datasubset}_{cands_name}.csv"

                    dcase_fpath = osp.join(dcase_dpath, dcase_fname)
                    export_to_dcase_task6a_csv(dcase_fpath, fnames, cands)

        def _compute_oracle_mean_nll(self, outputs: dict[str, list], datasubset: str, audio_paths: list[str], prefix: str, num_hypotheses: int, num_return_sequences: int) -> tuple[float, dict]:
            """Compute the oracle mean negative log-likelihood across hypotheses.

            For each example i, oracle_mean_nll[i] = - max_k 1/T(k) * sum_t=1^T log(p_k(y_t^i | y_<t^i, theta))
            where k indexes the hypotheses and T is the sequence length.
            This metric measures the quality of the best hypothesis for each example.

            Reference:
                "EFFICIENT AND EFFECTIVE UNCERTAINTY QUANTIFICATION FOR LLMS"
                https://openreview.net/pdf?id=QKRLH57ATT (Equation 3)

            Args:
                outputs: Dictionary containing model outputs for each hypothesis
                datasubset: Name of the current data subset (e.g., 'clotho_val')
                audio_paths: List of paths to audio files

            Returns:
                tuple: (oracle_mean_nll, full_scores)
                    - oracle_mean_nll: Average oracle NLL across examples
                    - full_scores: Dictionary mapping audio paths to per-example scores
            """
            lprobs_list = []
            num_predictions = num_hypotheses # Otherwise, we compute an oracle NLL.

            # Loop over hypotheses_0, ..., hypotheses_N
            for hypothesis_idx in range(num_predictions):
                hypothesis_key = f"{prefix}_{hypothesis_idx}"
                lprobs = outputs[hypothesis_key]['losses'] # List of tensors (of length num_examples)
                # stack them to get a tensor of shape (num_examples, num_captions_per_example)
                lprobs = torch.stack(lprobs) # (num_examples, num_captions_per_example)
                # lprobs_list.append(lprobs.mean(dim=1)) # Average over the captions per example
                lprobs_list.append(lprobs)

            stacked_lprobs = torch.stack(lprobs_list) # shape: (num_hypotheses, num_examples, num_captions_per_example)
            num_captions_per_example = stacked_lprobs.shape[2]
            oracle_nll = stacked_lprobs.min(dim=0).values.mean()
            oracle_nll_indexes = stacked_lprobs.argmin(dim=0) # shape: (num_examples, num_captions_per_example)

            num_examples = oracle_nll_indexes.shape[0]
            full_scores = {}
            best_indexes = {f'nt_hypothesis_{k}'.format(k): 0 for k in range(num_hypotheses)}
            
            for i in range(num_examples):
                for caption_idx in range(num_captions_per_example):
                    if type(oracle_nll_indexes[i, caption_idx]) == torch.Tensor:
                        idx = oracle_nll_indexes[i, caption_idx].item()
                    else:
                        idx = oracle_nll_indexes[i, caption_idx]
                    best_indexes[f'nt_hypothesis_{idx}'] += 1/num_examples

            # for i, audio_path in enumerate(audio_paths):
            #     audio_path = audio_path.split('/')[-1]
            #     full_scores[audio_path] = {}
            #     for hypothesis_idx in range(num_predictions):
            #         for caption_idx in range(num_captions_per_example):
            #             full_scores[audio_path]['{}_{}_{}'.format(prefix, hypothesis_idx, caption_idx)] = stacked_lprobs[hypothesis_idx, i, caption_idx].cpu().item()

            if self._verbose >= 1:
                pylog.info(f"Oracle NLL for {datasubset}: {oracle_nll:.4f}")

            return oracle_nll, full_scores, best_indexes

    def _get_datasubset_name(outputs: dict[str, Any]) -> str:
        if 'dataset' in outputs :
            datanames = list(sorted(set(map(str.lower, outputs["dataset"]))))
        else :
            datanames = list(sorted(set(map(str.lower, outputs[0]["dataset"]))))
        if 'subset' in outputs :
            subsets = list(sorted(set(map(str.lower, outputs["subset"]))))
        else :
            subsets = list(sorted(set(map(str.lower, outputs[0]["subset"]))))
        if len(datanames) == 1 and len(subsets) == 1:
            datasubset = f"{datanames[0]}_{subsets[0]}"
        else:
            datasubset = f"mix_{'_'.join(datanames)}_{'_'.join(subsets)}"
        return datasubset
    
    subrun_path = cfg.paths.output_dir

    test_tokenizer = AACTokenizer(
        level="word",
        lowercase=False,
        punctuation_mode="remove",
        normalize=True,
        backend="ptb",
        cache_path=os.environ["AAC_METRICS_CACHE"] if 'AAC_METRICS_CACHE' in os.environ else None,
        java_path=os.environ["JAVA_PATH"] if 'JAVA_PATH' in os.environ else None,
        tmp_path="/tmp"
    )

    evaluator = MH_AACEvaluator(
        subrun_path=subrun_path,
        test_tokenizer=test_tokenizer,
        cache_path="~/.cache",
        java_path=os.environ["JAVA_PATH"] if 'JAVA_PATH' in os.environ else None,
        tmp_path=tempfile.gettempdir(),
        ckpt_name="unk",
        verbose=1,
        debug=False,
        save_to_csv=True,
        save_dcase_csv_file=False,
        metric_device=None,
        cpus=None,
        save_predictions=False,
        load_predictions=cfg.pickle_path,
        cfg=cfg)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### concatenate cfg.model.generation_config and cfg.additional_generation_config
    generation_config = {**cfg.model.generation_config, **cfg.model.additional_generation_config}

    ### Allow to access with dot notation
    from types import SimpleNamespace
    generation_config_dot = SimpleNamespace(**generation_config)

    evaluator.on_test_start(test_loaders=data_module.test_dataloader(), generation_config=generation_config, device=device, num_hypotheses=cfg.model.num_hyps)
    evaluator.on_test_epoch_start(test_dataloaders=data_module.test_dataloader())
    metrics_all_subsets = evaluator.on_test_epoch_end(prefix='hypothesis', num_hypotheses=cfg.model.num_hyps, tokenizer=processor.tokenizer, generation_config=generation_config_dot)

    writer = SummaryWriter(log_dir=cfg.paths.output_dir)

    # Clean and log metrics to MLflow if enabled
    if cfg.mlflow.enabled:
        # Filter out metrics that shouldn't be logged
        metrics_to_log = {}
        for subset, metrics in metrics_all_subsets.items():
            for key, value in metrics.items():
                if key == 'oracle_mean_nll':
                    value = value.item()
                if key in ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'meteor', 'rouge_l', 'cider', 'spice', 'spider']:
                    metric_name = f"{subset}_{key}"
                    metric_value = metrics[key]['score']
                    metrics_to_log[metric_name] = metric_value
                    writer.add_scalar(f"test/{metric_name}", metric_value, 0)  # 0 is the step number
                elif 'full_' not in key and isinstance(value, (int, float)): # Skip full score dictionaries and non-numeric values
                    metric_name = f"{subset}_{key}"
                    metrics_to_log[metric_name] = value
                    # Log to Tensorboard
                    writer.add_scalar(f"test/{metric_name}", value, 0)  # 0 is the step number
            try:
                # Get the active run or start a new one if none exists
                active_run = mlflow.active_run()
                if active_run is None:
                    logger.warning("No active MLflow run found. Metrics might be logged to a different run.")
                
                # Log metrics within the active run context
                mlflow.log_metrics(metrics_to_log)
                logger.info("Successfully logged metrics to MLflow")
            except Exception as e:
                logger.error(f"Failed to log metrics to MLflow: {e}")
