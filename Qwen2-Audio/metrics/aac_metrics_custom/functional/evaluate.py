#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import time
import numpy as np
from functools import lru_cache as cache
from functools import partial
# from functools import cache, partial

from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

from typing import Any, ClassVar, Generic, Optional, TypeVar, Union
from typing import Dict
from typing import Tuple
from typing import List 

import torch
from torch import Tensor

# import torchmetrics
# from torchmetrics.functional.text.bert import _DEFAULT_MODEL, bert_score

from aac_metrics_custom.functional.bert_score_mrefs import bert_score_mrefs
from aac_metrics_custom.functional.bleu import bleu, bleu_1, bleu_2, bleu_3, bleu_4
from aac_metrics_custom.functional.cider_d import cider_d
from aac_metrics_custom.functional.fense import fense
from aac_metrics_custom.functional.fer import fer
from aac_metrics_custom.functional.meteor import meteor
from aac_metrics_custom.functional.rouge_l import rouge_l
from aac_metrics_custom.functional.sbert_sim import sbert_sim
from aac_metrics_custom.functional.spice import spice
from aac_metrics_custom.functional.spider import spider
from aac_metrics_custom.functional.spider_fl import spider_fl
from aac_metrics_custom.functional.spider_max import spider_max
from aac_metrics_custom.functional.vocab import vocab
from aac_metrics_custom.utils.checks import check_metric_inputs
from aac_metrics_custom.utils.tokenization import preprocess_mono_sents, preprocess_mult_sents

pylog = logging.getLogger(__name__)


METRICS_SETS: Dict[str, Tuple[str, ...]] = {
    # Legacy metrics for AAC
    "default": (
        "bleu_1",
        "bleu_2",
        "bleu_3",
        "bleu_4",
        "meteor",
        "rouge_l",
        "spider",  # includes cider_d, spice
    ),
    # DCASE challenge task6a metrics for 2020, 2021 and 2022
    "dcase2020": (
        "bleu_1",
        "bleu_2",
        "bleu_3",
        "bleu_4",
        "meteor",
        "rouge_l",
        "spider",  # includes cider_d, spice
    ),
    # DCASE challenge task6a metrics for 2023
    "dcase2023": (
        "meteor",
        "spider_fl",  # includes cider_d, spice, spider, fer
    ),
    # All metrics
    "all": (
        "bleu_1",
        "bleu_2",
        "bleu_3",
        "bleu_4",
        "meteor",
        "rouge_l",
        "fense",  # includes sbert, fer
        "spider_fl",  # includes cider_d, spice, spider, fer
        "vocab",
        "bert_score",
    ),
}
DEFAULT_METRICS_SET_NAME = "default"


def evaluate(
    candidates: List[str],
    mult_references: List[List[str]],
    preprocess: bool = True,
    metrics: Union[
        str, Iterable[str], Iterable[Callable[[list, list], tuple]]
    ] = DEFAULT_METRICS_SET_NAME,
    cache_path: Union[str, Path, None] = None,
    java_path: Union[str, Path, None] = None,
    tmp_path: Union[str, Path, None] = None,
    device: Union[str, torch.device, None] = "cuda_if_available",
    verbose: int = 0,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """Evaluate candidates with multiple references with custom metrics.

    :param candidates: The list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target.
    :param preprocess: If True, the candidates and references will be passed as input to the PTB stanford tokenizer before computing metrics.defaults to True.
    :param metrics: The name of the metric list or the explicit list of metrics to compute. defaults to "default".
    :param cache_path: The path to the external code directory. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_cache_path`.
    :param java_path: The path to the java executable. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_java_path`.
    :param tmp_path: Temporary directory path. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_tmp_path`.
    :param device: The PyTorch device used to run FENSE and SPIDErFL models.
        If None, it will try to detect use cuda if available. defaults to "cuda_if_available".
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple contains the corpus and sentences scores.
    """
    # check_metric_inputs(candidates, mult_references)

    metrics = _instantiate_metrics_functions(
        metrics, cache_path, java_path, tmp_path, device, verbose
    )

    if preprocess:
        common_kwds: Dict[str, Any] = dict(
            cache_path=cache_path,
            java_path=java_path,
            tmp_path=tmp_path,
            verbose=verbose,
        )
        candidates = preprocess_mono_sents(
            candidates,
            **common_kwds,
        )
        mult_references = preprocess_mult_sents(
            mult_references,
            **common_kwds,
        )

    outs_corpus = {}
    outs_sents = {}

    for i, metric in enumerate(metrics):
        if isinstance(metric, partial):
            name = metric.func.__qualname__
        elif hasattr(metric, "__qualname__"):
            name = metric.__qualname__
        else:
            name = metric.__class__.__qualname__

        if verbose >= 1:
            pylog.info(f"[{i+1:2d}/{len(metrics):2d}] Computing {name} metric...")

        start = time.perf_counter()
        outs_corpus_i, outs_sents_i = metric(candidates, mult_references)
        end = time.perf_counter()

        if verbose >= 1:
            pylog.info(
                f"[{i+1:2d}/{len(metrics):2d}] Metric {name} computed in {end - start:.2f}s."
            )

        if __debug__:
            corpus_overlap = tuple(
                set(outs_corpus_i.keys()).intersection(outs_corpus.keys())
            )
            sents_overlap = tuple(
                set(outs_sents_i.keys()).intersection(outs_sents.keys())
            )
            if len(corpus_overlap) > 0 or len(sents_overlap) > 0:
                warn_once(
                    f"Found overlapping metric outputs names. (found {corpus_overlap} and {sents_overlap} at least twice)"
                )

        outs_corpus.update(outs_corpus_i)
        outs_sents.update(outs_sents_i)

    return outs_corpus, outs_sents

def extract_oracle_metrics(
    outs_corpus: Dict[str, Tensor], outs_sents: Dict[str, Tensor]):
    oracle_corpus = {}
    oracle_sents = {}

    for metric_name in outs_corpus['0'].keys():

        oracle_corpus[metric_name] = max([outs_corpus[hypothesis_idx][metric_name].numpy().item() for hypothesis_idx in outs_corpus.keys()])

        # Find the hypothesis index (key) that corresponds to the maximum value for this metric
        max_key = max(outs_corpus, key=lambda idx: outs_corpus[idx][metric_name])
        
        # Retrieve the corresponding sentence-level metric from the outs_sents
        if metric_name in outs_sents[max_key]:
            oracle_sents[metric_name] = outs_sents[max_key][metric_name]
        else:
            oracle_sents[metric_name] = None

    return oracle_corpus, oracle_sents

def extract_oracle_sentence_metrics(
    outs_sents: Dict[str, Tensor],
    return_full_scores: bool = False,
    audio_paths: List[str] = None,
    higher_is_better: bool = True):
    """
    Extract the oracle-based sentence-level metrics from the outs_sents dictionary.
    Args:
        outs_sents: Dictionary containing the sentence-level metrics for each hypothesis.
        return_full_scores: Whether to return the full scores for each sentence.
        audio_paths: List of audio paths.
        higher_is_better: Whether the higher the metric value, the better.
    Returns:
        mean_oracle_sents: Dictionary containing the mean (over the examples of the eval set) of the oracle-based sentence-level metrics.
        individual_oracle_sents: Dictionary containing the individual oracle-based sentence-level metrics for each sentence in the eval set.
        full_sents: Dictionary containing the full scores for each sentence and for each hypothesis.
    """
    individual_oracle_sents = {}
    mean_oracle_sents = {}
    full_sents = {}

    for metric_name in outs_sents['0'].keys():
        full_sents[metric_name] = {}

    if '0' in outs_sents and len(outs_sents['0']) == 0:
        N_sentences = 0
    else :
        N_sentences = [outs_sents['0'][key] for key in outs_sents['0'].keys()][0].shape[0]

    for metric_name in outs_sents['0'].keys():
        # Create a tensor of shape [N_sentences, N_hypotheses] containing the metric values for each sentence
        metric_values = torch.stack([outs_sents[hypothesis_idx][metric_name] for hypothesis_idx in outs_sents.keys()], dim=1)

        if audio_paths is not None:
            for i, audio_path in enumerate(audio_paths):
                audio_path = audio_path.split('/')[-1]
                full_sents[metric_name][audio_path] = metric_values[i, :].numpy().tolist()

        assert metric_values.shape == (N_sentences, len(outs_sents))

        # compute the maximum value for each sentence
        if higher_is_better:
            individual_oracle_sents[metric_name] = torch.max(metric_values, dim=1).values# shape: [N_sentences]
        else:
            individual_oracle_sents[metric_name] = torch.min(metric_values, dim=1).values# shape: [N_sentences]
        mean_oracle_sents[metric_name] = torch.mean(individual_oracle_sents[metric_name]).item() # shape: []

        individual_oracle_sents[metric_name] = individual_oracle_sents[metric_name].numpy()

    if return_full_scores:
        return mean_oracle_sents, individual_oracle_sents, full_sents
    else:
        return mean_oracle_sents, individual_oracle_sents, None

def mh_evaluate(
    candidates: List[str],
    mult_references: List[List[str]],
    preprocess: bool = True,
    metrics: Union[
        str, Iterable[str], Iterable[Callable[[list, list], tuple]]
    ] = DEFAULT_METRICS_SET_NAME,
    cache_path: Union[str, Path, None] = None,
    java_path: Union[str, Path, None] = None,
    tmp_path: Union[str, Path, None] = None,
    device: Union[str, torch.device, None] = "cuda_if_available",
    verbose: int = 0,
    return_full_scores: bool = False,
    audio_paths: List[str] = None,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """Evaluate candidates with multiple references with custom metrics.

    :param candidates: The list of list of sentences to evaluate (multi-hypotheses).
    :param mult_references: The list of list of sentences used as target.
    :param preprocess: If True, the candidates and references will be passed as input to the PTB stanford tokenizer before computing metrics.defaults to True.
    :param metrics: The name of the metric list or the explicit list of metrics to compute. defaults to "default".
    :param cache_path: The path to the external code directory. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_cache_path`.
    :param java_path: The path to the java executable. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_java_path`.
    :param tmp_path: Temporary directory path. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_tmp_path`.
    :param device: The PyTorch device used to run FENSE and SPIDErFL models.
        If None, it will try to detect use cuda if available. defaults to "cuda_if_available".
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple contains the corpus and oracle-based sentences scores.
    """
    metrics = _instantiate_metrics_functions(
        metrics, cache_path, java_path, tmp_path, device, verbose
    )

    if preprocess:
        common_kwds: Dict[str, Any] = dict(
            cache_path=cache_path,
            java_path=java_path,
            tmp_path=tmp_path,
            verbose=verbose,
        )

        for key in candidates: # key = '0', ..., f'{n_hypotheses-1}'. Each candidate is a list of sentences (predictions of a given hypothesis).
            candidates[key] = preprocess_mono_sents(
                candidates[key],
                **common_kwds,
            )

        mult_references = preprocess_mult_sents(
            mult_references,
            **common_kwds,
        ) # mult_references is a list of lists of sentences (references).

    mean_oracle_metrics = {}
    outs_sents = {}
    full_scores = {}

    for i, metric in enumerate(metrics): # loop over the different metrics (BLEU, METEOR, ROUGE-L, etc.).

        if isinstance(metric, partial):
            name = metric.func.__qualname__
        elif hasattr(metric, "__qualname__"):
            name = metric.__qualname__
        else:
            name = metric.__class__.__qualname__

        if verbose >= 1:
            pylog.info(f"[{i+1:2d}/{len(metrics):2d}] Computing {name} metric...")
        
        start = time.perf_counter()

        mean_oracle_metric_i, outs_sents_i, full_scores_i = {}, {}, {}

        for key in candidates: # key = '0', ..., f'{n_hypotheses-1}'. Each candidate is a list of sentences (predictions of a given hypothesis).
            _, outs_sents_i[key] = metric(candidates[key], mult_references)

        if metric == 'Fer':
            higher_is_better = False # For Fer, lower is better
        else:
            higher_is_better = True # For other metrics, higher is better

        mean_oracle_metric_i, outs_sents_i, full_scores_i = extract_oracle_sentence_metrics(outs_sents_i, return_full_scores=return_full_scores, audio_paths=audio_paths, higher_is_better=higher_is_better) # mean_oracle_sents, individual_oracle_sents
        # 'full_scores_i': corresponds to the full scores for each example (and for each hypothesis).
        # 'outs_sents_i': corresponds to the oracle-based sentence-level metric for each example.
        # 'mean_oracle_metric_i': We named mean_oracle_metric_i to correspond to the mean of the outs_sents_i, i.e.,
        # the mean (over the examples of the eval set) of the oracle-based sentence-level metric.

        end = time.perf_counter()

        if verbose >= 1:
            pylog.info(
                f"[{i+1:2d}/{len(metrics):2d}] Metric {name} computed in {end - start:.2f}s."
            )

        if __debug__:
            corpus_overlap = tuple(
                set(mean_oracle_metric_i.keys()).intersection(mean_oracle_metrics.keys())
            )
            sents_overlap = tuple(
                set(outs_sents_i.keys()).intersection(outs_sents.keys())
            )
            if len(corpus_overlap) > 0 or len(sents_overlap) > 0:
                warn_once(
                    f"Found overlapping metric outputs names. (found {corpus_overlap} and {sents_overlap} at least twice)"
                )

        mean_oracle_metrics.update(mean_oracle_metric_i)
        outs_sents.update(outs_sents_i)
        if return_full_scores:
            full_scores.update(full_scores_i)

    if return_full_scores:
        return mean_oracle_metrics, outs_sents, full_scores
    else:
        return mean_oracle_metrics, outs_sents, None

def dcase2023_evaluate(
    candidates: List[str],
    mult_references: List[List[str]],
    preprocess: bool = True,
    cache_path: Union[str, Path, None] = None,
    java_path: Union[str, Path, None] = None,
    tmp_path: Union[str, Path, None] = None,
    device: Union[str, torch.device, None] = "cuda_if_available",
    verbose: int = 0,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """Evaluate candidates with multiple references with the DCASE2023 Audio Captioning metrics.

    :param candidates: The list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target.
    :param preprocess: If True, the candidates and references will be passed as input to the PTB stanford tokenizer before computing metrics.
        defaults to True.
    :param cache_path: The path to the external code directory. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_cache_path`.
    :param java_path: The path to the java executable. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_java_path`.
    :param tmp_path: Temporary directory path. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_tmp_path`.
    :param device: The PyTorch device used to run FENSE and SPIDErFL models.
        If None, it will try to detect use cuda if available. defaults to "cuda_if_available".
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple contains the corpus and sentences scores.
    """
    return evaluate(
        candidates=candidates,
        mult_references=mult_references,
        preprocess=preprocess,
        metrics="dcase2023",
        cache_path=cache_path,
        java_path=java_path,
        tmp_path=tmp_path,
        device=device,
        verbose=verbose,
    )


def _instantiate_metrics_functions(
    metrics: Union[str, Iterable[str], Iterable[Callable[[list, list], tuple]]] = "all",
    cache_path: Union[str, Path, None] = None,
    java_path: Union[str, Path, None] = None,
    tmp_path: Union[str, Path, None] = None,
    device: Union[str, torch.device, None] = "cuda_if_available",
    verbose: int = 0,
) -> List[Callable]:
    if isinstance(metrics, str) and metrics in METRICS_SETS:
        metrics = METRICS_SETS[metrics]

    if isinstance(metrics, str):
        metrics = [metrics]
    else:
        metrics = list(metrics)  # type: ignore

    if not all(isinstance(metric, (str, Callable)) for metric in metrics):
        raise TypeError(
            "Invalid argument type for metrics. (expected str, Iterable[str] or Iterable[Metric])"
        )

    metric_factory = _get_metric_factory_functions(
        return_all_scores=True,
        cache_path=cache_path,
        java_path=java_path,
        tmp_path=tmp_path,
        device=device,
        verbose=verbose,
    )

    metrics_inst: List[Callable] = []
    for metric in metrics:
        if isinstance(metric, str):
            metric = metric_factory[metric]
        metrics_inst.append(metric)
    return metrics_inst


def _get_metric_factory_functions(
    return_all_scores: bool = True,
    cache_path: Union[str, Path, None] = None,
    java_path: Union[str, Path, None] = None,
    tmp_path: Union[str, Path, None] = None,
    device: Union[str, torch.device, None] = "cuda_if_available",
    verbose: int = 0,
    init_kwds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Callable[[List[str], List[List[str]]], Any]]:
    if init_kwds is None or init_kwds is ...:
        init_kwds = {}

    # init_kwds = init_kwds | dict(return_all_scores=return_all_scores)
    # update version
    init_kwds.update(dict(return_all_scores=return_all_scores))

    import os 
    if 'LOCAL_CACHE' in os.environ and not os.path.exists(os.path.join(os.environ["LOCAL_CACHE"],'huggingface')):
        model_path_bert_score_DEFAULT_MODEL = "roberta-large"
    else:
        model_path_bert_score_DEFAULT_MODEL = os.path.join(os.environ["LOCAL_CACHE"],"huggingface/hub/models--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59/")
    if 'LOCAL_CACHE' in os.environ and not os.path.exists(os.path.join(os.environ["LOCAL_CACHE"],'torch')):
        model_path_DEFAULT_SBERT_SIM_MODEL = "sentence-transformers_paraphrase-TinyBERT-L6-v2"
    else:
        model_path_DEFAULT_SBERT_SIM_MODEL = os.path.join(os.environ["LOCAL_CACHE"],"torch/sentence_transformers/sentence-transformers_paraphrase-TinyBERT-L6-v2")
    if 'LOCAL_CACHE' in os.environ and not os.path.exists(os.path.join(os.environ["LOCAL_CACHE"],'torch/hub/fense_data')):
        model_path_DEFAULT_FER_MODEL = "echecker_clotho_audiocaps_base"
    else:
        model_path_DEFAULT_FER_MODEL = os.path.join(os.environ["LOCAL_CACHE"],"torch/hub/fense_data/echecker_clotho_audiocaps_base.ckpt")

    factory = {
        "bert_score": partial(
            bert_score_mrefs,
            model=model_path_bert_score_DEFAULT_MODEL,
            **init_kwds,
        ),
        "bleu": partial(
            bleu,
            **init_kwds,
        ),
        "bleu_1": partial(
            bleu_1,
            **init_kwds,
        ),
        "bleu_2": partial(
            bleu_2,
            **init_kwds,
        ),
        "bleu_3": partial(
            bleu_3,
            **init_kwds,
        ),
        "bleu_4": partial(
            bleu_4,
            **init_kwds,
        ),
        "cider_d": partial(
            cider_d,
            **init_kwds,
        ),
        "fer": partial(
            fer,
            device=device,
            verbose=verbose,
            sbert_model=model_path_DEFAULT_SBERT_SIM_MODEL,
            echecker=model_path_DEFAULT_FER_MODEL,
            **init_kwds,
        ),
        "fense": partial(
            fense,
            device=device,
            verbose=verbose,
            sbert_model=model_path_DEFAULT_SBERT_SIM_MODEL,
            echecker=model_path_DEFAULT_FER_MODEL,
            **init_kwds,
        ),
        "meteor": partial(
            meteor,
            cache_path=cache_path,
            java_path=java_path,
            verbose=verbose,
            **init_kwds,
        ),
        "rouge_l": partial(
            rouge_l,
            **init_kwds,
        ),
        "sbert_sim": partial(
            sbert_sim,
            device=device,
            verbose=verbose,
            sbert_model=model_path_DEFAULT_SBERT_SIM_MODEL,
            **init_kwds,
        ),
        "spice": partial(
            spice,
            cache_path=cache_path,
            java_path=java_path,
            tmp_path=tmp_path,
            verbose=verbose,
            **init_kwds,
        ),
        "spider": partial(
            spider,
            cache_path=cache_path,
            java_path=java_path,
            tmp_path=tmp_path,
            verbose=verbose,
            **init_kwds,
        ),
        "spider_max": partial(
            spider_max,
            cache_path=cache_path,
            java_path=java_path,
            tmp_path=tmp_path,
            verbose=verbose,
            **init_kwds,
        ),
        "spider_fl": partial(
            spider_fl,
            cache_path=cache_path,
            java_path=java_path,
            tmp_path=tmp_path,
            device=device,
            verbose=verbose,
            echecker=model_path_DEFAULT_FER_MODEL,
            **init_kwds,
        ),
        "vocab": partial(
            vocab,
            verbose=verbose,
            **init_kwds,
        ),
    }
    return factory


# @cache
def warn_once(msg: str) -> None:
    pylog.warning(msg)
