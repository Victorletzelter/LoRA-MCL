#!/usr/bin/env python

from pathlib import Path
import json
import csv
from typing import Dict, List, Union, Tuple, Any
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import yaml

__author__ = 'Samuel Lipping -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['evaluate_metrics']

def convert_numpy_types(data: Any) -> Any:
    """Recursively convert NumPy data types to native Python types."""
    if isinstance(data, np.ndarray):
        return data.tolist()  # Convert NumPy arrays to lists
    elif isinstance(data, np.generic):
        return data.item()    # Convert NumPy scalars to native Python scalars
    elif isinstance(data, torch.Tensor):
        return data.tolist() if data.dim() > 0 else data.item()  # Convert tensors to lists or scalars
    elif isinstance(data, dict):
        return {str(k).replace(" ", "_").replace("#", ""): convert_numpy_types(v) for k, v in data.items()}
        # return {k: convert_numpy_types(v) for k, v in data.items()}  # Recurse into dictionaries
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]  # Recurse into lists
    else:
        return data  # Return other types unchanged

def write_json(data: Union[List[Dict[str, Any]], Dict[str, Any]],
               path: Path) \
        -> None:
    """ Write a dict or a list of dicts into a JSON file

    :param data: Data to write
    :type data: list[dict[str, any]] | dict[str, any]
    :param path: Path to the output file
    :type path: Path
    """
    # Convert all NumPy types to native types
    clean_data = convert_numpy_types(data)

    with path.open("w") as f:
        json.dump(clean_data, f)

def write_yaml(data: Union[List[Dict[str, Any]], Dict[str, Any]],
               path: Path) -> None:
    """ Write a dict or a list of dicts into a YAML file

    :param data: Data to write
    :type data: list[dict[str, any]] | dict[str, any]
    :param path: Path to the output file
    :type path: Path
    """
    # Convert all NumPy types to native types
    clean_data = convert_numpy_types(data)

    with path.open("w") as f:
        yaml.dump(clean_data, f, default_flow_style=False)

def reformat_to_coco(predictions: List[str],
                     ground_truths: List[List[str]],
                     ids: Union[List[int], None] = None) \
        -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """ Reformat annotations to the MSCOCO format

    :param predictions: List of predicted captions
    :type predictions: list[str]
    :param ground_truths: List of lists of reference captions
    :type ground_truths: list[list[str]]
    :param ids: List of file IDs. If not given, a running integer\
                is used
    :type ids: list[int] | None
    :return: Predictions and reference captions in the MSCOCO format
    :rtype: list[dict[str, any]]
    """
    # Running number as ids for files if not given
    if ids is None:
        ids = range(len(predictions))

    # Captions need to be in format
    # [{
    #     "audio_id": : int,
    #     "caption"  : str
    # ]},
    # as per the COCO results format.
    pred = []
    ref = {
        'info': {'description': 'Clotho reference captions (2019)'},
        'audio samples': [],
        'licenses': [
            {'id': 1},
            {'id': 2},
            {'id': 3}
        ],
        'type': 'captions',
        'annotations': []
    }
    cap_id = 0
    for audio_id, p, gt in zip(ids, predictions, ground_truths):
        p = p[0] if isinstance(p, list) else p
        pred.append({
            'audio_id': audio_id,
            'caption': p
        })

        ref['audio samples'].append({
            'id': audio_id
        })

        for cap in gt:
            ref['annotations'].append({
                'audio_id': audio_id,
                'id': cap_id,
                'caption': cap
            })
            cap_id += 1

    return pred, ref


def mh_reformat_to_coco(predictions,
                     ground_truths,
                     ids= None) \
        -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """ Reformat annotations to the MSCOCO format

    :param predictions: List of predicted captions
    :type predictions: list[str]
    :param ground_truths: List of lists of reference captions
    :type ground_truths: list[list[str]]
    :param ids: List of file IDs. If not given, a running integer\
                is used
    :type ids: list[int] | None
    :return: Predictions and reference captions in the MSCOCO format
    :rtype: list[dict[str, any]]
    """
    # Running number as ids for files if not given
    if ids is None:
        ids = range(len(predictions['0'])) # 0 for hyp 0

    # Captions need to be in format
    # [{
    #     "audio_id": : int,
    #     "caption"  : str
    # ]},
    # as per the COCO results format.
    pred = []
    ref = {
        'info': {'description': 'Clotho reference captions (2019)'},
        'audio samples': [],
        'licenses': [
            {'id': 1},
            {'id': 2},
            {'id': 3}
        ],
        'type': 'captions',
        'annotations': []
    }
    cap_id = 0
    idx = 0
    for audio_id, gt in zip(ids, ground_truths):
        # p = p[0] if isinstance(p, list) else p
        pred.append({
            'audio_id': audio_id,
            'caption': [predictions[f'{hypothesis_idx}'][idx] for hypothesis_idx in range(len(predictions))]
        })
        idx += 1

        ref['audio samples'].append({
            'id': audio_id
        })

        for cap in gt:
            ref['annotations'].append({
                'audio_id': audio_id,
                'id': cap_id,
                'caption': cap
            })
            cap_id += 1

    return pred, ref


def check_and_read_csv(path: Union[str, Path, List[Dict[str, str]]]) \
        -> List[Dict[str, str]]:
    """ If input is a file path, returns the data as a list of dicts \
    (as returned by DictReader) Otherwise just returns the input

    :param path: Input file or its contents (as given by DictReader).
    :type path: Path | str | list[dict[str, str]]
    :return: File contents.
    :rtype: list[dict[str, str]]
    """
    if not isinstance(path, list):
        if isinstance(path, str):
            path = Path(path)

        with path.open('r') as f:
            reader = csv.DictReader(f, dialect='unix')

            result = [row for row in reader]
    else:
        result = path

    return result


