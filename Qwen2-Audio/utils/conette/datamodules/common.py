#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import os.path as osp
import random
import re
from typing import Any, Callable, Iterable, Optional, Union

import yaml
from torch import Tensor

from conette.tokenization.aac_tokenizer import AACTokenizer

pylog = logging.getLogger(__name__)


def get_hdf_fpaths(
    dataname: str,
    subsets: Iterable[str],
    hdf_root: str,
    hdf_suffix: Optional[str],
    hdf_dname: str = "HDF",
) -> dict[str, str]:
    """
    Get file paths for HDF datasets corresponding to the specified subsets.
    
    For each subset provided, this function constructs the HDF file path using the format:
        {hdf_root}/{hdf_dname}/{dataname_lower}_{subset_lower}_{hdf_suffix}.hdf
    
    Parameters:
      dataname (str): The base name of the dataset.
      subsets (Iterable[str]): An iterable of subset identifiers (e.g., ['train', 'val']).
      hdf_root (str): The root directory where HDF files are stored (supports environment variable expansion).
      hdf_suffix (Optional[str]): Suffix used in naming the HDF files. If None, returns an empty dictionary.
      hdf_dname (str, optional): The subdirectory within hdf_root that contains HDF files (default is "HDF").
    
    Returns:
      dict[str, str]: A dictionary mapping each subset (in lowercase) to its corresponding HDF file path.
    
    Raises:
      FileNotFoundError: If the HDF directory '{hdf_dname}' does not exist in hdf_root.
    """
    if hdf_suffix is None:
        return {}

    dataname = dataname.lower()
    subsets = list(map(str.lower, subsets))
    pattern = re.compile(
        r"(?P<dataname>[a-z]+)_(?P<subset>[a-z]+)_(?P<hdf_suffix>.+)\.hdf"
    )
    hdf_root = osp.expandvars(hdf_root)

    if not osp.isdir(osp.join(hdf_root, hdf_dname)):
        raise FileNotFoundError(f"Cannot find {hdf_dname} directory in {hdf_root=}.")

    hdf_fpaths = {}

    for subset in subsets:
        hdf_fname = f"{dataname}_{subset}_{hdf_suffix}.hdf"
        hdf_fpath = osp.join(hdf_root, hdf_dname, hdf_fname)

        if not osp.isfile(hdf_fpath):
            names = os.listdir(osp.join(hdf_root, hdf_dname))
            matches = [re.match(pattern, name) for name in names]
            availables_hdf_suffix = [
                match["hdf_suffix"]
                for match in matches
                if match is not None
                and match["dataname"] == dataname
                and match["subset"] == subset
            ]

            pylog.error(
                f"Cannot find HDF file '{hdf_fpath}' with {hdf_suffix=}.\n"
                f"Maybe run conette-prepare before and use another hdf_suffix for {dataname}.\n"
                f"Available hdf_suffix for '{dataname}_{subset}' are:\n{yaml.dump(availables_hdf_suffix, sort_keys=False)}"
            )
        hdf_fpaths[subset] = hdf_fpath

    return hdf_fpaths


class OnlineEncodeCaptionsTransform:
    def __init__(
        self,
        audio_tfm: Optional[Callable[[Tensor], Tensor]],
        ref_selection: Union[str, int, slice],
        add_raw_refs: bool,
        tokenizer: AACTokenizer,
        encode_kwargs: dict[str, Any],
        mrefs_src_key: Optional[str] = "captions",
        audio_time_dim: int = -2,
        ref_tfm: Optional[Callable[[str], str]] = None,
    ) -> None:
        super().__init__()
        """
        Initialize an OnlineEncodeCaptionsTransform instance.

        Parameters:
          audio_tfm (Optional[Callable[[Tensor], Tensor]]): Optional transformation function to apply on the audio tensor.
          ref_selection (Union[str, int, slice]): Specifies which caption reference to select. Either:
              - "random": choose a random caption,
              - an integer index: pick the caption at that index,
              - or slice(None) to process all captions.
          add_raw_refs (bool): If True, include the raw reference text in the output.
          tokenizer (AACTokenizer): Tokenizer to encode the captions.
          encode_kwargs (dict[str, Any]): Additional keyword arguments for the tokenizer encoding functions.
          mrefs_src_key (Optional[str]): Key in the input dictionary to extract the caption references (default "captions").
          audio_time_dim (int): The dimension of the audio tensor representing time (default -2).
          ref_tfm (Optional[Callable[[str], str]]): Optional transformation function to apply on caption text.
        """
        self.audio_tfm = audio_tfm
        self.ref_selection = ref_selection
        self.add_raw_refs = add_raw_refs
        self.tokenizer = tokenizer
        self.encode_kwargs = encode_kwargs
        self.mrefs_src_key = mrefs_src_key
        self.audio_time_dim = audio_time_dim
        self.ref_tfm = ref_tfm

    def __call__(self, item: dict[str, Any]) -> dict[str, Any]:
        """
        Process a data sample by applying audio transformation (if provided) and encoding captions.

        This function performs the following steps:
          - Applies the audio transformation (audio_tfm) on the audio data if available.
          - Based on the 'ref_selection' parameter, selects a caption:
              - If an integer, encodes a single caption.
              - If "random", picks a random caption.
              - If slice(None), encodes the entire batch of captions.
          - Uses the tokenizer to encode the selected caption(s).
          - Optionally stores the raw caption(s) in the output if 'add_raw_refs' is True.

        Parameters:
          item (dict[str, Any]): A dictionary containing at least:
              - "audio": the audio tensor.
              - "audio_shape": expected shape of the audio tensor.
              - Caption data under the key specified by 'mrefs_src_key'.

        Returns:
          dict[str, Any]: The updated dictionary with the transformed 'audio' and encoded caption(s).
        """
        if self.audio_tfm is not None:
            audio = item["audio"]
            audio_shape = item["audio_shape"]
            audio_len = audio_shape[self.audio_time_dim]
            if audio_len < audio.shape[self.audio_time_dim]:
                mask = [slice(None) for _ in range(audio.ndim)]
                mask[self.audio_time_dim] = slice(audio_len)
                audio[mask] = self.audio_tfm(audio[mask])
            else:
                audio = self.audio_tfm(audio)
            item["audio"] = audio.contiguous()

        if self.mrefs_src_key is not None:
            refs = item[self.mrefs_src_key]

            if isinstance(self.ref_selection, str):
                if self.ref_selection == "random":
                    idxs = random.randint(0, len(refs) - 1)
                else:
                    raise ValueError(f"Invalid argument {self.ref_selection=}.")
            else:
                idxs = self.ref_selection

            if isinstance(idxs, int):
                ref = refs[idxs]
                if self.ref_tfm is not None:
                    ref = self.ref_tfm(ref)

                cap = self.tokenizer.encode_single(
                    ref,
                    **self.encode_kwargs,
                )

                item["captions"] = cap
                if self.add_raw_refs:
                    item["references"] = ref

            elif idxs == slice(None):
                item.pop("captions")

                if self.ref_tfm is not None:
                    refs = [self.ref_tfm(ref) for ref in refs]

                mcaps = self.tokenizer.encode_batch(
                    refs,
                    **self.encode_kwargs,
                )
                item["mult_captions"] = mcaps

                if self.add_raw_refs:
                    item["mult_references"] = refs

            else:
                raise ValueError(
                    f"Invalid argument {idxs=} with {self.ref_selection=}."
                )

        return item


class OnlineEncodeCaptionsTransformWithEmbs:
    def __init__(
        self,
        audio_tfm: Optional[Callable],
        ref_selection: Union[str, int, slice],
        add_raw_refs: bool,
        tokenizer: AACTokenizer,
        encode_kwargs: dict[str, Any],
        mrefs_src_key: str = "captions",
        mrefs_embs_src_key: str = "captions_embs",
    ) -> None:
        super().__init__()
        """
        Initialize an OnlineEncodeCaptionsTransformWithEmbs instance.

        Parameters:
          audio_tfm (Optional[Callable]): Optional transformation function to apply to the audio tensor.
          ref_selection (Union[str, int, slice]): Determines which caption to select:
              - "random": pick a random caption,
              - an integer: select a specific caption,
              - or slice(None) to process all captions.
          add_raw_refs (bool): Whether to include the raw caption text in the output.
          tokenizer (AACTokenizer): Tokenizer to encode the captions.
          encode_kwargs (dict[str, Any]): Additional keyword arguments for the tokenizer encoding.
          mrefs_src_key (str): Key to retrieve captions from the input dictionary (default "captions").
          mrefs_embs_src_key (str): Key to retrieve caption embeddings from the input dictionary (default "captions_embs").
        """
        self.audio_tfm = audio_tfm
        self.ref_selection = ref_selection
        self.add_raw_refs = add_raw_refs
        self.tokenizer = tokenizer
        self.encode_kwargs = encode_kwargs
        self.mrefs_src_key = mrefs_src_key
        self.mrefs_embs_src_key = mrefs_embs_src_key

    def __call__(self, item: dict[str, Any]) -> dict[str, Any]:
        """
        Process a data sample by applying audio transformation (if provided) and encoding captions with corresponding embeddings.

        This function performs the following steps:
          - Applies the audio transformation (audio_tfm) to the "audio" key if available.
          - Based on 'ref_selection', selects a caption to encode:
              - If an integer, encodes a single caption and selects its corresponding embedding.
              - If "random", randomly selects a caption.
              - If slice(None), processes a batch of captions.
          - Uses the tokenizer to encode the caption(s).
          - Attaches the corresponding caption embedding(s) to the output.
          - Optionally, includes the raw caption text(s) if 'add_raw_refs' is True.

        Parameters:
          item (dict[str, Any]): A dictionary containing:
              - "audio": the audio tensor.
              - Caption text under the key specified by 'mrefs_src_key'.
              - Caption embeddings under the key specified by 'mrefs_embs_src_key'.

        Returns:
          dict[str, Any]: The updated dictionary with processed audio, encoded caption(s),
                          and associated caption embedding(s).
        """
        references = item[self.mrefs_src_key]
        references_embs = item[self.mrefs_embs_src_key]

        if self.audio_tfm is not None:
            item["audio"] = self.audio_tfm(item["audio"])

        if isinstance(self.ref_selection, str):
            if self.ref_selection == "random":
                idxs = random.randint(0, len(references) - 1)
            else:
                raise ValueError(f"Invalid argument {self.ref_selection=}.")
        else:
            idxs = self.ref_selection

        if isinstance(idxs, int):
            reference = references[idxs]
            caption = self.tokenizer.encode_single(
                reference,
                **self.encode_kwargs,
            )

            item["captions"] = caption
            item[self.mrefs_embs_src_key] = references_embs[idxs]
            if self.add_raw_refs:
                item["references"] = reference

        elif idxs == slice(None):
            item.pop("captions")
            mult_captions = self.tokenizer.encode_batch(
                references,
                **self.encode_kwargs,
            )

            item["mult_captions"] = mult_captions
            item[f"mult_{self.mrefs_embs_src_key}"] = references_embs[idxs]
            if self.add_raw_refs:
                item["mult_references"] = references

        else:
            raise ValueError(f"Invalid argument {idxs=} with {self.ref_selection=}.")

        return item
