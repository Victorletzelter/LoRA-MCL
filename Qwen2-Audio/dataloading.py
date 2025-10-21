"""This script is used to load the data (stored in HDF files) for the training and inference.
It is adapted from the [Conette library](https://github.com/Labbeti/conette-audio-captioning/), for Clotho and AudioCaps datasets."""

import os
import logging
import os.path as osp
from typing import Iterable, Optional, Union

from torch import nn
from torchoutil.utils.hdf import HDFDataset
import numpy as np
from conette.datamodules.aac_dm import AACDataModule
from conette.datasets.utils import (
    AACSelectColumnsWrapper,
)
from conette.tokenization.aac_tokenizer import AACTokenizer

pylog = logging.getLogger(__name__)


DEFAULT_TRAIN_COLS = ("audio", "audio_shape", "captions")
DEFAULT_VAL_COLS = ("audio", "audio_shape", "captions")
DEFAULT_TEST_COLS = (
    "audio",
    "audio_shape",
    "captions",
    "dataset",
    "subset",
    "fname",
    "index",
)

def get_last_prompt_token_index(processor, prompt_template_with_space):
    # Define the constant prompt portion.
    if prompt_template_with_space:
        prefix = f"{processor.audio_bos_token}{processor.audio_token}{processor.audio_eos_token} Generate the caption in English:"
    else:
        prefix = f"{processor.audio_bos_token}{processor.audio_token}{processor.audio_eos_token}Generate the caption in English:"
    # Tokenize the prefix without adding extra special tokens.
    prefix_ids = processor.tokenizer.encode(prefix, add_special_tokens=False)
    # The last token before gt_text is at position len(prefix_ids)-1.
    return prefix_ids[-1]
class HDFDataModule(AACDataModule):
    """
    Data module for handling HDF-based datasets for audio captioning tasks
    with batch-wise processing using the processor's collate functionality.
    This version processes batches via the processor (i.e., longest padding is computed per batch),
    and removes unused parameters (e.g. for dataset duplication or balancing).
    """
    def __init__(
        self,
        processor = None,
        root: str = "data",
        bsize: int = 512,
        n_workers: Optional[int] = 0,
        pin_memory: bool = True,
        verbose: int = 1,
        train_cols: Iterable[str] = DEFAULT_TRAIN_COLS,
        val_cols: Iterable[str] = DEFAULT_VAL_COLS,
        test_cols: Iterable[str] = DEFAULT_TEST_COLS,
        train_audio_tfm: Optional[nn.Module] = None,
        val_audio_tfm: Optional[nn.Module] = None,
        test_audio_tfm: Optional[nn.Module] = None,
        train_tokenizer: Optional[AACTokenizer] = None,
        train_hdfs: Union[str, Iterable[str]] = (),
        val_hdfs: Union[str, Iterable[str]] = (),
        test_hdfs: Union[str, Iterable[str]] = (),
        max_length: Optional[int] = None,
        text_padding: Optional[str] = None,
        audio_padding: str = "batch",
        prompt_template_with_space: bool = False,
        add_eos_token_data: bool = False,
        use_mcl_wrapper: bool = False,
    ) -> None:
        root = osp.expanduser(osp.expandvars(root))
        super().__init__(
            root=root,
            bsize=bsize,
            test_bsize=bsize,
            n_workers=n_workers,
            pin_memory=pin_memory,
            verbose=verbose,
            train_cols=list(train_cols),
            val_cols=list(val_cols),
            test_cols=list(test_cols),
        )
        self.use_mcl_wrapper = use_mcl_wrapper

        # Process HDF paths
        def process_hdfs_args(hdfs: Union[str, Iterable[str]]) -> list[str]:
            if isinstance(hdfs, str):
                return [hdfs]
            else:
                return list(hdfs)
        self.train_hdfs = process_hdfs_args(train_hdfs)
        self.val_hdfs = process_hdfs_args(val_hdfs)
        self.test_hdfs = process_hdfs_args(test_hdfs)

        # Store additional parameters
        self.processor = processor
        self.max_length = max_length
        self.text_padding = text_padding
        self.audio_padding = audio_padding
        self._train_audio_tfm = train_audio_tfm
        self._val_audio_tfm = val_audio_tfm
        self._test_audio_tfm = test_audio_tfm
        self._train_tokenizer = train_tokenizer
        self.prompt_template_with_space = prompt_template_with_space
        self.add_eos_token_data = add_eos_token_data

        # Propagate into hp for compatibility with parent module
        self.hp.train_hdfs = self.train_hdfs
        self.hp.val_hdfs = self.val_hdfs
        self.hp.test_hdfs = self.test_hdfs
        self.hp.audio_padding = self.audio_padding
        self.hp.train_cols = list(train_cols)
        self.hp.val_cols = list(val_cols)
        self.hp.test_cols = list(test_cols)

    def batch_processor_fn(self, batch: list[dict]) -> dict:
        """
        Processes a batch of samples using the processor in a batch-wise manner.

        Each sample is expected to have:
          - "audio": the audio tensor (or numpy array)
          - "captions": a caption string or a list of captions.

        This function formats each caption (prepending the audio tokens) and calls
        the processor with the full list so that padding is computed on the batch.
        """
        import random
        gt_texts = []
        texts = []
        audios = []
        for item in batch:
            captions = item["captions"]
            if isinstance(captions, list):
                idx = random.randint(0, len(captions) - 1)
                gt_text = captions[idx]
            else:
                gt_text = captions
            gt_text = f"{gt_text}"
            gt_texts.append(gt_text)

            if self.prompt_template_with_space:
                text = f"{self.processor.audio_bos_token}{self.processor.audio_token}{self.processor.audio_eos_token} Generate the caption in English: {gt_text}"
            else:
                text = f"{self.processor.audio_bos_token}{self.processor.audio_token}{self.processor.audio_eos_token}Generate the caption in English: {gt_text}"
            if self.add_eos_token_data is True:
                text = text + f"{self.processor.tokenizer.eos_token}"
            texts.append(text)

            audio = item["audio"]
            if hasattr(audio, "cpu"):
                audio_np = audio.cpu().numpy()
            else:
                audio_np = audio

            audios.append(audio_np)
            
        if self.use_mcl_wrapper:
            processed = self.processor(
            text=texts,
            # audio=audios,
            audios=audios,
            return_tensors="pt",
            padding="longest",
            max_length=self.max_length,
            sampling_rate=16000,
            chunk_length=self.max_length/16000,
        )
        else:
            processed = self.processor(
                text=texts,
                # audio=audios,
                audios=audios,
                return_tensors="pt",
                padding="longest",
                max_length=self.max_length,
                chunk_length=self.max_length/16000,
                sampling_rate=16000,
            )
        processed["labels"] = processed["input_ids"].clone()

        # Find positions of audio_eos_token in each sequence
        last_prompt_token_index = get_last_prompt_token_index(self.processor, self.prompt_template_with_space)
        for i in range(processed["input_ids"].shape[0]):  # iterate over batch dimension
            # Convert comparison to tensor and find matches
            matches = (processed["input_ids"][i] == last_prompt_token_index).nonzero(as_tuple=True)[0]
            if len(matches) > 0:
                if len(matches) > 1:
                    print(f"Multiple matches found for : in sequence {i}")
                # Mask out everything up to and including audio_eos_token
                processed["labels"][i, :matches[0] + 1] = -100
                
        return processed

    def batch_processor_fn_mrefs(self, batch: list[dict]) -> dict:
        """
        Processes a batch of samples using the processor in a batch-wise manner.

        Each sample is expected to have:
          - "audio": the audio tensor (or numpy array)
          - "captions": a caption string or a list of captions.

        This function formats each caption (prepending the audio tokens) and calls
        the processor with the full list so that padding is computed on the batch.
        """

        N_refs = len(batch[0]["captions"])
        output_processed = {}

        for j in range(N_refs):
            gt_texts = []
            texts = []
            audios = []

            for item in batch:
                captions = item["captions"]
                gt_text = captions[j]
                gt_text = f"{gt_text}"
                gt_texts.append(gt_text)

                if self.prompt_template_with_space:
                    text = f"{self.processor.audio_bos_token}{self.processor.audio_token}{self.processor.audio_eos_token} Generate the caption in English: {gt_text}"
                else:
                    text = f"{self.processor.audio_bos_token}{self.processor.audio_token}{self.processor.audio_eos_token}Generate the caption in English: {gt_text}"
                if self.add_eos_token_data is True:
                    text = text + f"{self.processor.tokenizer.eos_token}"
                texts.append(text)

                audio = item["audio"]
                if hasattr(audio, "cpu"):
                    audio_np = audio.cpu().numpy()
                else:
                    audio_np = audio

                audios.append(audio_np)

            if self.use_mcl_wrapper:
                processed = self.processor(
                text=texts,
                # audio=audios,
                audios=audios,
                return_tensors="pt",
                padding="longest",
                max_length=self.max_length,
                sampling_rate=16000,
                chunk_length=self.max_length/16000,
            )
            else:
                processed = self.processor(
                text=texts,
                # audio=audios,
                audios=audios,
                return_tensors="pt",
                padding="longest",
                max_length=self.max_length,
                chunk_length=self.max_length/16000,
                sampling_rate=16000,
            )

            processed["labels"] = processed["input_ids"].clone()

            # Find positions of audio_eos_token in each sequence
            # audio_eos_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.processor.audio_eos_token)
            last_prompt_token_index = get_last_prompt_token_index(self.processor, self.prompt_template_with_space)
            for i in range(processed["input_ids"].shape[0]):  # iterate over batch dimension
                # Convert comparison to tensor and find matches
                matches = (processed["input_ids"][i] == last_prompt_token_index).nonzero(as_tuple=True)[0]
                if len(matches) > 0:
                    if len(matches) > 1:
                        print(f"Multiple matches found for : in sequence {i}")
                    # Mask out everything up to and including audio_eos_token
                    processed["labels"][i, :matches[0] + 1] = -100

            for key in processed.keys():
                output_processed[f"{key}_ref_{j}"] = processed[key]

        output_processed["fname"] = [item["fname"] for item in batch]
        output_processed["subset"] = [item["subset"] for item in batch]
        output_processed["index"] = [item["index"] for item in batch]
        output_processed["dataset"] = [item["dataset"] for item in batch]
                
        return output_processed

    def batch_processor_test_fn(self, batch: list[dict]) -> dict:
        """
        Processes a batch of samples using the processor in a batch-wise manner.

        Each sample is expected to have:
          - "audio": the audio tensor (or numpy array)
          - "captions": a caption string or a list of captions.

        This function formats each caption (prepending the audio tokens) and calls
        the processor with the full list so that padding is computed on the batch.
        """
        orig_batch = batch.copy()
        import random
        raw_gt_texts_mrefs = []
        gt_texts = []
        texts = []
        audios = []
        for item in batch:
            captions = item["captions"]
            raw_gt_texts_mrefs.append(captions)

            if self.prompt_template_with_space:
                text = f"{self.processor.audio_bos_token}{self.processor.audio_token}{self.processor.audio_eos_token} Generate the caption in English:"
            else:
                text = f"{self.processor.audio_bos_token}{self.processor.audio_token}{self.processor.audio_eos_token}Generate the caption in English:"
            if self.add_eos_token_data is True:
                text = text + f"{self.processor.tokenizer.eos_token}"
            texts.append(text)

            audio = item["audio"]
            if hasattr(audio, "cpu"):
                audio_np = audio.cpu().numpy()
            else:
                audio_np = audio

            audios.append(audio_np)
        if self.use_mcl_wrapper:
            processed = self.processor(
                text=texts,
                # audio=audios,
                audios=audios,
                return_tensors="pt",
                padding="longest",
                max_length=self.max_length,
                sampling_rate=16000,
                chunk_length=self.max_length/16000,
            )
        else:
            processed = self.processor(
                text=texts,
                # audio=audios,
                audios=audios,
                return_tensors="pt",
                padding="longest",
                max_length=self.max_length,
                chunk_length=self.max_length/16000,
                sampling_rate=16000,
            )
        processed["labels"] = processed["input_ids"].clone()
        processed["mrefs"] = raw_gt_texts_mrefs
        # Find positions of audio_eos_token in each sequence
        
        processed["fname"] = [item["fname"] for item in batch]
        processed["subset"] = [item["subset"] for item in batch]
        processed["index"] = [item["index"] for item in batch]
        processed["dataset"] = [item["dataset"] for item in batch]
                
        return processed


    def _setup_fit(self) -> None:
        """
        Sets up the training and validation datasets without applying an item-level processor_fn.
        Batch processing will be used via the collate function.
        """
        keep_padding = ("audio",) if self.audio_padding in ("crop", "longest") else ()
        train_dsets_lst = [
            HDFDataset(
                osp.join(self.hp.root, "HDF", fname),
                keep_padding=keep_padding,
                return_added_columns=True,
            )
            for fname in self.hp.train_hdfs
        ]
        val_dsets_lst = [
            HDFDataset(
                osp.join(self.hp.root, "HDF", fname),
                keep_padding=keep_padding,
                return_added_columns=True,
            )
            for fname in self.hp.val_hdfs
        ]
        if self.hp.verbose >= 2:
            pylog.debug(f"HDF datasets loaded. (train={len(train_dsets_lst)}, val={len(val_dsets_lst)})")

        train_dsets_lst = [AACSelectColumnsWrapper(dset, include=self.hp.train_cols) for dset in train_dsets_lst]
        val_dsets_lst = [AACSelectColumnsWrapper(dset, include=self.hp.val_cols) for dset in val_dsets_lst]

        from conette.datasets.utils import AACConcat
        if len(train_dsets_lst) == 1:
            train_dset = train_dsets_lst[0]
        else:
            train_dset = AACConcat(*train_dsets_lst)
        if len(val_dsets_lst) == 1:
            val_dset = val_dsets_lst[0]
        else:
            val_dset = AACConcat(*val_dsets_lst)

        self._train_dset = train_dset
        self._val_dset = val_dset

        # Set collate functions to use batch-wise processing.
        self._train_collate = self.batch_processor_fn
        self._val_collate = self.batch_processor_fn

        if self.hp.verbose >= 1:
            pylog.info(f"Train dataset size: {len(train_dset)}")
            pylog.info(f"Validation dataset size: {len(val_dset)}")

    def _setup_test(self) -> None:
        """
        Sets up the test datasets with batch-wise processing.
        """
        keep_padding = ("audio",) if self.hp.audio_padding in ("crop", "longest") else ()
        dsets = {
            fname: HDFDataset(
                osp.join(self.hp.root, "HDF", fname),
                keep_padding=keep_padding,
                return_added_columns=True,
            )
            for fname in self.hp.test_hdfs
        }
        dsets = {fname: AACSelectColumnsWrapper(dset, include=self.hp.test_cols) for fname, dset in dsets.items()}
        self._test_dsets = dsets
        self._test_collate = self.batch_processor_test_fn

    def _setup_predict(self) -> None:
        """
        Sets up the prediction datasets with batch-wise processing.
        """
        keep_padding = ("audio",) if self.hp.audio_padding in ("crop", "longest") else ()
        dsets = {
            fname: HDFDataset(
                osp.join(self.hp.root, "HDF", fname),
                keep_padding=keep_padding,
                return_added_columns=True,
            )
            for fname in self.hp.predict_hdfs
        }
        # from conette.datamodules.aac_dm import AACSelectColumnsWrapper
        dsets = {fname: AACSelectColumnsWrapper(dset, include=self.hp.test_cols, exclude=("captions",))
                 for fname, dset in dsets.items()}
        self._predict_dsets = dsets
        self._predict_collate = self.batch_processor_fn

        """
        Sets up the prediction datasets with batch-wise processing.
        """
        keep_padding = ("audio",) if self.hp.audio_padding in ("crop", "longest") else ()
        dsets = {
            fname: HDFDataset(
                osp.join(self.hp.root, "HDF", fname),
                keep_padding=keep_padding,
                return_added_columns=True,
            )
            for fname in self.hp.predict_hdfs
        }
        # from conette.datamodules.aac_dm import AACSelectColumnsWrapper
        dsets = {fname: AACSelectColumnsWrapper(dset, include=self.hp.test_cols, exclude=("captions",))
                 for fname, dset in dsets.items()}
        self._predict_dsets = dsets
        self._predict_collate = self.batch_processor_fn