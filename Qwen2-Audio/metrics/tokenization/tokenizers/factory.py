#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tokenization.tokenizers.base import StrTokenizer
from tokenization.tokenizers.word import WordTokenizer


LEVELS = ("word",)


def _pre_tokenizer_factory(level: str, *args, **kwargs) -> StrTokenizer:
    if level == "word":
        return WordTokenizer(*args, **kwargs)
    else:
        raise ValueError(f"Invalid argument {level=}. (expected one of {LEVELS})")
