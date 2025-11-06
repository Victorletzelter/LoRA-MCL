"""
MCL (Multiple Choice Learning) Wrapper for Transformers Models

This package provides a general wrapper that can transform any Hugging Face
transformers model into its MCL version with Winner-Take-All training.
"""

from .peft_patches import patch_peft_for_mcl
from .config import MCLConfig
from .utils import MCLModelWrapper
from .wrapper import MCLForwardMixin, MCLModelWrapper, get_peft_mcl
from .trainer import MCLTrainer
from .forward_mixin import MCLModelOutput


__version__ = "0.1.0"
__all__ = [
    "MCLConfig",
    "MCLForwardMixin",
    "MCLModelWrapper",
    "MCLModelOutput",
    "get_peft_mcl",
    "MCLTrainer",
    "patch_peft_for_mcl",
]
