"""
Main MCL wrapper that can transform any Hugging Face model into its MCL version.
"""

import torch
import torch.nn as nn
from typing import Type, Optional, Dict, Any
from transformers import PreTrainedModel, AutoConfig, AutoModelForCausalLM
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
import inspect
import logging
from peft import LoraConfig
from .config import MCLConfig
from .forward_mixin import MCLForwardMixin

logger = logging.getLogger(__name__)


def create_mcl_model_class(
    base_model_class: Type[PreTrainedModel], model_name_suffix: str = "_MCL"
) -> Type[PreTrainedModel]:
    """
    Create an MCL version of any ForConditionalGeneration model class.

    Args:
        base_model_class: The base model class (e.g., LlamaForCausalLM)
        model_name_suffix: Suffix to add to the class name

    Returns:
        New MCL model class
    """

    class MCLModel(MCLForwardMixin, base_model_class):
        """Dynamically created MCL model class"""

        def __init__(self, config, *args, **kwargs):
            # Initialize base class first
            super().__init__(config, *args, **kwargs)

        def _original_forward(self, *args, **kwargs):
            """Call the original forward method from the base class"""
            # Get the original forward method, skipping the MCL mixin
            for cls in self.__class__.__mro__[2:]:  # Skip MCLForwardMixin and MCLModel
                if hasattr(cls, "forward") and cls != MCLForwardMixin:
                    return cls.forward(self, *args, **kwargs)
            raise RuntimeError("Could not find original forward method")

    # Set the class name
    MCLModel.__name__ = base_model_class.__name__ + model_name_suffix
    MCLModel.__qualname__ = base_model_class.__qualname__ + model_name_suffix

    return MCLModel


def apply_mcl_to_config(config, mcl_config: MCLConfig):
    """Apply MCL configuration to a model config"""
    config.num_hyps = mcl_config.num_hyps
    config.wta_training_mode = mcl_config.wta_training_mode
    config.wta_params_epsilon = mcl_config.wta_params_epsilon
    config.wta_params_ini_temp = mcl_config.wta_params_ini_temp
    config.wta_params_fin_temp = mcl_config.wta_params_fin_temp
    config.wta_params_decay_rate = mcl_config.wta_params_decay_rate
    config.wta_params_schedule_mode = mcl_config.wta_params_schedule_mode
    # config.native_group_lora_enabled = mcl_config.native_group_lora_enabled
    return config


class MCLModelWrapper:
    """
    General wrapper that can transform any Hugging Face model into its MCL version.
    """

    def __init__(self, lora_config=None, mcl_config=None):
        # Use standard LoraConfig from PEFT
        self.lora_config = lora_config  # Standard PEFT LoraConfig
        self.mcl_config = mcl_config or MCLConfig()  # Our MCL config
        self._mcl_classes = {}  # Cache for created MCL classes

    def wrap_model(self, model_name_or_path: str, **model_kwargs) -> PreTrainedModel:
        """
        Create an MCL version of any Hugging Face model.

        Args:
            model_name_or_path: Model identifier or path
            **model_kwargs: Additional arguments for model loading

        Returns:
            MCL-enabled model instance
        """
        # Load the original config
        config = AutoConfig.from_pretrained(model_name_or_path, **model_kwargs)

        # Apply MCL configuration
        config = apply_mcl_to_config(config, self.mcl_config)

        # Get the original model class
        base_model_class = self._get_model_class_for_config(config)

        # Create or get cached MCL class
        mcl_class_key = base_model_class.__name__
        if mcl_class_key not in self._mcl_classes:
            self._mcl_classes[mcl_class_key] = create_mcl_model_class(base_model_class)

        mcl_model_class = self._mcl_classes[mcl_class_key]

        # Create the MCL model instance
        model = mcl_model_class.from_pretrained(model_name_or_path, config=config, **model_kwargs)

        # Apply LoRA if specified
        if self.lora_config:
            model = self._apply_lora(model)

        return model

    def wrap_existing_model(self, model: PreTrainedModel) -> PreTrainedModel:
        """
        Wrap an existing model instance with MCL functionality.

        Args:
            model: Existing model instance

        Returns:
            MCL-enabled model instance
        """
        # Apply MCL configuration to the model's config
        model.config = apply_mcl_to_config(model.config, self.mcl_config)

        # Get the model class
        base_model_class = model.__class__

        # Create or get cached MCL class
        mcl_class_key = base_model_class.__name__
        if mcl_class_key not in self._mcl_classes:
            self._mcl_classes[mcl_class_key] = create_mcl_model_class(base_model_class)

        mcl_model_class = self._mcl_classes[mcl_class_key]

        # Convert the model instance to MCL class
        model.__class__ = mcl_model_class

        # Initialize MCL components
        MCLForwardMixin.__init__(model)

        # Apply LoRA if specified
        if self.lora_config:
            model = self._apply_lora(model)

        return model

    def _get_model_class_for_config(self, config):
        """Get the appropriate model class for a config"""
        # Try to get from the mapping
        for config_class, model_class in MODEL_FOR_CAUSAL_LM_MAPPING.items():
            if isinstance(config, config_class):
                return model_class

        # Fallback: try AutoModelForCausalLM
        return AutoModelForCausalLM

    def _apply_lora(self, model):
        """Apply LoRA to the model"""
        try:
            from peft import get_peft_model
        except ImportError:
            logger.error("PEFT library not found. Please install it to use LoRA functionality.")
            return model

        # Handle group LoRA configuration
        # if self.mcl_config.use_group_lora:
        #     # Use custom GroupLinear implementation
        #     model.config.native_group_lora_enabled = True

        # Apply multiple LoRA adapters for multi-hypothesis
        if self.mcl_config.num_hyps > 1:
            for hyp_idx in range(self.mcl_config.num_hyps):
                adapter_name = f"lora{hyp_idx}"
                if hyp_idx == 0:
                    model = get_peft_model(model, self.lora_config, adapter_name=adapter_name)
                else:
                    model.add_adapter(adapter_name, self.lora_config)
        else:
            model = get_peft_model(model, self.lora_config)

        return model


# Convenience function
def create_mcl_model(
    model_name_or_path: str,
    num_hyps: int = 3,
    wta_training_mode: str = "wta",
    use_group_lora: bool = False,
    lora_config: Optional[LoraConfig] = None,
    wta_params_epsilon: float = 0.0,
    wta_params_ini_temp: float = 1.0,
    wta_params_fin_temp: float = 0.01,
    wta_params_decay_rate: float = 0.999,
    wta_params_schedule_mode: str = "global_step",
    **model_kwargs,
) -> PreTrainedModel:
    """
    Convenience function to create an MCL model with common settings.

    Args:
        model_name_or_path: Model identifier or path
        num_hyps: Number of hypotheses
        wta_training_mode: WTA training mode
        use_group_lora: Whether to use GroupLinear
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        target_modules: LoRA target modules
        **model_kwargs: Additional model arguments

    Returns:
        MCL-enabled model
    """
    lora_config.use_group_lora = use_group_lora

    # Create MCL configuration
    mcl_config = MCLConfig(
        num_hyps=num_hyps,
        wta_training_mode=wta_training_mode,
        wta_params_epsilon=wta_params_epsilon,
        wta_params_ini_temp=wta_params_ini_temp,
        wta_params_fin_temp=wta_params_fin_temp,
        wta_params_decay_rate=wta_params_decay_rate,
        wta_params_schedule_mode=wta_params_schedule_mode,
    )

    # Create wrapper and apply to model
    wrapper = MCLModelWrapper(lora_config, mcl_config)
    return wrapper.wrap_model(model_name_or_path, **model_kwargs)
