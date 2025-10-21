"""
MCL Forward Method Mixin

Provides the MCL forward functionality that can be mixed into any 
ForConditionalGeneration class.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, List
from abc import ABC, abstractmethod
import os

from typing import Type, Optional
from transformers import PreTrainedModel, AutoConfig, AutoModelForCausalLM
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
import inspect
import logging
from peft.tuners.lora.layer import LoraLayer
from peft import LoraConfig
from .forward_mixin import MCLLoraConfig, MCLModelConfig, MCLForwardMixin
from .custom_lora_layers import register_mcl_custom_layers

logger = logging.getLogger(__name__)

def create_mcl_model_class(base_model_class: Type[PreTrainedModel], 
                          model_name_suffix: str = "_MCL") -> Type[PreTrainedModel]:
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
                if hasattr(cls, 'forward') and cls != MCLForwardMixin:
                    return cls.forward(self, *args, **kwargs)
            raise RuntimeError("Could not find original forward method")
    
    # Set the class name
    MCLModel.__name__ = base_model_class.__name__ + model_name_suffix
    MCLModel.__qualname__ = base_model_class.__qualname__ + model_name_suffix
    
    return MCLModel

def apply_mcl_to_config(config, mcl_config: MCLModelConfig):
    """Apply MCL configuration to a model config"""
    config.num_hyps = mcl_config.num_hyps
    config.wta_training_mode = mcl_config.wta_training_mode
    config.wta_params_epsilon = mcl_config.wta_params_epsilon
    config.wta_params_ini_temp = mcl_config.wta_params_ini_temp
    config.wta_params_fin_temp = mcl_config.wta_params_fin_temp
    config.wta_params_decay_rate = mcl_config.wta_params_decay_rate
    config.wta_params_schedule_mode = mcl_config.wta_params_schedule_mode
    return config

class MCLModelWrapper:
    """
    General wrapper that can transform any Hugging Face model into its MCL version.
    """
    def __init__(self, mcl_lora_config: Optional[MCLLoraConfig] = None, 
                 mcl_model_config: Optional[MCLModelConfig] = None):
        """Initializes the MCLModelWrapper.
        
        Args:
            mcl_lora_config (Optional[MCLLoraConfig], optional): The LoRA configuration for the MCL model. Defaults to None.
            mcl_model_config (Optional[MCLModelConfig], optional): The model configuration for the MCL model. Defaults to None.
        """
        self.mcl_lora_config = mcl_lora_config
        self.mcl_model_config = mcl_model_config or MCLModelConfig()
        self._mcl_classes = {}  # Cache for created MCL classes
    
    def wrap_model(self, pretrained_repo: str, model_name_or_path: str, loading_kwargs: dict = {}, **model_kwargs) -> PreTrainedModel:
        """
        Create an MCL version of any Hugging Face model.
        
        Args:
            model_name_or_path: Model identifier or path
            **model_kwargs: Additional arguments for model loading
            
        Returns:
            MCL-enabled model instance
        """
        # Load the original config
        config = AutoConfig.from_pretrained(pretrained_repo, **model_kwargs)

        if "max_source_positions" in model_kwargs: # Specfic case of Qwen2-Audio model, where the max source positions are provided as a model_kwargs.
            config.audio_config.max_source_positions = model_kwargs.pop("max_source_positions")

        # Apply MCL configuration
        config = apply_mcl_to_config(config, self.mcl_model_config)
        
        # Get the original model class
        base_model_class = self._get_model_class_for_config(config)
        
        # Create or get cached MCL class
        mcl_class_key = base_model_class.__name__
        if mcl_class_key not in self._mcl_classes:
            self._mcl_classes[mcl_class_key] = create_mcl_model_class(base_model_class)
        
        mcl_model_class = self._mcl_classes[mcl_class_key]

        if len(loading_kwargs) == 0: # If no loading kwargs are provided, use low cpu mem usage and bfloat16 dtype by default
            loading_kwargs = {
                "low_cpu_mem_usage": True,
                "torch_dtype": torch.bfloat16,
            }
        
        # Create the MCL model instance
        model = mcl_model_class.from_pretrained(pretrained_repo, 
                                                config=config, 
                                                **loading_kwargs,
                                                **model_kwargs)
        
        # Apply LoRA if specified
        if self.mcl_lora_config:
            model = self._apply_lora(model)

        if pretrained_repo != model_name_or_path: # In this case, we can to load LoRA adapters from the model_name_or_path
            # We are assume here the model_name_or_path is a local path, and the LoRA adapters are in the model_name_or_path/lora0, model_name_or_path/lora1, ...
            num_hyps_to_load = model_kwargs['num_hyps'] if 'native_group_lora_enabled' not in model_kwargs or model_kwargs['native_group_lora_enabled'] is False else 1
            # When use group LoRA, all the adapters are in lora0 (the GroupLinear adapter), so in this case num_hyps_to_load is 1.
            for k in range(num_hyps_to_load):
                model.load_adapter(os.path.join(model_name_or_path, f"lora{k}"), adapter_name=f"lora{k}", torch_device="cuda:0")
            
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
        model.config = apply_mcl_to_config(model.config, self.mcl_model_config)

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
        if self.mcl_lora_config:
            model = self._apply_lora(model)
        
        return model
    
    def _get_model_class_for_config(self, config):
        """Get the appropriate model class for a config"""
        # Import the additional mapping
        from transformers.models.auto.modeling_auto import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
        
        # Try to get from the causal LM mapping first
        for config_class, model_class in MODEL_FOR_CAUSAL_LM_MAPPING.items():
            if isinstance(config, config_class):
                return model_class
        
        # Try to get from the seq2seq causal LM mapping
        for config_class, model_class in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.items():
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

        if self.mcl_lora_config.use_group_lora:
            register_mcl_custom_layers(self.mcl_lora_config, num_hyps=self.mcl_lora_config.num_hyps)
            model.config.native_group_lora_enabled = True
        
        # Apply multiple LoRA adapters (one for each hypothesis)
        # If use_group_lora is True, we only add one adapter (the GroupLinear adapter which contains all the hypotheses)
        num_adapter_to_add = self.mcl_lora_config.num_hyps if self.mcl_lora_config.use_group_lora is False else 1
        for hyp_idx in range(num_adapter_to_add):
            adapter_name = f"lora{hyp_idx}"
            if hyp_idx == 0:
                model = get_peft_model(model, self.mcl_lora_config, adapter_name=adapter_name)
            elif hyp_idx > 0 and self.mcl_lora_config.use_group_lora is False:
                model.add_adapter(adapter_name, self.mcl_lora_config)
                
        if self.mcl_lora_config.use_group_lora is True:
            all_adapters = [f"lora{i}" for i in range(self.mcl_lora_config.num_hyps)]
            for module in model.modules():
                if hasattr(module, 'set_adapter') and hasattr(module, 'max_hyps'):
                    # This is a GroupLinear layer, enabled all adapters for training
                    module.set_adapter(all_adapters)
        
        return model

# Convenience function
def get_peft_mcl(model_name_or_path: str, 
                     lora_config: Optional[LoraConfig] = None,
                     num_hyps: int = 3,
                     wta_training_mode: str = "wta",
                     use_group_lora: bool = False,
                     generation_config = None,
                     loading_kwargs: dict = {},
                     pretrained_repo: str = None,
                     **model_kwargs) -> PreTrainedModel:
    """
    Convenience function to create an MCL model with common settings.
    
    Args:
        model_name_or_path: Model identifier or path
        num_hyps: Number of hypotheses
        wta_training_mode: WTA training mode
        use_group_lora: Whether to use GroupLinear
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        target_modules: LoRA target modules^
        loading_kwargs: Loading kwargs for the model which are passed to the from_pretrained method. These include in particular:
        - low_cpu_mem_usage: Whether to use low cpu mem usage (default: True)
        - torch_dtype: The dtype to load the model with (default: bfloat16)
        - ... (any other kwargs supported by the from_pretrained method from the PreTrainedModel class)
        **model_kwargs: Additional model arguments
        
    Returns:
        MCL-enabled model
    """

    if pretrained_repo is None:
        pretrained_repo = model_name_or_path # If no pretrained repo is provided, use the model name or path as the pretrained repo
        # This is okay to not provide a pretrained repo, except when model_name_or_path is a local path, as it may raise the Unrecognized model error.

    if use_group_lora is True:
        register_mcl_custom_layers(lora_config, num_hyps=num_hyps)

    # Create configurations
    mcl_lora_config = MCLLoraConfig(
        num_hyps=num_hyps,
        use_group_lora=use_group_lora,
        wta_training_mode=wta_training_mode,
        **lora_config.to_dict()
    )
    
    mcl_model_config = MCLModelConfig(
        num_hyps=num_hyps,
        wta_training_mode=wta_training_mode,
        native_group_lora_enabled=use_group_lora
    )

    model_kwargs['num_hyps'] = num_hyps
    model_kwargs['wta_training_mode'] = wta_training_mode
    model_kwargs['native_group_lora_enabled'] = use_group_lora
    
    # Create wrapper and apply to model
    wrapper = MCLModelWrapper(mcl_lora_config=mcl_lora_config, 
                              mcl_model_config=mcl_model_config)
    model = wrapper.wrap_model(pretrained_repo=pretrained_repo, model_name_or_path=model_name_or_path, loading_kwargs=loading_kwargs, **model_kwargs)
    if generation_config: # Set the generation config if provided
        model.generation_config = generation_config
    return model