"""
Configuration classes for MCL (Multiple Choice Learning).
"""

from dataclasses import dataclass, field
from typing import Optional, Union, List, Literal, Tuple
from peft.config import PeftConfig
from peft.utils import PeftType
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from peft.tuners.lora.layer import LoraLayer
import inspect
from typing import Dict, Any
from transformers.cache_utils import Cache
from collections import OrderedDict
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MCLModelOutput(OrderedDict):
    """Simple wrapper to preserve all_losses attribute in model outputs."""

    def __init__(self, base_output, all_losses=None):
        super().__init__()
        self._base_output = base_output
        self.all_losses = all_losses

        # If base_output is already a dict-like object, copy its contents
        if hasattr(base_output, "items"):
            for key, value in base_output.items():
                if value is not None:
                    self[key] = value
        else:
            # Otherwise, copy attributes manually
            for attr in [
                "logits",
                "loss",
                "hidden_states",
                "attentions",
                "past_key_values",
                "attention_mask",
            ]:
                if hasattr(base_output, attr):
                    value = getattr(base_output, attr)
                    if value is not None:
                        self[attr] = value

    def __getattr__(self, name):
        return getattr(self._base_output, name)

    def __getitem__(self, key):
        if isinstance(key, str):
            return super().__getitem__(key)
        else:
            return tuple(self.values())[key]


@dataclass
class LoraRuntimeConfig:
    """Runtime configuration for LoRA"""

    ephemeral_gpu_offload: bool = field(default=False)


@dataclass
class MCLLoraConfig(PeftConfig):
    """
    Configuration class for MCL (Multiple Choice Learning) with LoRA.
    Extends standard LoraConfig with MCL-specific parameters.
    """

    # Standard LoRA parameters (copied from PEFT's LoraConfig)
    r: int = field(default=8, metadata={"help": "LoRA attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with LoRA"
        },
    )
    exclude_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to exclude from LoRA"
        },
    )
    lora_alpha: int = field(default=8, metadata={"help": "LoRA alpha parameter"})
    lora_dropout: float = field(default=0.0, metadata={"help": "LoRA dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={
            "help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"
        },
    )
    bias: Literal["none", "all", "lora_only"] = field(
        default="none", metadata={"help": "Bias type for LoRA. Can be 'none', 'all' or 'lora_only'"}
    )
    use_rslora: bool = field(
        default=False, metadata={"help": "Whether to use Rank-Stabilized LoRA"}
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint"
        },
    )
    init_lora_weights: Union[bool, str] = field(
        default=True, metadata={"help": "How to initialize the weights of the adapter layers"}
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(
        default=None, metadata={"help": "The layer indices to transform"}
    )
    layers_pattern: Optional[Union[List[str], str]] = field(
        default=None, metadata={"help": "The layer pattern name"}
    )
    rank_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={"help": "The mapping from layer names or regexp expression to ranks"},
    )
    alpha_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={"help": "The mapping from layer names or regexp expression to alphas"},
    )
    megatron_config: Optional[dict] = field(default=None)
    megatron_core: Optional[str] = field(default="megatron.core")
    loftq_config: Optional[dict] = field(default_factory=dict)
    eva_config: Optional[dict] = field(default=None)
    use_dora: bool = field(default=False, metadata={"help": "Enable DoRA"})
    layer_replication: Optional[List[tuple]] = field(
        default=None, metadata={"help": "Layer replication configuration"}
    )
    runtime_config: LoraRuntimeConfig = field(
        default_factory=LoraRuntimeConfig, metadata={"help": "Runtime configurations"}
    )
    lora_bias: bool = field(
        default=False, metadata={"help": "Whether to enable the bias term for the LoRA B parameter"}
    )
    _custom_modules: Optional[dict] = field(
        default=None, metadata={"help": "Custom modules for LoRA"}
    )

    # MCL-specific parameters
    num_hyps: int = field(default=1, metadata={"help": "Number of hypotheses for MCL"})
    use_group_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use GroupLinear for efficient multi-hypothesis training"},
    )

    # WTA (Winner-Take-All) parameters
    wta_training_mode: str = field(
        default="wta",
        metadata={"help": "WTA training mode: 'wta', 'relaxed-wta', 'annealed-wta', etc."},
    )
    wta_params_epsilon: float = field(
        default=0.0, metadata={"help": "Epsilon parameter for relaxed WTA"}
    )
    wta_params_ini_temp: float = field(
        default=1.0, metadata={"help": "Initial temperature for annealed WTA"}
    )
    wta_params_fin_temp: float = field(
        default=0.01, metadata={"help": "Final temperature for annealed WTA"}
    )
    wta_params_decay_rate: float = field(default=0.999, metadata={"help": "Temperature decay rate"})
    wta_params_schedule_mode: str = field(
        default="global_step", metadata={"help": "Temperature schedule mode"}
    )

    def __post_init__(self):
        self.peft_type = PeftType.LORA
        self.task_type = "CAUSAL_LM"

        # Validate MCL parameters
        if self.num_hyps < 1:
            raise ValueError("num_hyps must be >= 1")

    def to_dict(self):
        """Returns the configuration as a dictionary, removing runtime configurations"""
        rv = super().to_dict()
        rv.pop("runtime_config", None)
        return rv

    def _register_custom_module(self, mapping: dict[type[nn.Module], type[nn.Module]]) -> None:
        if self._custom_modules is None:
            self._custom_modules = {}
        self._custom_modules.update(mapping)


@dataclass
class MCLModelConfig:
    """
    Configuration for MCL model wrapper.
    This gets merged into the base model's config.
    """

    # MCL parameters that need to be added to model config
    num_hyps: int = 1
    wta_training_mode: str = "wta"
    wta_params_epsilon: float = 0.0
    wta_params_ini_temp: float = 1.0
    wta_params_fin_temp: float = 0.01
    wta_params_decay_rate: float = 0.999
    wta_params_schedule_mode: str = "global_step"
    native_group_lora_enabled: bool = False


class MCLForwardMixin(ABC):
    """
    Mixin class that provides MCL forward functionality.
    Can be mixed into any ForConditionalGeneration class.
    """

    def __init__(self, *args, **kwargs):

        # MCL parameters should be set by the wrapper
        self.num_hyps = kwargs.pop("num_hyps", 1)
        self.wta_training_mode = kwargs.pop("wta_training_mode", "wta")
        self.wta_params_epsilon = kwargs.pop("wta_params_epsilon", 0.0)
        self.wta_params_ini_temp = kwargs.pop("wta_params_ini_temp", 1.0)
        self.wta_params_fin_temp = kwargs.pop("wta_params_fin_temp", 0.01)
        self.wta_params_decay_rate = kwargs.pop("wta_params_decay_rate", 0.999)
        self.wta_params_schedule_mode = kwargs.pop("wta_params_schedule_mode", "global_step")
        self.temperature = self.wta_params_ini_temp
        self.native_group_lora_enabled = kwargs.pop("native_group_lora_enabled", False)
        super().__init__(*args, **kwargs)

    def _set_adapter(self, adapter_name):
        """
        Set the active adapter for all LoRA layers in the model.

        Args:
            adapter_name: Name of the adapter to activate (e.g., 'lora0', 'lora1', etc.)
        """

        # Set adapter for all LoRA layers
        for module in self.modules():
            if isinstance(module, LoraLayer):
                module.set_adapter(adapter_name)

        # Verify that all layers have the correct active adapter
        for module in self.modules():
            if isinstance(module, LoraLayer):
                assert module.active_adapters == [
                    adapter_name
                ], f"Expected adapter {adapter_name} but got {module.active_adapters}"

    def _filter_model_kwargs(self, kwargs):
        """
        Filter kwargs to only include valid model inputs.
        Removes trainer-specific arguments like num_items_in_batch.
        """
        # Common model input names
        valid_keys = {
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "position_ids",
            "head_mask",
            "inputs_embeds",
            "labels",
            "output_attentions",
            "output_hidden_states",
            "return_dict",
            "past_key_values",
            "use_cache",
            "encoder_hidden_states",
            "encoder_attention_mask",
            "input_features",
            "feature_attention_mask",
        }

        return {k: v for k, v in kwargs.items() if k in valid_keys}

    def _compute_per_sample_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample loss for causal language modeling.

        Args:
            logits: Model logits of shape (batch_size, seq_len, vocab_size)
            labels: Target labels of shape (batch_size, seq_len)

        Returns:
            Per-sample losses of shape (batch_size,)
        """
        # Shift logits and labels for causal LM (predict next token)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        # Compute loss with reduction='none' to get per-token losses
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fct(shift_logits, shift_labels)

        # Reshape to (batch_size, seq_len-1)
        batch_size = labels.shape[0]
        seq_len = labels.shape[1] - 1
        per_token_loss = per_token_loss.view(batch_size, seq_len)

        # Mask out padding tokens (label == -100)
        shift_labels_reshaped = labels[..., 1:].contiguous()
        mask = (shift_labels_reshaped != -100).float()

        # Compute average loss per sample (only over non-padded tokens)
        per_sample_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return per_sample_loss

    def model_temperature(
        self, global_step: Optional[int] = None, epoch_number: Optional[int] = None
    ):
        """Compute temperature for annealed WTA"""
        if self.wta_params_schedule_mode == "global_step" and global_step is not None:
            temperature = self.wta_params_fin_temp + (
                self.wta_params_ini_temp - self.wta_params_fin_temp
            ) * (self.wta_params_decay_rate**global_step)
        elif self.wta_params_schedule_mode == "epoch" and epoch_number is not None:
            temperature = self.wta_params_fin_temp + (
                self.wta_params_ini_temp - self.wta_params_fin_temp
            ) * (self.wta_params_decay_rate**epoch_number)
        else:
            temperature = self.temperature
        return temperature

    def compute_wta_loss(
        self,
        loss_tensor: torch.Tensor,
        wta_training_mode: str,
        global_step: Optional[int] = None,
        epoch_number: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute WTA loss from stacked losses.

        Args:
            loss_tensor: Tensor of shape (num_hypotheses, batch_size)
            wta_training_mode: WTA mode string
            global_step: Current global step
            epoch_number: Current epoch number

        Returns:
            Scalar loss tensor
        """
        if wta_training_mode == "wta":
            loss = torch.min(loss_tensor, dim=0).values.mean()
        elif wta_training_mode == "random-wta":
            batch_size = loss_tensor.shape[1]
            idx_winner = torch.randint(0, self.num_hyps, (batch_size,), device=loss_tensor.device)
            loss = loss_tensor[
                idx_winner, torch.arange(batch_size, device=loss_tensor.device)
            ].mean()
        elif wta_training_mode == "relaxed-wta":
            wta_loss = torch.min(loss_tensor, dim=0).values  # shape (batch_size)
            epsilon = self.wta_params_epsilon
            loss = (1 - epsilon - epsilon / (self.num_hyps - 1)) * wta_loss + (
                epsilon / (self.num_hyps - 1)
            ) * loss_tensor.sum(dim=0)
            loss = loss.mean()
        elif wta_training_mode == "annealed-wta":
            temperature = self.model_temperature(global_step=global_step, epoch_number=epoch_number)
            if temperature <= self.wta_params_fin_temp:
                loss = torch.min(loss_tensor, dim=0).values.mean()
            else:
                weights = torch.exp(-loss_tensor / temperature)
                weights = weights / weights.sum(dim=0, keepdim=True)
                weights = weights.detach()
                loss = (loss_tensor * weights).sum(dim=0).mean()
        else:
            raise ValueError(f"Invalid wta_training_mode: {wta_training_mode}")

        return loss

    @abstractmethod
    def _original_forward(self, *args, **kwargs):
        """
        This should call the original forward method of the base class.
        Must be implemented by the concrete wrapper class.
        """
        pass

    def forward(
        self,
        hypothesis_idx: int = None,
        return_all_hyps: Optional[bool] = False,
        return_all_losses: Optional[bool] = False,
        global_step: Optional[int] = None,
        epoch_number: Optional[int] = None,
        wta_training_mode: Optional[str] = None,
        num_items_in_batch: Optional[int] = None,
        **kwargs,
    ):
        """
        MCL forward pass that handles multiple hypotheses and WTA loss computation.
        """
        # Use provided wta_training_mode or fall back to config
        wta_mode = wta_training_mode if wta_training_mode is not None else self.wta_training_mode

        # Multi-hypothesis case
        if not self.native_group_lora_enabled:
            # Standard multi-hypothesis: iterate through hypotheses
            return self._multi_hypothesis_forward_standard(
                wta_mode=wta_mode,
                global_step=global_step,
                epoch_number=epoch_number,
                hypothesis_idx=hypothesis_idx,
                return_all_hyps=return_all_hyps,
                return_all_losses=return_all_losses,
                **kwargs,
            )
        else:
            # Group LoRA: efficient batched computation
            return self._multi_hypothesis_forward_group_lora(
                wta_mode=wta_mode,
                global_step=global_step,
                epoch_number=epoch_number,
                hypothesis_idx=hypothesis_idx,
                return_all_hyps=return_all_hyps,
                return_all_losses=return_all_losses,
                **kwargs,
            )

    def _multi_hypothesis_forward_standard(
        self, wta_mode, global_step, epoch_number, hypothesis_idx, return_all_losses, **kwargs
    ):
        """Standard multi-hypothesis forward (iterate through adapters)"""

        if hypothesis_idx is None:
            list_hyps = range(self.num_hyps)
        else:
            list_hyps = [hypothesis_idx]

        labels = kwargs.get("labels")

        batch_size = kwargs.get("input_ids").shape[0]
        loss_list = []
        all_outputs = []

        # Filter and remove labels from kwargs to get logits only
        filtered_kwargs = self._filter_model_kwargs(kwargs)
        kwargs_no_labels = {k: v for k, v in filtered_kwargs.items() if k != "labels"}

        # Iterate through hypotheses
        for hyp_idx in list_hyps:
            # Set the appropriate adapter if using PEFT
            if hasattr(self, "_set_adapter"):
                self._set_adapter(f"lora{hyp_idx}")

            # Forward pass WITHOUT computing loss (get logits only)
            outputs = self._original_forward(**kwargs_no_labels)

            # Compute per-sample loss manually
            logits = outputs.logits

            if labels is not None:
                # Compute loss with reduction='none' to get per-sample losses
                per_sample_loss = self._compute_per_sample_loss(logits=logits, labels=labels)
                loss_list.append(per_sample_loss)

                # Stack losses and compute WTA loss
                loss_tensor = torch.stack(loss_list, dim=0)  # (num_hyps, batch_size)

                if return_all_losses:
                    return loss_tensor

                final_loss = self.compute_wta_loss(
                    loss_tensor=loss_tensor,
                    wta_training_mode=wta_mode,
                    global_step=global_step,
                    epoch_number=epoch_number,
                )

                # Create MCL output wrapper with all_losses preserved
                outputs = MCLModelOutput(outputs, all_losses=loss_tensor)
                outputs.loss = final_loss
                outputs["loss"] = final_loss
                outputs["all_losses"] = loss_tensor
            else:
                final_loss = None
                outputs = MCLModelOutput(outputs, all_losses=None)
                outputs.loss = final_loss
                outputs["loss"] = final_loss
                outputs["all_losses"] = None

        return outputs

    def _multi_hypothesis_forward_group_lora(
        self, wta_mode, global_step, epoch_number, hypothesis_idx, return_all_losses, **kwargs
    ):
        """Group LoRA multi-hypothesis forward (batched computation)"""
        labels = kwargs.get("labels")
        original_batch_size = kwargs.get("input_ids").shape[0]

        if hypothesis_idx is None:
            list_hyps = range(self.num_hyps)  # Training mode
        else:
            list_hyps = [hypothesis_idx]  # Generation (or single hypothesis training) mode

        if len(list_hyps) == 1:
            adapter_name = f"lora{list_hyps[0]}"  # Generation (or single hypothesis training) mode
            # Load the adapter
            self._set_adapter(adapter_name)

        if labels is None:
            filtered_kwargs = self._filter_model_kwargs(kwargs)
            return self._original_forward(**filtered_kwargs)

        # Filter valid model inputs first
        filtered_kwargs = self._filter_model_kwargs(kwargs)
        logger.info(f"[Memory] After filtering: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        # Expand inputs for all hypotheses (excluding labels)
        expanded_kwargs = {}
        for key, value in filtered_kwargs.items():
            if key == "labels":
                continue  # Don't expand labels yet
            if key == "input_features":
                expanded_value = value.unsqueeze(1).expand(-1, len(list_hyps), -1, -1)
                expanded_value = expanded_value.reshape(-1, *expanded_value.shape[2:])
                expanded_kwargs[key] = expanded_value
            elif (
                isinstance(value, torch.Tensor)
                and value.dim() > 0
                and key in ["input_ids", "attention_mask", "feature_attention_mask"]
            ):
                # Repeat for each hypothesis
                expanded_value = value.unsqueeze(1).expand(-1, len(list_hyps), -1)
                expanded_value = expanded_value.reshape(-1, *value.shape[1:])
                expanded_kwargs[key] = expanded_value

            else:
                expanded_kwargs[key] = value

        logger.info(f"[Memory] After expansion: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        # Single forward pass with batched hypotheses (without labels to get logits)
        outputs = self._original_forward(**expanded_kwargs)

        logger.info(f"[Memory] After forward: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        # Get logits and reshape: (num_hyps * batch_size, seq_len, vocab_size) -> (num_hyps, batch_size, seq_len, vocab_size)
        logits = outputs.logits
        seq_len = logits.shape[1]
        vocab_size = logits.shape[2]
        logits = logits.view(original_batch_size, len(list_hyps), seq_len, vocab_size)

        if labels is not None:
            # Compute per-sample loss for each hypothesis
            loss_list = []
            for hyp_idx in range(len(list_hyps)):
                hyp_logits = logits[:, hyp_idx, :, :]  # (batch_size, seq_len, vocab_size)
                per_sample_loss = self._compute_per_sample_loss(hyp_logits, labels)
                loss_list.append(per_sample_loss)

            # Stack losses to (num_hyps, batch_size)
            loss_tensor = torch.stack(loss_list, dim=0)

            if return_all_losses:
                return loss_tensor

            final_loss = self.compute_wta_loss(
                loss_tensor=loss_tensor,
                wta_training_mode=wta_mode,
                global_step=global_step,
                epoch_number=epoch_number,
            )
            outputs = MCLModelOutput(outputs, all_losses=loss_tensor)
            outputs.loss = final_loss
            outputs["loss"] = final_loss
            outputs["all_losses"] = loss_tensor
        else:
            outputs = MCLModelOutput(outputs, all_losses=None)
            outputs.loss = None
            outputs["loss"] = None
            outputs["all_losses"] = None

        return outputs

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        """Validate model kwargs is not supported in this version of the MCL wrapper."""
        pass
