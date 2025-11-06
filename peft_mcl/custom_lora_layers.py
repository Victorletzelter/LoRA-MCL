"""
Custom LoRA layers for MCL (Multiple Choice Learning).
These can be registered with PEFT without modifying the source code.
"""

import warnings
import torch
import torch.nn as nn
from peft.tuners.lora.layer import LoraLayer
from typing import Any
import math


class GroupLinear(nn.Module, LoraLayer):
    """
    LoRA layer implemented with group convolutions to handle multiple hypotheses efficiently.

    This is the core MCL layer that enables efficient parallel processing of multiple hypotheses
    using group convolutions.

    This is a custom PEFT-compatible layer that can be registered using:
    lora_config._register_custom_module({torch.nn.Linear: GroupLinear})
    """

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        num_hyps: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_lora_weights: bool = True,
        **kwargs,
    ):
        """
        Initialize the GroupLinear layer.

        Args:
            base_layer: The base layer to replace with LoRA
            adapter_name: The name of the adapter (in the format of "lora<hypothesis_idx>")
            num_hyps: Number of hypotheses for MCL
            r: Rank of the LoRA parameters
            lora_alpha: Alpha parameter for the LoRA parameters
            lora_dropout: Dropout probability for the LoRA parameters
            fan_in_fan_out: Whether the layer stores weight in (fan_in, fan_out) format
            init_lora_weights: Whether to initialize the LoRA parameters
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        LoraLayer.__init__(self, base_layer=base_layer)

        self.fan_in_fan_out = fan_in_fan_out

        # Track whether we're using group convolutions yet
        self.initialized = False

        self.max_hyps = num_hyps

        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()

        # Set the active adapter
        self._active_adapter = adapter_name

        # Initialize this adapter
        self.update_layer(
            adapter_name=adapter_name,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
        )

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, **kwargs):
        """Initialize or update an adapter"""
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        # Store adapter parameters
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha

        # Setup dropout
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        # Calculate scaling
        self.scaling[adapter_name] = lora_alpha / r

        # Initialize group convolutions if this is the first adapter
        if not self.initialized:
            # Create shared group convolution layers
            device = self.base_layer.weight.device
            dtype = self.base_layer.weight.dtype

            self.lora_A["lora0"] = nn.Conv1d(
                in_channels=self.max_hyps * self.in_features,
                out_channels=self.max_hyps * r,
                kernel_size=1,
                groups=self.max_hyps,
                bias=False,
            ).to(device=device, dtype=dtype)

            self.lora_B["lora0"] = nn.Conv1d(
                in_channels=self.max_hyps * r,
                out_channels=self.max_hyps * self.out_features,
                kernel_size=1,
                groups=self.max_hyps,
                bias=False,
            ).to(device=device, dtype=dtype)

            self.initialized = True

        # Initialize all adapters weights
        if init_lora_weights:
            for k in range(self.max_hyps):
                # Initialize first A layer with small random values
                A_slice = self.lora_A["lora0"].weight[k * r : (k + 1) * r]
                nn.init.kaiming_uniform_(A_slice, a=math.sqrt(5))

                # Initialize B layer with zeros
                B_slice = self.lora_B["lora0"].weight[
                    k * self.out_features : (k + 1) * self.out_features
                ]
                nn.init.zeros_(B_slice)

    def get_delta_weight(self, adapter):
        """Compute the delta weight for the given adapter"""
        raise NotImplementedError("Get delta weight is not implemented yet for MCL wrapper")

    def merge(self, safe_merge=False, adapter_names=None):
        """Merge the active adapter weights into the base weights"""
        raise NotImplementedError("Merge is not implemented yet for MCL wrapper")

    def unmerge(self):
        """Unmerge adapters from the base weights"""
        raise NotImplementedError("Unmerge is not implemented yet for MCL wrapper")

    def forward(self, x, *args, **kwargs):
        """Forward pass using group convolutions for efficiency"""
        # If the layer is merged or disabled, return the base layer's output
        if getattr(self, "merged", False) or getattr(self, "disable_adapters", False):
            return self.base_layer(x, *args, **kwargs)

        # Forward pass of the base layer
        result = self.base_layer(
            x, *args, **kwargs
        )  # in training: shape [batch*adapter_count, out_features, seq], in inference: shape [batch, seq, out_features]
        torch_result_dtype = result.dtype

        # If no active adapters, return base result
        if not self.active_adapters:
            return result

        # Get the active adapter
        active_adapter = self.active_adapters[0]  # Use first adapter in list

        # Get dropout layer for the active adapter
        dropout = self.lora_dropout["lora0"]  # We assume the dropout is the same for all adapters

        if len(self.active_adapters) > 1:  # during training
            # During training, process ALL adapters at once
            # Prepare input for group convolution
            x = x.to(self.lora_A["lora0"].weight.dtype)
            x = x.permute(0, 2, 1)  # [batch*adapter_count, in_features, seq]
            x = x.reshape(-1, self.max_hyps * self.in_features, x.shape[-1])
            batch_size = x.shape[0]
            if self.training is True:
                x = dropout(x)  # shape [batch, adapter_count*in_features, seq]

            # Forward through group convolutions - computes ALL adapters at once
            lora_A_out = self.lora_A["lora0"](x)  # [batch, r*adapter_count, seq]
            lora_out = self.lora_B["lora0"](lora_A_out)  # [batch, out_features*adapter_count, seq]
            lora_out = lora_out.reshape(batch_size * self.max_hyps, self.out_features, x.shape[-1])
            lora_out = lora_out.permute(0, 2, 1)  # [batch*adapter_count, seq, out_features]
            # Add LoRA output
            result = result + lora_out.to(
                torch_result_dtype
            )  # [batch*adapter_count, seq, out_features]

        elif len(self.active_adapters) == 1:  # during inference
            # Inference mode - just process the active adapter
            active_idx = int(active_adapter.replace("lora", ""))

            r = self.r["lora0"]  # We assume the rank is the same for all adapters
            out_features = self.out_features

            x = x.to(self.lora_A["lora0"].weight.dtype)
            if self.training is True:
                x = dropout(x)

            # Get the specific slices for this adapter
            A_slice = (
                self.lora_A["lora0"].weight[active_idx * r : (active_idx + 1) * r].squeeze(-1)
            )  # [r, in_features]
            B_slice = (
                self.lora_B["lora0"]
                .weight[active_idx * out_features : (active_idx + 1) * out_features]
                .squeeze(-1)
            )  # [out_features, r]

            # Compute the LoRA adjustment
            lora_output = (x @ A_slice.T @ B_slice.T) * self.scaling[
                "lora0"
            ]  # We assume the scaling is the same for all adapters
            result = result + lora_output.to(torch_result_dtype)

        return result


def register_mcl_custom_layers(lora_config, num_hyps: int):
    """
    Register MCL custom layers with a PEFT LoraConfig.

    Usage:
        from peft_mcl.custom_lora_layers import register_mcl_custom_layers

        lora_config = LoraConfig(...)
        register_mcl_custom_layers(lora_config, num_hyps=3)

    Args:
        lora_config: PEFT LoraConfig instance
        num_hyps: Number of hypotheses for MCL
    """

    # Create a wrapper that injects num_hyps
    class GroupLinearWithNumHyps(GroupLinear):
        def __init__(self, base_layer, adapter_name, **kwargs):
            if "num_hyps" in kwargs:
                assert kwargs.pop("num_hyps") == num_hyps, "Mismatch in number of hypotheses"
            super().__init__(base_layer, adapter_name, num_hyps=num_hyps, **kwargs)

    # Register with PEFT
    lora_config._register_custom_module({torch.nn.Linear: GroupLinearWithNumHyps})

    # Store flag for use_group_lora
    lora_config.use_group_lora = True

    return lora_config
