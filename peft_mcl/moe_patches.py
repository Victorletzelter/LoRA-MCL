"""
MoE LoRA patches and layer implementations.

This module contains all code related to Mixture-of-Experts LoRA, which is a baseline method
(not part of the core MCL approach). Import this module only if you want to use MoE LoRA.

Usage:
    from peft_mcl.moe_patches import patch_peft_for_moe, MoELoraLinear

    # Enable MoE support
    patch_peft_for_moe(enable=True)

    # Create your model with use_moe_lora=True
    lora_config = LoraConfig(use_moe_lora=True, num_experts=4, ...)
    model = get_peft_model(base_model, lora_config)
"""

import os
import math
import warnings
import torch
import torch.nn as nn
from typing import Optional
from peft.tuners.lora.layer import LoraLayer


class MoELoraLinear(nn.Module, LoraLayer):
    """
    Mixture-of-Experts LoRA Linear layer with a router that selects among k LoRA experts.
    Each expert is a LoRA adapter (A, B), and the router outputs a softmax distribution over experts per token.

    This is a baseline method for comparison, not the core MCL approach.
    """

    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", "router")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        num_experts: int,
        init_zero_router: bool = False,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_lora_weights: bool = True,
        stochastic_router=None,
        expert_specific=None,
        fast_version: bool = False,
        sparse_moe_enabled: bool = False,
        top_k_sparse_moe: int = 1,
        **kwargs,
    ):
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self.num_experts = num_experts
        self._active_adapter = adapter_name
        self.fast_version = fast_version
        self.sparse_moe_enabled = sparse_moe_enabled
        self.top_k_sparse_moe = top_k_sparse_moe

        # Router: projects input to logits over experts
        self.router = nn.Linear(self.in_features, num_experts)

        # Initialize adapter parameters
        self.update_layer(
            adapter_name=adapter_name,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
        )

        # Initialize router weights
        if init_lora_weights:
            if init_zero_router:
                nn.init.zeros_(self.router.weight)
                if self.router.bias is not None:
                    nn.init.zeros_(self.router.bias)
            else:
                nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
                if self.router.bias is not None:
                    nn.init.zeros_(self.router.bias)

        self.expert_specific = expert_specific
        self.stochastic_router = stochastic_router
        self._last_router_logits = None

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, **kwargs):
        """Initialize or update an adapter"""
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        # Store adapter parameters
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.scaling[adapter_name] = lora_alpha / r

        # Setup dropout
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        # Create k LoRA adapters with proper naming
        self.lora_A[adapter_name] = nn.ModuleDict(
            {
                f"expert{i}": nn.Linear(self.in_features, r, bias=False)
                for i in range(self.num_experts)
            }
        )
        self.lora_B[adapter_name] = nn.ModuleDict(
            {
                f"expert{i}": nn.Linear(r, self.out_features, bias=False)
                for i in range(self.num_experts)
            }
        )

        # After creating the layers, add explicit dtype/device casting
        device = self.base_layer.weight.device
        dtype = self.base_layer.weight.dtype

        for expert_A, expert_B in zip(
            self.lora_A[adapter_name].values(), self.lora_B[adapter_name].values()
        ):
            expert_A.to(device=device, dtype=dtype)
            expert_B.to(device=device, dtype=dtype)

        # Initialize weights for this adapter
        if init_lora_weights:
            for i in range(self.num_experts):
                nn.init.kaiming_uniform_(
                    self.lora_A[adapter_name][f"expert{i}"].weight, a=math.sqrt(5)
                )
                nn.init.zeros_(self.lora_B[adapter_name][f"expert{i}"].weight)

        # Set the active adapters
        if adapter_name not in self.active_adapters:
            self.active_adapters.append(adapter_name)

    def get_delta_weight(self, adapter):
        """Compute the delta weight for the given adapter"""
        device = self.lora_B[adapter]["expert0"].weight.device
        dtype = self.lora_A[adapter]["expert0"].weight.dtype

        # Compute average delta weight across all experts
        delta_weights = []
        for i in range(self.num_experts):
            A_weight = self.lora_A[adapter][f"expert{i}"].weight
            B_weight = self.lora_B[adapter][f"expert{i}"].weight

            # Matrix multiplication
            cast_to_fp32 = device.type == "cpu" and (
                dtype == torch.float16 or dtype == torch.bfloat16
            )
            if cast_to_fp32:
                A_weight = A_weight.float()
                B_weight = B_weight.float()

            delta = B_weight @ A_weight * self.scaling[adapter]

            if cast_to_fp32:
                delta = delta.to(dtype)

            delta_weights.append(delta)

        # Return average delta weight
        return torch.stack(delta_weights).mean(dim=0)

    def merge(self, safe_merge=False, adapter_names=None):
        """Merge the active adapter weights into the base weights"""
        if getattr(self, "merged", False):
            return

        adapter_names = adapter_names or self.active_adapters
        if adapter_names is None:
            return

        for adapter_name in adapter_names:
            if adapter_name in self.r:
                delta_weight = self.get_delta_weight(adapter_name)

                if safe_merge:
                    base_weight = self.base_layer.weight.data.clone()
                    base_weight += delta_weight

                    if not torch.isfinite(base_weight).all():
                        raise ValueError(f"NaNs detected when merging adapter {adapter_name}")

                    self.base_layer.weight.data = base_weight
                else:
                    self.base_layer.weight.data += delta_weight

                if not hasattr(self, "merged_adapters"):
                    self.merged_adapters = []
                self.merged_adapters.append(adapter_name)

        self.merged = True

    def unmerge(self):
        """Unmerge adapters from the base weights"""
        if not getattr(self, "merged", False):
            return

        for adapter_name in self.merged_adapters:
            delta_weight = self.get_delta_weight(adapter_name)
            self.base_layer.weight.data -= delta_weight

        self.merged_adapters = []
        self.merged = False

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        base_out = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = base_out.dtype

        if x.dim() == 2:
            x_ = x.unsqueeze(1)
        else:
            x_ = x

        if self.active_adapters is None:
            active_adapter = None
        else:
            active_idx = self.active_adapters[0].replace("lora", "")
            active_adapter = "lora0"

        x_ = x_.to(self.router.weight.dtype)

        self.expert_specific = str(self.expert_specific).lower()
        self.stochastic_router = str(self.stochastic_router).lower()

        if self.fast_version and not self.sparse_moe_enabled:
            if self.expert_specific != "true" and self.stochastic_router != "true":
                router_logits = self.router(x_)
                router_weights = torch.softmax(router_logits, dim=-1)

                lora_outputs = []
                for i in range(self.num_experts):
                    if self.training:
                        lora_a = self.lora_A[active_adapter][f"expert{i}"](
                            self.lora_dropout[active_adapter](x_)
                        )
                    else:
                        lora_a = self.lora_A[active_adapter][f"expert{i}"](x_)
                    lora_b = self.lora_B[active_adapter][f"expert{i}"](lora_a)
                    lora_outputs.append(lora_b)

                lora_outputs = torch.stack(lora_outputs, dim=-1)
                router_weights_exp = router_weights.unsqueeze(-2)
                moe_lora = (lora_outputs * router_weights_exp).sum(dim=-1) * self.scaling[
                    active_adapter
                ]

            elif self.expert_specific == "true" and self.stochastic_router != "true":
                if self.training:
                    lora_a = self.lora_A[active_adapter][f"expert{int(active_idx)}"](
                        self.lora_dropout[active_adapter](x_)
                    )
                else:
                    lora_a = self.lora_A[active_adapter][f"expert{int(active_idx)}"](x_)
                moe_lora = (
                    self.lora_B[active_adapter][f"expert{int(active_idx)}"](lora_a)
                    * self.scaling[active_adapter]
                )

            elif self.stochastic_router == "true" and self.expert_specific != "true":
                router_logits = self.router(x_)
                router_weights = torch.softmax(router_logits, dim=-1)

                batch_size, seq_len, num_experts = router_weights.shape
                router_weights_2d = router_weights.view(-1, num_experts)
                expert_choice_flat = torch.multinomial(router_weights_2d, 1, replacement=True)
                expert_choice = expert_choice_flat.view(batch_size, seq_len, 1)

                moe_lora = x_.new_zeros(batch_size, seq_len, self.out_features)
                choice = expert_choice.squeeze(-1)

                for i in range(self.num_experts):
                    mask = choice == i
                    if not mask.any():
                        continue
                    b_idx, s_idx = mask.nonzero(as_tuple=True)
                    x_sel = x_[b_idx, s_idx]
                    if self.training:
                        x_sel = self.lora_dropout[active_adapter](x_sel)
                    h = self.lora_A[active_adapter][f"expert{i}"](x_sel)
                    moe_lora[b_idx, s_idx] = self.lora_B[active_adapter][f"expert{i}"](h)

                moe_lora = moe_lora * self.scaling[active_adapter]

        elif self.sparse_moe_enabled:
            self._last_router_logits = None
            router_logits = self.router(x_)
            router_weights = torch.softmax(router_logits, dim=-1)

            router_weights, selected_experts = torch.topk(
                router_weights, self.top_k_sparse_moe, dim=-1
            )
            router_weights /= router_weights.sum(dim=-1, keepdim=True)

            B, S, d_in = x_.shape
            d_out = self.out_features

            weights_full = x_.new_zeros(B, S, self.num_experts)
            weights_full.scatter_(dim=2, index=selected_experts, src=router_weights)

            mask_per_expert = selected_experts.new_zeros(B, S, self.num_experts, dtype=torch.bool)
            mask_per_expert.scatter_(2, selected_experts, True)

            lora_outputs = x_.new_zeros(B, S, d_out, self.num_experts)

            for i in range(self.num_experts):
                b_idx, s_idx = torch.where(mask_per_expert[..., i])
                if b_idx.numel() == 0:
                    continue
                x_sel = x_[b_idx, s_idx]
                if self.training:
                    x_sel = self.lora_dropout[active_adapter](x_sel)
                out = self.lora_B[active_adapter][f"expert{i}"](
                    self.lora_A[active_adapter][f"expert{i}"](x_sel)
                )

                if out.dtype != lora_outputs.dtype:
                    out = out.to(lora_outputs.dtype)

                lora_outputs[b_idx, s_idx, :, i] = out

            router_weights_exp = weights_full.unsqueeze(-2)
            moe_lora = (lora_outputs * router_weights_exp).sum(dim=-1) * self.scaling[
                active_adapter
            ]

            self._last_router_logits = router_logits

        if x.dim() == 2:
            moe_lora = moe_lora.squeeze(1)

        return base_out + moe_lora.to(torch_result_dtype)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Custom state_dict that properly handles MoE structure"""
        if destination is None:
            destination = {}
        state_dict = super().state_dict(destination, prefix, keep_vars)
        return state_dict

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """Override to ensure proper saving of all components"""
        super()._save_to_state_dict(destination, prefix, keep_vars)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """Custom loading that handles MoE structure"""
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


def moe_dispatch_default(
    target: torch.nn.Module, adapter_name: str, lora_config, **kwargs
) -> Optional[torch.nn.Module]:
    """
    Dispatch function that supports MoELoraLinear.
    """
    from peft.tuners.lora.layer import Linear, Embedding, Conv2d, Conv1D
    from peft.tuners.tuners_utils import BaseTunerLayer

    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Embedding):
        embedding_kwargs = kwargs.copy()
        embedding_kwargs.pop("fan_in_fan_out", None)
        embedding_kwargs.update(lora_config.loftq_config)
        new_module = Embedding(target, adapter_name, **embedding_kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        kwargs.update(lora_config.loftq_config)
        new_module = Conv2d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Linear):
        # Check if we should use MoE LoRA
        if getattr(lora_config, "use_moe_lora", False):
            # Extract MoE-specific config parameters
            num_experts = getattr(lora_config, "num_experts", 4)
            init_zero_router = getattr(lora_config, "init_zero_router", False)
            stochastic_router = getattr(lora_config, "stochastic_router", None)
            expert_specific = getattr(lora_config, "expert_specific", None)
            fast_version = getattr(lora_config, "fast_version", False)
            sparse_moe_enabled = getattr(lora_config, "sparse_moe_enabled", False)
            top_k_sparse_moe = getattr(lora_config, "top_k_sparse_moe", 1)

            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            kwargs.update(lora_config.loftq_config)

            new_module = MoELoraLinear(
                target,
                adapter_name=adapter_name,
                num_experts=num_experts,
                init_zero_router=init_zero_router,
                stochastic_router=stochastic_router,
                expert_specific=expert_specific,
                fast_version=fast_version,
                sparse_moe_enabled=sparse_moe_enabled,
                top_k_sparse_moe=top_k_sparse_moe,
                **kwargs,
            )
        else:
            # Use regular Linear LoRA
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            kwargs.update(lora_config.loftq_config)
            new_module = Linear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        if not kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                "Setting fan_in_fan_out to True."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        kwargs.update(lora_config.loftq_config)
        new_module = Linear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)

    return new_module


def moe_get_peft_model_state_dict(
    model,
    state_dict=None,
    adapter_name="default",
    unwrap_compiled=False,
    save_embedding_layers="auto",
):
    """
    MoE-compatible version of get_peft_model_state_dict that includes router weights.
    """
    from peft.utils.other import EMBEDDING_LAYER_NAMES, check_file_exists_on_hf_hub
    from peft.utils.save_and_load import get_embedding_layer_name, has_valid_embedding_base_layer
    from peft.utils import PeftType

    if unwrap_compiled:
        model = getattr(model, "_orig_mod", model)

    config = model.peft_config[adapter_name]
    if state_dict is None:
        state_dict = model.state_dict()

    # TUNER SPECIFIC CODE
    if config.peft_type in (PeftType.LORA, PeftType.ADALORA):
        # MoE MODIFICATION: Include "router" in addition to "lora_"
        bias = config.bias
        if bias == "none":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k or "router" in k}
        elif bias == "all":
            to_return = {
                k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k or "router" in k
            }
        elif bias == "lora_only":
            to_return = {}
            for k in state_dict:
                if "lora_" in k or "router" in k:
                    to_return[k] = state_dict[k]
                    if "router" in k:
                        bias_name = k.split("router")[0] + "bias"
                    elif "lora_" in k:
                        bias_name = k.split("lora_")[0] + "bias"
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError

        # Include router in the filter
        to_return = {
            k: v
            for k, v in to_return.items()
            if (("lora_" in k and adapter_name in k) or ("bias" in k)) or ("router" in k)
        }

        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                rank_pattern = {
                    k.replace(f".{adapter_name}", ""): v for k, v in rank_pattern.items()
                }
                config.rank_pattern = rank_pattern
                to_return = model.resize_state_dict_by_rank_pattern(
                    rank_pattern, to_return, adapter_name
                )

        if config.use_dora:
            new_dora_suffix = f"lora_magnitude_vector.{adapter_name}.weight"

            def renamed_dora_weights(k):
                if k.endswith(new_dora_suffix):
                    k = k[:-7]
                return k

            to_return = {renamed_dora_weights(k): v for k, v in to_return.items()}

    elif config.peft_type == PeftType.BOFT:
        bias = config.bias
        if bias == "none":
            to_return = {k: state_dict[k] for k in state_dict if "boft_" in k}
        elif bias == "all":
            to_return = {k: state_dict[k] for k in state_dict if "boft_" in k or "bias" in k}
        elif bias == "boft_only":
            to_return = {}
            for k in state_dict:
                if "boft_" in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split("boft_")[0] + "bias"
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError

    elif config.peft_type == PeftType.LOHA:
        to_return = {k: state_dict[k] for k in state_dict if "hada_" in k}

    elif config.peft_type == PeftType.LOKR:
        to_return = {k: state_dict[k] for k in state_dict if "lokr_" in k}

    elif config.peft_type == PeftType.ADAPTION_PROMPT:
        to_return = {
            k: state_dict[k] for k in state_dict if k.split(".")[-1].startswith("adaption_")
        }

    elif config.is_prompt_learning:
        to_return = {}
        if config.peft_type == PeftType.MULTITASK_PROMPT_TUNING:
            to_return["prefix_task_cols"] = model.prompt_encoder[adapter_name].prefix_task_cols
            to_return["prefix_task_rows"] = model.prompt_encoder[adapter_name].prefix_task_rows
            prompt_embeddings = model.prompt_encoder[adapter_name].embedding.weight
        else:
            if config.inference_mode:
                prompt_embeddings = model.prompt_encoder[adapter_name].embedding.weight
            else:
                prompt_embeddings = model.get_prompt_embedding_to_save(adapter_name)
        to_return["prompt_embeddings"] = prompt_embeddings

    elif config.peft_type == PeftType.IA3:
        to_return = {k: state_dict[k] for k in state_dict if "ia3_" in k}

    elif config.peft_type == PeftType.OFT:
        to_return = {k: state_dict[k] for k in state_dict if "oft_" in k}

    elif config.peft_type == PeftType.POLY:
        to_return = {k: state_dict[k] for k in state_dict if "poly_" in k}

    elif config.peft_type == PeftType.LN_TUNING:
        to_return = {k: state_dict[k] for k in state_dict if "ln_tuning_" in k}

    elif config.peft_type == PeftType.VERA:
        to_return = {k: state_dict[k] for k in state_dict if "vera_lambda_" in k}
        if config.save_projection:
            if f"base_model.vera_A.{adapter_name}" not in state_dict:
                raise ValueError(
                    "Model was initialised to not save vera_A and vera_B but config now specifies to save projection!"
                    " Set `config.save_projection` to `False`."
                )
            to_return["base_model.vera_A." + adapter_name] = state_dict[
                "base_model.vera_A." + adapter_name
            ]
            to_return["base_model.vera_B." + adapter_name] = state_dict[
                "base_model.vera_B." + adapter_name
            ]
    elif config.peft_type == PeftType.FOURIERFT:
        to_return = {k: state_dict[k] for k in state_dict if "fourierft_" in k}
    elif config.peft_type == PeftType.XLORA:
        to_return = {k: state_dict[k] for k in state_dict if "internal_xlora_classifier" in k}
    elif config.peft_type == PeftType.HRA:
        to_return = {k: state_dict[k] for k in state_dict if "hra_" in k}
    elif config.peft_type == PeftType.VBLORA:
        to_return = {}
        if config.num_vectors < 2**8:
            indices_dtype = torch.uint8
        elif config.num_vectors < 2**15:
            indices_dtype = torch.int16
        elif config.num_vectors < 2**31:
            indices_dtype = torch.int32
        else:
            indices_dtype = torch.int64
        if config.save_only_topk_weights:
            for k in state_dict:
                if "vblora_logits" in k:
                    logits, indices = state_dict[k].topk(config.topk)
                    to_return.update({k + "_topk_indices": indices.to(dtype=indices_dtype)})
                    to_return.update(
                        {k + "_topk_weights": torch.softmax(logits, dim=-1)[:, :, :-1].contiguous()}
                    )
        else:
            to_return = {k: state_dict[k] for k in state_dict if "vblora_logits" in k}
        to_return["base_model.vblora_vector_bank." + adapter_name] = state_dict[
            "base_model.vblora_vector_bank." + adapter_name
        ]
    elif config.peft_type == PeftType.BONE:
        to_return = {k: state_dict[k] for k in state_dict if "bone_" in k}
    else:
        raise ValueError(f"Unknown PEFT type passed: {config.peft_type}")

    # MODULES TO SAVE
    if getattr(model, "modules_to_save", None) is not None:
        for key, value in state_dict.items():
            if any(
                f"{module_name}.modules_to_save.{adapter_name}" in key
                for module_name in model.modules_to_save
            ):
                to_return[key.replace("modules_to_save.", "")] = value

    # DEAL WITH EMBEDDINGS
    is_embedding_in_target_modules = False
    if (
        save_embedding_layers == "auto"
        and hasattr(config, "target_modules")
        and any(k in config.target_modules for k in EMBEDDING_LAYER_NAMES)
    ):
        warnings.warn(
            "Setting `save_embedding_layers` to `True` as embedding layers found in `target_modules`."
        )
        save_embedding_layers = is_embedding_in_target_modules = True
    elif save_embedding_layers == "auto":
        vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
        model_id = getattr(config, "base_model_name_or_path", None)

        has_base_config = False

        if model_id is not None:
            local_config_exists = os.path.exists(os.path.join(model_id, "config.json"))
            exists = local_config_exists or check_file_exists_on_hf_hub(model_id, "config.json")
            if exists is None:
                warnings.warn(
                    f"Could not find a config file in {model_id} - will assume that the vocabulary was not modified."
                )
                has_base_config = False
            else:
                has_base_config = exists

        if (
            vocab_size
            and model_id
            and has_base_config
            and (vocab_size != model.config.__class__.from_pretrained(model_id).vocab_size)
        ):
            warnings.warn(
                "Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning."
            )
            save_embedding_layers = True
        else:
            save_embedding_layers = False

    if save_embedding_layers and hasattr(model, "get_input_embeddings"):
        for layer in [model.get_input_embeddings(), model.get_output_embeddings()]:
            if not is_embedding_in_target_modules or has_valid_embedding_base_layer(layer):
                embedding_module_name = get_embedding_layer_name(
                    model, layer, is_embedding_in_target_modules
                )
                if embedding_module_name:
                    to_return.update(
                        {k: v for k, v in state_dict.items() if embedding_module_name in k}
                    )
    elif save_embedding_layers:
        warnings.warn(
            "Could not identify embedding layer(s) because the model is not a 🤗 transformers model."
        )

    # REMOVE ADAPTER NAME
    to_return = {k.replace(f".{adapter_name}", ""): v for k, v in to_return.items()}
    return to_return


def get_peft_moe(
    model_name_or_path: str,
    lora_config,
    num_experts: int = 4,
    init_zero_router: bool = False,
    stochastic_router=None,
    expert_specific=None,
    fast_version: bool = True,
    sparse_moe_enabled: bool = False,
    top_k_sparse_moe: int = 1,
    **model_kwargs,
):
    """
    Convenience function to create a MoE LoRA model with common settings.

    This is a baseline method for comparison (not core MCL).

    Args:
        model_name_or_path: Model identifier or path
        lora_config: Base LoraConfig with r, alpha, target_modules, etc.
        num_experts: Number of expert LoRA adapters (default: 4)
        init_zero_router: Initialize router weights to zero (default: False)
        stochastic_router: Use stochastic routing - set to "true" to enable
        expert_specific: Use specific expert per hypothesis - set to "true" to enable
        fast_version: Use optimized forward pass (default: True)
        sparse_moe_enabled: Enable sparse MoE with top-k routing (default: False)
        top_k_sparse_moe: Number of experts to activate per token (default: 1)
        **model_kwargs: Additional arguments for AutoModel.from_pretrained()

    Returns:
        Model with MoE LoRA adapters
    """
    from transformers import AutoModel
    from peft import get_peft_model

    # Apply MoE patches
    patch_peft_for_moe(enable=True)

    # Add MoE-specific attributes to config
    lora_config.use_moe_lora = True
    lora_config.num_experts = num_experts
    lora_config.init_zero_router = init_zero_router
    lora_config.stochastic_router = stochastic_router
    lora_config.expert_specific = expert_specific
    lora_config.fast_version = fast_version
    lora_config.sparse_moe_enabled = sparse_moe_enabled
    lora_config.top_k_sparse_moe = top_k_sparse_moe

    # Load base model
    base_model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)

    # Apply PEFT with MoE
    model = get_peft_model(base_model, lora_config)

    print(f"✓ Created MoE LoRA model with {num_experts} experts")
    if sparse_moe_enabled:
        print(f"  - Sparse MoE: top-{top_k_sparse_moe} routing")
    if fast_version:
        print(f"  - Fast version enabled")

    return model


def patch_peft_for_moe(enable: bool = True):
    """
    Patch PEFT to support MoE LoRA layers.

    This patches:
    - dispatch_default: Uses MoELoraLinear when use_moe_lora=True
    - get_peft_model_state_dict: Includes router weights when saving

    Args:
        enable: If True, apply MoE patches. If False, restore original behavior.

    Usage:
        ```python
        from peft_mcl.moe_patches import patch_peft_for_moe

        # Enable MoE support
        patch_peft_for_moe(enable=True)

        # Create model with MoE
        lora_config = LoraConfig(use_moe_lora=True, num_experts=4, ...)
        model = get_peft_model(base_model, lora_config)
        ```
    """
    if enable:
        # Patch dispatch_default
        try:
            from peft.tuners.lora import layer as lora_layer_module

            if not hasattr(lora_layer_module, "_moe_original_dispatch_default"):
                lora_layer_module._moe_original_dispatch_default = (
                    lora_layer_module.dispatch_default
                )
            lora_layer_module.dispatch_default = moe_dispatch_default
        except ImportError as e:
            print(f"⚠ Could not patch dispatch_default: {e}")

        # Patch get_peft_model_state_dict
        try:
            from peft.utils import save_and_load as peft_save_and_load_module

            if not hasattr(peft_save_and_load_module, "_moe_original_get_peft_model_state_dict"):
                peft_save_and_load_module._moe_original_get_peft_model_state_dict = (
                    peft_save_and_load_module.get_peft_model_state_dict
                )
            peft_save_and_load_module.get_peft_model_state_dict = moe_get_peft_model_state_dict
        except (ImportError, AttributeError) as e:
            print(f"⚠ Could not patch get_peft_model_state_dict: {e}")

        print("✓ MoE LoRA patches applied")
    else:
        # Restore dispatch_default
        try:
            from peft.tuners.lora import layer as lora_layer_module

            if hasattr(lora_layer_module, "_moe_original_dispatch_default"):
                lora_layer_module.dispatch_default = (
                    lora_layer_module._moe_original_dispatch_default
                )
                delattr(lora_layer_module, "_moe_original_dispatch_default")
        except ImportError:
            pass

        # Restore get_peft_model_state_dict
        try:
            from peft.utils import save_and_load as peft_save_and_load_module

            if hasattr(peft_save_and_load_module, "_moe_original_get_peft_model_state_dict"):
                peft_save_and_load_module.get_peft_model_state_dict = (
                    peft_save_and_load_module._moe_original_get_peft_model_state_dict
                )
                delattr(peft_save_and_load_module, "_moe_original_get_peft_model_state_dict")
        except (ImportError, AttributeError):
            pass

        print("✓ MoE LoRA patches removed")
