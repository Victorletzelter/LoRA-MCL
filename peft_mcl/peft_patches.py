"""
Patches for PEFT library to support MCL (Multiple Choice Learning) training without source code modification.

This module contains the core MCL patches. For MoE LoRA support (baseline method),
see moe_patches.py.
"""

import os
import json
import warnings
import torch
import torch.nn as nn
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.config import PeftConfig
from peft.utils import CONFIG_NAME


def mcl_infer_device() -> str:
    """
    MCL-compatible version of infer_device that respects LOCAL_RANK for distributed training.

    This is useful for multi-GPU training where each process should use a different GPU
    based on its LOCAL_RANK environment variable.

    Returns:
        Device string (e.g., "cuda:0", "cuda", "cpu")
    """
    if torch.cuda.is_available():
        if "LOCAL_RANK" in os.environ:
            return f"cuda:{int(os.environ['LOCAL_RANK'])}"
        else:
            return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    # Check for MLU (Machine Learning Unit - Cambricon)
    elif hasattr(torch, "mlu") and hasattr(torch.mlu, "is_available") and torch.mlu.is_available():
        return "mlu"
    # Check for XPU (Intel GPU)
    elif hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available") and torch.xpu.is_available():
        return "xpu"
    # Check for NPU (Ascend NPU)
    elif hasattr(torch, "npu") and hasattr(torch.npu, "is_available") and torch.npu.is_available():
        return "npu"
    return "cpu"


def mcl_set_adapter(self, adapter_names: str | list[str]) -> None:
    """
    MCL-compatible version of set_adapter that keeps all adapters trainable.

    In MCL training, we need gradients for all adapters simultaneously because
    the winner-takes-all loss computation requires comparing losses across all
    hypotheses. Therefore, we don't deactivate gradients on inactive adapters.

    Args:
        adapter_names: Name of the adapter(s) to be activated.
    """
    if isinstance(adapter_names, str):
        adapter_names = [adapter_names]

    # Activate grads on the active adapter(s)
    # BUT: Don't deactivate grads on inactive adapters (MCL needs them all!)
    for layer_name in self.adapter_layer_names:
        module_dict = getattr(self, layer_name)
        if not isinstance(module_dict, nn.ModuleDict):
            continue
        for key, layer in module_dict.items():
            if key in adapter_names:
                layer.requires_grad_(True)
            # MCL MODIFICATION: We intentionally don't set requires_grad=False
            # for inactive adapters because MCL training needs gradients for all

    self._active_adapter = adapter_names


def mcl_save_pretrained(self, save_directory: str, **kwargs) -> None:
    """
    MCL-compatible version of save_pretrained that handles OmegaConf serialization.

    This patched version converts OmegaConf ListConfig objects to regular lists
    before JSON serialization, preventing serialization errors when using Hydra configs.

    Args:
        save_directory: The directory where the configuration will be saved.
        kwargs: Additional keyword arguments passed along to push_to_hub.
    """
    if os.path.isfile(save_directory):
        raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

    os.makedirs(save_directory, exist_ok=True)
    auto_mapping_dict = kwargs.pop("auto_mapping_dict", None)

    output_dict = self.to_dict()

    # MCL MODIFICATION: Handle OmegaConf and other special types
    for key, value in output_dict.items():
        if isinstance(value, set):
            output_dict[key] = list(value)
        # Check for ListConfig from OmegaConf
        elif str(type(value)).endswith("ListConfig'>"):
            try:
                # Try to convert to a regular list
                output_dict[key] = list(value)
            except Exception:
                # If that fails, try a more specific approach
                try:
                    # For OmegaConf specifically
                    import omegaconf

                    if isinstance(value, omegaconf.ListConfig):
                        output_dict[key] = omegaconf.OmegaConf.to_container(value)
                except (ImportError, Exception):
                    # If all else fails, convert to string
                    output_dict[key] = str(value)

    output_path = os.path.join(save_directory, CONFIG_NAME)

    # Add auto mapping details for custom models.
    if auto_mapping_dict is not None:
        output_dict["auto_mapping"] = auto_mapping_dict

    # save it
    with open(output_path, "w") as writer:
        output_dict.pop("_custom_modules", None)
        writer.write(json.dumps(output_dict, indent=2, sort_keys=True))


def patch_peft_for_mcl(enable: bool = True):
    """
    Patch PEFT methods to be compatible with MCL training.

    This patches:
    - set_adapter: Keeps all adapters trainable for MCL winner-takes-all loss
    - save_pretrained: Handles OmegaConf serialization for Hydra configs
    - infer_device: Respects LOCAL_RANK for distributed training

    Note: For MoE LoRA support (baseline method), use patch_peft_for_moe() from moe_patches.py instead.

    Call this function once before creating your MCL model.

    Args:
        enable: If True, apply MCL patches. If False, restore original behavior.

    Usage:
        ```python
        from peft_mcl.peft_patches import patch_peft_for_mcl

        # Before creating your model
        patch_peft_for_mcl(enable=True)

        model = get_peft_mcl_model(...)
        ```
    """
    if enable:
        # Patch set_adapter
        if not hasattr(BaseTunerLayer, "_original_set_adapter"):
            BaseTunerLayer._original_set_adapter = BaseTunerLayer.set_adapter
        BaseTunerLayer.set_adapter = mcl_set_adapter

        # Patch save_pretrained
        if not hasattr(PeftConfig, "_original_save_pretrained"):
            PeftConfig._original_save_pretrained = PeftConfig.save_pretrained
        PeftConfig.save_pretrained = mcl_save_pretrained

        # Patch infer_device
        try:
            from peft.utils import other as peft_other_module

            if not hasattr(peft_other_module, "_original_infer_device"):
                peft_other_module._original_infer_device = peft_other_module.infer_device
            peft_other_module.infer_device = mcl_infer_device
        except (ImportError, AttributeError):
            pass  # If infer_device doesn't exist or module not found, skip this patch

        print("✓ PEFT patched for MCL training (all adapters trainable + OmegaConf + distributed)")
    else:
        # Restore original behavior
        if hasattr(BaseTunerLayer, "_original_set_adapter"):
            BaseTunerLayer.set_adapter = BaseTunerLayer._original_set_adapter
            delattr(BaseTunerLayer, "_original_set_adapter")

        if hasattr(PeftConfig, "_original_save_pretrained"):
            PeftConfig.save_pretrained = PeftConfig._original_save_pretrained
            delattr(PeftConfig, "_original_save_pretrained")

        # Restore infer_device
        try:
            from peft.utils import other as peft_other_module

            if hasattr(peft_other_module, "_original_infer_device"):
                peft_other_module.infer_device = peft_other_module._original_infer_device
                delattr(peft_other_module, "_original_infer_device")
        except (ImportError, AttributeError):
            pass

        print("✓ PEFT restored to original behavior")


def context_manager_mcl_training():
    """
    Context manager for MCL training that temporarily patches PEFT.

    Usage:
        ```python
        from peft_mcl.peft_patches import context_manager_mcl_training

        with context_manager_mcl_training():
            trainer.train()  # MCL training with all adapters trainable
        ```
    """
    from contextlib import contextmanager

    @contextmanager
    def _context():
        patch_peft_for_mcl(enable=True)
        try:
            yield
        finally:
            patch_peft_for_mcl(enable=False)

    return _context()
