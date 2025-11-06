"""
MCL Trainer - Custom Trainer that passes global_step and epoch to the model.

This trainer automatically injects global_step and epoch_number into the model's
forward pass, eliminating the need to modify the Transformers Trainer source code.
"""

from transformers import Trainer
from typing import Dict, Union, Any, Optional, Tuple
import torch
import torch.nn as nn


class MCLTrainer(Trainer):
    """
    Custom Trainer for MCL models that automatically passes global_step and epoch_number
    to the model's forward method during training.

    This eliminates the need to modify the Transformers Trainer source code.

    Usage:
        ```python
        from peft_mcl import MCLTrainer, get_peft_mcl

        model = get_peft_mcl(
            model_name_or_path="gpt2",
            num_hyps=3,
            wta_training_mode="annealed-wta",
            lora_config=lora_config,
        )

        trainer = MCLTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        trainer.train()
        ```
    """

    def compute_loss(
        self,
        model,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
        **kwargs,
    ):
        """
        Override compute_loss to inject global_step and epoch_number into inputs.

        Args:
            model: The model to compute loss for
            inputs: Dictionary of inputs to the model
            return_outputs: Whether to return outputs along with loss
            num_items_in_batch: Number of items in the batch

        Returns:
            Loss tensor, or tuple of (loss, outputs) if return_outputs=True
        """
        # Extract labels if using label smoother
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # Inject global_step and epoch from trainer state
        # These will be passed to the model's forward method
        if hasattr(self, "state") and self.state is not None:
            inputs["global_step"] = self.state.global_step
            inputs["epoch_number"] = self.state.epoch

        # Add num_items_in_batch if provided
        if num_items_in_batch is not None:
            inputs["num_items_in_batch"] = num_items_in_batch

        # Call the model's forward method
        outputs = model(**inputs)

        # Handle past state if needed (for models with past_key_values)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # Extract loss from outputs
        if isinstance(outputs, dict):
            loss = outputs.get("loss", None)
        else:
            # Handle tuple or CausalLMOutput
            loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss

        # Handle label smoothing if needed
        if labels is not None and self.label_smoother is not None:
            loss = self.label_smoother(outputs, labels)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list] = None,
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Override prediction_step to ensure evaluation uses WTA mode.

        During evaluation, we typically want to use standard WTA mode rather than
        annealed-wta or other training-specific modes.

        Args:
            model: The model to evaluate
            inputs: Dictionary of inputs
            prediction_loss_only: Whether to return only the loss
            ignore_keys: Keys to ignore in the output

        Returns:
            Tuple of (loss, logits, labels)
        """
        # For evaluation, enforce WTA mode (not annealed) unless explicitly specified
        if "wta_training_mode" not in inputs and hasattr(model, "wta_training_mode"):
            inputs["wta_training_mode"] = "wta"

        # Add global_step and epoch for consistency (though typically not used in eval)
        if hasattr(self, "state") and self.state is not None:
            inputs["global_step"] = self.state.global_step
            inputs["epoch_number"] = self.state.epoch

        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys, **kwargs)
