# %%
from typing import Dict, Any
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import os
import sys

sys.path.append(os.path.join(os.environ["PROJECT_ROOT"], "..", "Qwen2-Audio"))
import torch
import numpy as np
from torch.utils.data import Dataset
import math
from multiprocessing import Pool
from transformers import (
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    GPTNeoConfig,
    GPTNeoForCausalLM,
)
from transformers import Trainer, TrainingArguments
from torch.utils.data import default_collate
import itertools

# Create a callback to track metrics
from transformers.integrations import TensorBoardCallback
import matplotlib.pyplot as plt
import pickle
from peft import get_peft_model, LoraConfig, TaskType

base_path = os.environ["PROJECT_ROOT"]

# %%

class MarkovChainDataset(Dataset):
    def __init__(
        self,
        n_sequences=10000,
        seq_len=30,
        vocab_size=10,
        seed=42,
        transition_matrices=None,
        generate_sequences_parallel=False,
        generate_on_fly=False,
        n_latent_variables=10,
        use_stationary_initial_distribution=False,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.generate_on_fly = generate_on_fly
        self.n_sequences = n_sequences
        self.n_latent_variables = n_latent_variables
        self.use_stationary_initial_distribution = use_stationary_initial_distribution

        # Use provided transition matrices or generate random ones with disjoint non-zero entries
        if transition_matrices is not None:
            self.transition_matrices = transition_matrices
        else:
            # Generate random transition matrices with disjoint non-zero entries
            self.transition_matrices = self._generate_transition_matrices(
                vocab_size, n_latent_variables
            )

        if self.use_stationary_initial_distribution:
            self.stationary_distributions = [
                calculate_stationary_distribution(transition_matrix)
                for transition_matrix in self.transition_matrices
            ]

        # Generate latent variables for each sequence
        self.latent_variables = np.random.randint(
            0, n_latent_variables, size=n_sequences
        )

        # Generate sequences only if not using on-the-fly generation
        if not generate_on_fly:
            if generate_sequences_parallel:
                self.data = self._generate_sequences_parallel(n_sequences)
            else:
                self.data = [self._generate_sequence(i) for i in range(n_sequences)]

    def _generate_transition_matrices(self, vocab_size, n_latent_variables):
        """
        Generate independent row-stochastic transition matrices (not necessarily disjoint).

        Each matrix is sampled independently with positive entries and then
        row-normalized so every row sums to 1.
        """
        matrices = []
        for _ in range(n_latent_variables):
            mat = np.random.rand(vocab_size, vocab_size)
            row_sums = mat.sum(axis=1, keepdims=True)
            # Guard against any numerical edge cases
            row_sums[row_sums == 0.0] = 1.0
            mat = mat / row_sums
            matrices.append(mat)
        return matrices

    def _generate_sequence(self, idx):
        # Get the latent variable for this sequence
        # latent_var = self.latent_variables[idx]
        latent_var = np.random.randint(0, self.n_latent_variables)
        # Use the corresponding transition matrix
        transition_matrix = self.transition_matrices[latent_var]

        if self.use_stationary_initial_distribution:
            seq = [
                np.random.choice(
                    self.vocab_size, p=self.stationary_distributions[latent_var]
                )
            ]
        else:
            seq = [np.random.choice(self.vocab_size)]

        while len(seq) < self.seq_len:
            next_token = np.random.choice(self.vocab_size, p=transition_matrix[seq[-1]])
            seq.append(next_token)

        return seq

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        if self.generate_on_fly:
            # Generate sequence on the fly
            seq = self._generate_sequence(idx)
        else:
            seq = self.data[idx]
        return {
            "input_ids": torch.tensor(seq, dtype=torch.long),
            "labels": torch.tensor(seq, dtype=torch.long),
            "latent_variable": torch.tensor(
                self.latent_variables[idx], dtype=torch.long
            ),
        }

    def _generate_sequences_parallel(self, n_sequences):
        """Generate sequences using parallel processing"""
        # Generate different seeds for each sequence
        seeds = np.random.randint(0, 2**32, size=n_sequences)

        # Prepare arguments for each sequence
        args_list = [
            (
                seeds[i],
                self.seq_len,
                self.vocab_size,
                self.transition_matrices[self.latent_variables[i]],
            )
            for i in range(n_sequences)
        ]

        # Use multiprocessing to generate sequences
        with Pool() as pool:
            sequences = pool.map(_generate_single_sequence_for_parallel, args_list)

        return np.array(sequences)


class MyTrainer(Trainer):
    def __init__(
        self,
        model,
        wta_mode="wta",
        wta_params_epsilon=0.0,
        lr_scheduler_type="cosine",
        normalize_loss_by_T=False,
        seq_len=None,
        log_transition_matrices_during_training=False,
        results_folder=None,
        transition_matrices=None,
        my_args=None,
        *args,
        **kwargs,
    ):
        super().__init__(model=model, *args, **kwargs)  # Initialize with the model
        self.model = model  # Store the model
        self.win_counts = [0] * model.num_hyps  # Track wins per adapter
        self.model_losses = []  # Track losses per adapter per step
        self.validation_win_counts = [
            0
        ] * model.num_hyps  # Track validation wins per adapter
        self.total_validation_steps = 0  # Track total validation steps
        self.wta_mode = wta_mode
        self.wta_params_epsilon = wta_params_epsilon
        self.lr_scheduler_type = lr_scheduler_type
        self.normalize_loss_by_T = normalize_loss_by_T
        self.seq_len = seq_len
        self.log_transition_matrices_during_training = (
            log_transition_matrices_during_training
        )
        self.results_folder = results_folder
        self.transition_matrices = transition_matrices
        self.vocab_size = my_args["vocab_size"]

        # Create optimizer using the same configuration as the parent class
        self.optimizer = self.create_optimizer()

        # Create scheduler
        num_training_steps = self.state.max_steps
        self.scheduler = self.create_scheduler(self.optimizer, num_training_steps)

    def create_scheduler(self, optimizer, num_training_steps):
        """
        Create a scheduler for the given optimizer.
        """
        if self.lr_scheduler_type == "constant":
            # Create a dummy scheduler that accepts step but does nothing
            class DummyScheduler:
                def __init__(self):
                    pass

                def step(self, *args, **kwargs):
                    pass

            return DummyScheduler()
        elif self.lr_scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif self.lr_scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {self.lr_scheduler_type}")
        return scheduler

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Only keep the inputs model expects (no extra keys)
        inputs = {
            k: v
            for k, v in inputs.items()
            if k in ["input_ids", "labels", "attention_mask"]
        }
        batch_size = inputs["input_ids"].shape[0]
        losses = []
        for batch_idx in range(batch_size):
            inputs_batchs = {k: v[batch_idx].unsqueeze(0) for k, v in inputs.items()}
            outputs = model(**inputs_batchs)
            loss = outputs.loss
            losses.append(loss)
        losses = torch.stack(losses)  # shape (batch_size)
        return (losses, outputs) if return_outputs else losses

    def compute_loss_only_previous(
        self,
        model,
        inputs,
        return_outputs=False,
        return_transition_matrix=True,
        **kwargs,
    ):
        """Compute loss using only the previous token as context, like a Markov model.

        For each position t, we:
        1. Take only token t-1 as input
        2. Make the model predict token t
        3. Compute cross-entropy loss for that single prediction
        """
        inputs = {
            k: v
            for k, v in inputs.items()
            if k in ["input_ids", "labels", "attention_mask"]
        }
        batch_size = inputs["input_ids"].shape[0]
        seq_len = inputs["input_ids"].shape[1]
        losses = []

        # Get transition matrix with vocab_size forward passes
        if return_transition_matrix:
            transition_matrix = torch.zeros(
                (self.vocab_size, self.vocab_size), device=inputs["input_ids"].device
            )
            for prev_token in range(self.vocab_size):
                # Create input with just the previous token
                context = torch.tensor(
                    [prev_token], device=inputs["input_ids"].device
                ).unsqueeze(0)
                # Forward pass
                with torch.no_grad():
                    outputs = model(input_ids=context)
                    # Get predicted probabilities for next token
                    logits = outputs.logits[0, -1]  # take logits for last position
                    probs = torch.softmax(logits, dim=-1)
                    transition_matrix[prev_token] = probs
        else:
            transition_matrix = None

        for batch_idx in range(batch_size):
            sequence_losses = []

            # For each position t in the sequence (except t=0)
            for t in range(1, seq_len):
                # Create a mini-batch with just the previous token
                prev_token = inputs["input_ids"][batch_idx, t - 1 : t]
                next_token = inputs["input_ids"][batch_idx, t : t + 1]

                # Make a 2-token sequence: [prev_token, next_token]
                context = torch.cat([prev_token, next_token])

                # Create the corresponding labels (-100 for prev_token since we don't want to predict it)
                labels = torch.tensor(
                    [-100, next_token.item()], device=context.device, dtype=torch.long
                )

                # Forward pass with just this bigram
                outputs = model(
                    input_ids=context.unsqueeze(0), labels=labels.unsqueeze(0)
                )

                sequence_losses.append(outputs.loss)

            # Average over positions for this sequence
            avg_seq_loss = torch.stack(sequence_losses).mean()
            losses.append(avg_seq_loss)

        # Stack losses for all sequences in batch
        losses = torch.stack(losses)  # shape (batch_size)

        if return_transition_matrix is True:
            return (losses, transition_matrix)
        else:
            return (losses, outputs) if return_outputs else losses

    def compute_loss_wta(self, stacked_losses, mode="wta", epsilon=0.0):
        if mode == "wta":
            wta_loss = stacked_losses.min(dim=0).values
            wta_loss = wta_loss.mean()
        elif mode == "relaxed-wta":
            wta_loss = torch.min(stacked_losses, dim=0).values  # shape (batch_size)
            epsilon = self.wta_params_epsilon
            wta_loss = (
                1 - epsilon - epsilon / (len(self.model.peft_config) - 1)
            ) * wta_loss + (
                epsilon / (len(self.model.peft_config) - 1)
            ) * stacked_losses.sum(
                dim=0
            )  # shape (batch_size)
            wta_loss = wta_loss.mean()
        return wta_loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a training step with Winner Takes All dynamics using min-loss approach.
        All adapters receive gradients, but they're weighted by their performance.
        """
        # Move inputs to the right device
        inputs = {
            k: v.to(model.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # Get losses for each adapter
        model_losses = []
        for k in range(model.num_hyps):
            model.set_adapter(f"lora{k}")
            model.train()  # Ensure model is in training mode
            loss = self.compute_loss(model, inputs)
            model_losses.append(loss)

        # Stack losses to compare across adapters
        stacked_losses = torch.stack(model_losses)  # Shape: [n_adapters, batch_size]
        unit_test = False
        if unit_test is True:
            stacked_losses = stacked_losses[0].unsqueeze(0)

        wta_loss = self.compute_loss_wta(
            stacked_losses, mode=self.wta_mode, epsilon=self.wta_params_epsilon
        )

        if self.normalize_loss_by_T is True:
            wta_loss = wta_loss / (self.seq_len - 1)

        # Backward pass
        wta_loss.backward()

        # Store losses for this step
        self.model_losses.append([l.mean().item() for l in model_losses])

        return wta_loss.item()

    def train(self, *args, **kwargs):
        # Log initial losses (step 0) before training starts
        self.model.eval()

        # Compute training loss for each adapter
        train_dataloader = self.get_train_dataloader()
        train_loss = 0.0
        train_batches = 0
        train_losses = [0.0] * self.model.num_hyps

        with torch.no_grad():
            for batch in train_dataloader:
                batch = {
                    k: v.to(self.model.device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                stacked_losses = []
                for i in range(self.model.num_hyps):
                    self.model.set_adapter(f"lora{i}")
                    loss = self.compute_loss(self.model, batch)
                    stacked_losses.append(loss)
                    train_losses[i] += loss.mean().item()
                stacked_losses = torch.stack(stacked_losses)
                wta_loss = self.compute_loss_wta(
                    stacked_losses, mode=self.wta_mode, epsilon=self.wta_params_epsilon
                )
                train_loss += wta_loss.item()
                train_batches += 1
                train_losses = [loss / train_batches for loss in train_losses]

        train_loss = train_loss / train_batches

        eval_dataloader = self.get_eval_dataloader()

        eval_loss = 0.0
        eval_batches = 0
        eval_losses = [0.0] * self.model.num_hyps
        total_samples = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {
                    k: v.to(self.model.device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                stacked_losses = []
                batch_size = batch["input_ids"].size(0)
                for i in range(self.model.num_hyps):
                    self.model.set_adapter(f"lora{i}")
                    loss = self.compute_loss(self.model, batch)
                    eval_losses[i] += (
                        loss.mean().item() * batch_size
                    )  # accumulate sum of losses
                    stacked_losses.append(loss)
                stacked_losses = torch.stack(stacked_losses)
                wta_loss = self.compute_loss_wta(
                    stacked_losses, mode="wta", epsilon=0.0
                )
                eval_loss += (
                    wta_loss.item() * batch_size
                )  # accumulate sum of wta losses
                eval_batches += 1
                total_samples += batch_size

        eval_losses = [loss / total_samples for loss in eval_losses]
        eval_loss = eval_loss / total_samples

        # Log both losses at step 0
        self.state.log_history.append(
            {
                "loss": train_loss,  # Best model's loss
                "eval_loss": eval_loss,  # Best model's eval loss
                "step": 0,
                "epoch": 0,
                "model_losses": train_losses,
                "model_eval_losses": eval_losses,
                "win_counts": self.win_counts,
            }
        )

        if self.state.is_world_process_zero:
            for callback in self.callback_handler.callbacks:
                if hasattr(callback, "on_log"):
                    callback.on_log(
                        self.args,
                        self.state,
                        self.control,
                        logs={
                            "loss": train_loss,
                            "eval_loss": eval_loss,
                            "model_losses": train_losses,
                            "model_eval_losses": eval_losses,
                            "win_counts": self.win_counts,
                        },
                    )

        # Custom training loop
        self.state.epoch = 0
        self.state.global_step = 0

        for epoch in range(self.args.num_train_epochs):
            self.state.epoch = epoch
            for step, batch in enumerate(train_dataloader):
                self.state.global_step = epoch * len(train_dataloader) + step

                # Training step
                loss = self.training_step(self.model, batch)

                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Scheduler step
                self.scheduler.step()

                # Logging
                if step % self.args.logging_steps == 0:
                    self.log(
                        {"loss": loss, "epoch": epoch, "step": self.state.global_step}
                    )

                # Evaluation
                if (
                    self.args.evaluation_strategy == "steps"
                    and step % self.args.eval_steps == 0
                ):
                    self.evaluate()

            # Evaluate at the end of each epoch if strategy is "epoch"
            if self.args.evaluation_strategy == "epoch":
                self.evaluate()

        return self.state

    def evaluate(self, *args, **kwargs):
        # Reset validation counters
        self.validation_win_counts = [0] * self.model.num_hyps
        self.total_validation_steps = 0

        # Get evaluation dataloader
        eval_dataloader = self.get_eval_dataloader()

        # Compute losses for each adapter
        total_samples = 0

        with torch.no_grad():
            avg_model_losses = [0] * self.model.num_hyps
            wta_loss = 0.0
            for batch in eval_dataloader:
                batch = {
                    k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                model_losses = []

                for i in range(self.model.num_hyps):
                    self.model.set_adapter(f"lora{i}")
                    self.model.eval()
                    loss = self.compute_loss(self.model, batch)  # shape (batch_size)
                    model_losses.append(loss)
                stacked_losses = torch.stack(
                    model_losses
                )  # shape (n_adapters, batch_size)

                unit_test = False
                if unit_test is True:
                    stacked_losses = stacked_losses[0].unsqueeze(0)

                winning_idxes_per_batch = torch.argmin(
                    stacked_losses, dim=0
                )  # shape (batch_size)

                wta_loss += stacked_losses.min(dim=0).values.mean() * len(
                    batch["input_ids"]
                )

                # Update validation win counts
                for winning_idx in winning_idxes_per_batch:
                    self.validation_win_counts[winning_idx] += 1
                self.total_validation_steps += len(batch["input_ids"])

                # Accumulate losses
                for i, model_loss in enumerate(model_losses):
                    avg_model_losses[i] += model_loss.mean().item() * len(
                        batch["input_ids"]
                    )
                total_samples += len(batch["input_ids"])

        # Calculate average losses
        avg_losses = [e / total_samples for e in avg_model_losses]

        wta_loss = wta_loss / total_samples

        # Calculate win percentages
        win_percentages = [
            count / self.total_validation_steps * 100
            for count in self.validation_win_counts
        ]

        # Log results
        metrics = {
            "model_eval_steps": self.state.global_step,
            "eval_loss": wta_loss.item(),
            "model_eval_losses": avg_losses,
            "validation_win_percentages": win_percentages,
        }

        self.log(metrics)
        # Print detailed evaluation results
        print("\n=== Evaluation Results ===")
        print(f"Global step: {self.state.global_step}")
        print("\nAdapter Performance:")
        for i in range(self.model.num_hyps):
            print(f"Adapter {i+1}:")
            print(f"  - Loss: {avg_losses[i]:.4f}")
            print(f"  - Win percentage: {win_percentages[i]:.2f}%")
            print(f"  - Win count: {self.validation_win_counts[i]}")
        print("=======================\n")

        if self.log_transition_matrices_during_training is True:
            from train import plot_transition_matrices_comparison_v2_weighted

            plot_transition_matrices_comparison_v2_weighted(
                self.model,
                transition_matrices=self.transition_matrices,
                vocab_size=self.vocab_size,
                do_save_plot=True,
                results_folder=self.results_folder,
                save_path=f"step_{self.state.global_step}_transition_matrices_comparison_n_hyps_{self.model.num_hyps}.png",
            )

        return metrics


class MetricsCallback(TensorBoardCallback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.eval_losses = []
        self.steps = []
        self.eval_steps = []  # Track evaluation steps separately
        self.model_losses = []
        self.model_eval_losses = []
        self.model_eval_steps = []
        self.win_counts_history = []
        self.validation_win_percentages = []
        # Initialize TensorBoard writer
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir="./logs")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Handle scalar values
            if "loss" in logs:
                self.train_losses.append(logs["loss"])
                self.steps.append(state.global_step)
            if "eval_loss" in logs:
                self.eval_losses.append(logs["eval_loss"])
                self.eval_steps.append(
                    state.global_step
                )  # Store the step when evaluation occurred

            if "model_eval_steps" in logs:
                self.model_eval_steps.append(logs["model_eval_steps"])

            # Handle list values
            if "model_losses" in logs:
                self.model_losses.append(logs["model_losses"])
                # Log each model's loss separately
                for i, loss in enumerate(logs["model_losses"]):
                    self.writer.add_scalar(
                        f"train/model_{i}_loss", loss, state.global_step
                    )

            if "model_eval_losses" in logs:
                self.model_eval_losses.append(logs["model_eval_losses"])
                # Log each model's eval loss separately
                for i, loss in enumerate(logs["model_eval_losses"]):
                    self.writer.add_scalar(
                        f"eval/model_{i}_loss", loss, state.global_step
                    )

            if "win_counts" in logs:
                self.win_counts_history.append(logs["win_counts"])
                # Log each model's win count separately
                for i, count in enumerate(logs["win_counts"]):
                    self.writer.add_scalar(
                        f"train/model_{i}_wins", count, state.global_step
                    )

            if "validation_win_percentages" in logs:
                self.validation_win_percentages.append(
                    logs["validation_win_percentages"]
                )
                # Log each model's validation win percentage separately
                for i, percentage in enumerate(logs["validation_win_percentages"]):
                    self.writer.add_scalar(
                        f"eval/model_{i}_win_percentage", percentage, state.global_step
                    )

    def on_train_end(self, args, state, control, **kwargs):
        # Close the TensorBoard writer when training ends
        self.writer.close()


class MarkovModel(torch.nn.Module):
    def __init__(self, transition_matrix):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transition_matrix = torch.tensor(
            transition_matrix, dtype=torch.float32
        ).to(device)
        # self.register_buffer("transition_matrix", torch.tensor(transition_matrix, dtype=torch.float32))

    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape

        # Get the probability distribution for each input token
        logits = torch.log(
            self.transition_matrix[input_ids]
        )  # Shape: [batch_size, seq_len, vocab_size]

        loss = None
        if labels is not None:
            # Compute cross-entropy loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
            # Reshape for loss calculation

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            logits_view = shift_logits.view(
                -1, shift_logits.size(-1)
            )  # shape (batch_size * seq_len, vocab_size)
            labels_view = shift_labels.view(-1)  # shape (batch_size * seq_len)
            loss = loss_fct(logits_view, labels_view)

        return type("MarkovOutput", (), {"loss": loss, "logits": logits})()

    def compute_loss(self, inputs, return_outputs=False, **kwargs):
        # Only keep the inputs model expects (no extra keys)
        inputs = {
            k: v
            for k, v in inputs.items()
            if k in ["input_ids", "labels", "attention_mask"]
        }
        batch_size = inputs["input_ids"].shape[0]
        losses = []
        for batch_idx in range(batch_size):
            inputs_batchs = {k: v[batch_idx].unsqueeze(0) for k, v in inputs.items()}
            outputs = self(**inputs_batchs)
            loss = outputs.loss
            losses.append(loss)
        losses = torch.stack(losses)  # shape (batch_size)
        return (losses, outputs) if return_outputs else losses


class MarkovMHModel(torch.nn.Module):
    def __init__(self, transition_matrices):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transition_matrices = [
            torch.tensor(e, dtype=torch.float32).to(device) for e in transition_matrices
        ]

    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape

        # Get the probability distribution for each input token
        logits_list = []

        for k in range(len(self.transition_matrices)):
            logits_list.append(
                torch.log(self.transition_matrices[k][input_ids])
            )  # Shape: [batch_size, seq_len, vocab_size]

        logits_list = torch.stack(
            logits_list, dim=0
        )  # Shape [num_hyps, batch_size, seq_len, vocab_size]

        loss = None
        if labels is not None:
            # Compute cross-entropy loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
            # Reshape for loss calculation

            shift_logits = logits_list[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            labels_view = shift_labels.view(-1)  # shape (batch_size * seq_len)

            loss_list = []

            for k in range(len(self.transition_matrices)):
                logits_view = shift_logits[k, :, :, :].view(
                    -1, shift_logits.size(-1)
                )  # shape (batch_size * seq_len, vocab_size)
                loss_list.append(loss_fct(logits_view, labels_view))

            loss_list = torch.stack(loss_list, dim=0)
            loss = self.compute_loss_wta(loss_list, mode="wta")

        return type("MarkovOutput", (), {"loss": loss, "logits": logits_list[0]})()

    def compute_loss(self, inputs, return_outputs=False, **kwargs):
        # Only keep the inputs model expects (no extra keys)
        inputs = {
            k: v
            for k, v in inputs.items()
            if k in ["input_ids", "labels", "attention_mask"]
        }
        batch_size = inputs["input_ids"].shape[0]
        losses = []
        for batch_idx in range(batch_size):
            inputs_batchs = {k: v[batch_idx].unsqueeze(0) for k, v in inputs.items()}
            outputs = self(**inputs_batchs)
            loss = outputs.loss
            losses.append(loss)
        losses = torch.stack(losses)  # shape (batch_size)
        return (losses, outputs) if return_outputs else losses

    def compute_loss_wta(self, stacked_losses, mode="wta", epsilon=0.0):
        if mode == "wta":
            wta_loss = stacked_losses.min(dim=0).values
            wta_loss = wta_loss.mean()
        elif mode == "relaxed-wta":
            wta_loss = torch.min(stacked_losses, dim=0).values  # shape (batch_size)
            epsilon = self.wta_params_epsilon
            wta_loss = (
                1 - epsilon - epsilon / (len(self.model.peft_config) - 1)
            ) * wta_loss + (
                epsilon / (len(self.model.peft_config) - 1)
            ) * stacked_losses.sum(
                dim=0
            )  # shape (batch_size)
            wta_loss = wta_loss.mean()
        return wta_loss


class StationnaryDistributionModel(torch.nn.Module):
    def __init__(self, transition_matrix):
        super().__init__()
        self.register_buffer(
            "transition_matrix", torch.tensor(transition_matrix, dtype=torch.float32)
        )

        stationary_distribution = self.calculate_stationary_distribution(
            transition_matrix
        )
        self.register_buffer(
            "stationary_distribution",
            torch.tensor(stationary_distribution, dtype=torch.float32),
        )

    def calculate_stationary_distribution(self, transition_matrix):
        """
        Calculate the stationary distribution of a Markov chain.

        Args:
            transition_matrix (np.ndarray): The transition matrix P where P[i,j] is the probability of transitioning from state i to state j

        Returns:
            np.ndarray: The stationary distribution
        """
        # Convert to numpy array if it's a torch tensor
        if torch.is_tensor(transition_matrix):
            transition_matrix = transition_matrix.cpu().numpy()

        # Find eigenvalues and eigenvectors of the transpose of the transition matrix
        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)

        # Find the index of the eigenvalue closest to 1
        stationary_idx = np.argmin(np.abs(eigenvalues - 1.0))

        # The corresponding eigenvector is the stationary distribution
        stationary_dist = np.real(eigenvectors[:, stationary_idx])

        # Normalize to ensure it sums to 1
        stationary_dist = stationary_dist / stationary_dist.sum()

        return stationary_dist

    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape

        # For each position, output the stationary distribution as logits
        # Expand stationary distribution to match batch size and sequence length
        logits = (
            self.stationary_distribution.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, seq_len, -1)
        )
        logits = torch.log(logits)
        loss = None
        if labels is not None:
            # Compute cross-entropy loss
            loss_fct = torch.nn.CrossEntropyLoss()
            # Reshape for loss calculation
            logits_view = logits.view(
                -1, logits.size(-1)
            )  # [batch_size * seq_len, vocab_size]
            labels_view = labels.view(-1)  # [batch_size * seq_len]
            loss = loss_fct(logits_view, labels_view)

        return type("StationnaryOutput", (), {"loss": loss, "logits": logits})()


def build_hypotheses_list(n_hyps_max: int, always_include_one: bool = False):
    """Return hypotheses list while preserving original behavior.
    - If always_include_one is True: always return [1, n_hyps_max]
    - Else: return [n_hyps_max] when n_hyps_max == 1, otherwise [1, n_hyps_max]
    """
    if always_include_one:
        return [1, n_hyps_max]
    return [n_hyps_max] if n_hyps_max == 1 else [1, n_hyps_max]


def create_results_folder(base_path: str, args: Dict[str, Any]) -> str:
    """Create a timestamped results folder."""
    import datetime
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = (
        f"{current_datetime}_seqlen_{args['seq_len']}_Nit_{args['N_it']}_"
        f"N_hyps_{args['N_hyps_min']}-{args['N_hyps_max']}_seed_{args['seed']}_"
        f"name_{args['name']}_window_{args['window_size']}_"
        f"div_rank_{args['divide_rank_mcl']}_mult_rank_1_hyp_{args['multiply_rank_1_hyp']}"
    )

    results_folder = os.path.join(base_path, "results", "pkl_files", folder_name)
    os.makedirs(results_folder, exist_ok=True)
    return results_folder


def plot_transition_matrices_comparison(
    model_classes,
    predicted_matrices=None,
    dataset=None,
    transition_matrices=None,
    vocab_size=None,
    do_save_plot=True,
    results_folder=None,
    save_data=False,
):
    """
    Plot the predicted and target transition matrices in a 2x3 grid.
    Top row: Predictions (Pred hypothesis 1, Pred hypothesis 2, Pred MLE)
    Bottom row: Targets (P_1, P_2, \bar{P})

    Args:
        model_classes: Dictionary containing models with different numbers of hypotheses
        transition_matrices: List of target transition matrices
        vocab_size: Size of vocabulary
        do_save_plot: Whether to save the plot
    """
    if transition_matrices is not None:
        target_matrices = transition_matrices
    else:
        target_matrices = dataset.transition_matrices
    device = next(iter(model_classes.values())).device

    if predicted_matrices is None:

        # Get predicted matrices from both models (1 and 2 hypotheses)
        predicted_matrices = []

        # Get predictions from 2-hypothesis model
        model_2hyps = model_classes[2]
        for i in range(2):
            model_2hyps.set_adapter(f"lora{i}")
            model_2hyps.eval()
            pred_matrix = np.zeros((vocab_size, vocab_size))

            # For each input token, get model's predictions
            for input_token in range(vocab_size):
                input_ids = torch.tensor([[input_token]], device=device)
                with torch.no_grad():
                    outputs = model_2hyps(input_ids=input_ids)
                    logits = outputs.logits[0, -1]
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    pred_matrix[input_token] = probs

            predicted_matrices.append(pred_matrix)

        # Revert the order of the predicted matrices
        predicted_matrices = predicted_matrices[::-1]

        # Get predictions from 1-hypothesis model (MLE)
        model_1hyp = model_classes[1]
        model_1hyp.set_adapter("lora0")
        model_1hyp.eval()
        mle_matrix = np.zeros((vocab_size, vocab_size))

        for input_token in range(vocab_size):
            input_ids = torch.tensor([[input_token]], device=device)
            with torch.no_grad():
                outputs = model_1hyp(input_ids=input_ids)
                logits = outputs.logits[0, -1]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                mle_matrix[input_token] = probs

        predicted_matrices.append(mle_matrix)

    # Create figure with 2x5 grid (with blank middle column)
    fig = plt.figure(figsize=(14.5, 10))
    gs = plt.GridSpec(2, 5, width_ratios=[1, 0.0, 1, 0.0, 1])

    # Adjust subplot parameters to remove spacing
    plt.subplots_adjust(wspace=0, hspace=0.1)

    # Find the global min and max for colorbar
    vmin = min(
        min(m.min() for m in target_matrices), min(m.min() for m in predicted_matrices)
    )
    vmax = max(
        max(m.max() for m in target_matrices), max(m.max() for m in predicted_matrices)
    )

    # Plot predicted matrices (top row)
    titles_pred = [
        r"$\hat{P}_{\mathrm{MCL}}(\theta_1)$",
        r"$\hat{P}_{\mathrm{MCL}}(\theta_2)$",
        r"$\hat{P}_{\mathrm{MLE}}(\theta)$",
    ]
    for i, col in enumerate([0, 2, 4]):  # Use columns 0, 2, and 4 (skipping 1 and 3)
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(predicted_matrices[i], cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(titles_pred[i], fontsize=45, pad=18)
        ax.set_xticks([])
        ax.set_yticks([])
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Replace target_matrices[-1] by the weighted sum
    target_matrices += [np.zeros((vocab_size, vocab_size))]
    vocab_size = target_matrices[0].shape[0]
    stationary_distributions = [
        calculate_stationary_distribution(target_matrices[k])
        for k in range(len(target_matrices) - 1)
    ]
    weighted_transition_matrices = np.zeros((vocab_size, vocab_size))
    for i in range(vocab_size):
        for k in range(len(target_matrices) - 1):
            weighted_transition_matrices[i, :] += (
                target_matrices[k][i, :] * stationary_distributions[k][i]
            )
        denominator = np.sum(
            [stationary_distributions[k][i] for k in range(len(target_matrices) - 1)]
        )
        weighted_transition_matrices[i, :] = (
            weighted_transition_matrices[i, :] / denominator
        )
    print(target_matrices)
    target_matrices[-1] = weighted_transition_matrices

    # Plot target matrices (bottom row)
    titles_target = ["$P_1$", "$P_2$", "$\\bar{{P}}$"]
    for i, col in enumerate([0, 2, 4]):  # Use columns 0, 2, and 4 (skipping 1 and 3)
        ax = fig.add_subplot(gs[1, col])
        if i < len(target_matrices):
            print(i)
            print(target_matrices[i])
            im = ax.imshow(target_matrices[i], cmap="viridis", vmin=vmin, vmax=vmax)
        else:
            avg_matrix = target_matrices[-1]
            im = ax.imshow(avg_matrix, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(titles_target[i], fontsize=45, pad=18)
        ax.set_xticks([])
        ax.set_yticks([])
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Add colorbar
    cbar_ax = fig.add_axes([1.0, 0.015, 0.03, 0.9])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=25)
    # Remove colorbar outline
    cbar.outline.set_visible(False)

    if do_save_plot is True:
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                results_folder, f"transition_matrices_comparison_grid_weighted.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(
                results_folder, f"transition_matrices_comparison_grid_weighted.pdf"
            ),
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()

def calculate_stationary_distribution(transition_matrix):
    """
    Calculate the stationary distribution of a Markov chain.
    Args:
        transition_matrix (np.ndarray): The transition matrix P where P[i,j] is the probability of transitioning from state i to state j

    Returns:
        np.ndarray: The stationary distribution
    """
    # Convert to numpy array if it's a torch tensor
    if torch.is_tensor(transition_matrix):
        transition_matrix = transition_matrix.cpu().numpy()

    # Find eigenvalues and eigenvectors of the transpose of the transition matrix
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)

    # Find the index of the eigenvalue closest to 1
    stationary_idx = np.argmin(np.abs(eigenvalues - 1.0))

    # The corresponding eigenvector is the stationary distribution
    stationary_dist = np.real(eigenvectors[:, stationary_idx])

    # Normalize to ensure it sums to 1
    stationary_dist = stationary_dist / stationary_dist.sum()

    return stationary_dist


def _generate_single_sequence_for_parallel(args):
    """
    Standalone function for parallel sequence generation.

    Args:
        args (tuple): (seed, seq_len, vocab_size, transition_matrix)
    """
    seed, seq_len, vocab_size, transition_matrix = args
    # np.random.seed(seed)
    seq = [np.random.choice(vocab_size)]
    for _ in range(seq_len - 1):
        next_token = np.random.choice(vocab_size, p=transition_matrix[seq[-1]])
        seq.append(next_token)
    return seq


def build_gpt_model(
    vocab_size,
    hidden_size,
    n_layer,
    n_head,
    use_lora=True,
    lora_r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    num_hyps=1,
    window_size=2,
):
    # More powerful model with sliding window attention
    config = GPTNeoConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=n_layer,
        num_heads=n_head,
        window_size=window_size,  # Size of the sliding window
        attention_types=[
            [["local"], n_layer]
        ],  # Use only local attention for all layers
        max_position_embeddings=1024,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        activation_function="gelu_new",
        layer_norm_epsilon=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
    )

    model = GPTNeoForCausalLM(config)
    model.num_hyps = num_hyps

    if use_lora:
        model = prepare_model_with_lora(
            model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout, num_hyps=num_hyps
        )
        # Add these lines to verify parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(
            f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%"
        )

    return model


def calculate_entropy_rate(transition_matrix):
    # Find the stationary distribution by solving the eigenvalue problem
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    stationary_idx = np.argmin(np.abs(eigenvalues - 1.0))
    stationary_dist = np.real(eigenvectors[:, stationary_idx])
    stationary_dist = stationary_dist / stationary_dist.sum()  # Normalize

    # Calculate entropy for each state's transition probabilities
    state_entropies = []
    for i in range(len(transition_matrix)):
        row = transition_matrix[i]
        # Entropy of this state: -∑_j P_ij * log(P_ij)
        # Use natural log (base-e) to match PyTorch's cross-entropy implementation
        state_entropy = -np.sum(
            row * np.log(row + 1e-10)
        )  # Adding small epsilon to avoid log(0)
        state_entropies.append(state_entropy)
        # print(f"State {i} entropy: {state_entropy:.4f}")

    # Entropy rate is the weighted average of state entropies according to stationary distribution
    entropy_rate = np.sum(stationary_dist * np.array(state_entropies))

    # print(f"Stationary distribution: {stationary_dist}")
    # print(f"State-by-state entropies: {state_entropies}")
    # print(f"Note: Using natural log (base-e) to match PyTorch's cross-entropy implementation")

    return entropy_rate


def calculate_entropy(stationary_distribution):
    # Calculate entropy of the stationary distribution
    entropy = -np.sum(stationary_distribution * np.log(stationary_distribution + 1e-10))
    return entropy


def calculate_entropy_rates(transition_matrices):
    """
    Calculate the entropy rate for each transition matrix in a list of matrices.

    Args:
        transition_matrices (list): List of transition matrices, where each matrix is a numpy array

    Returns:
        tuple: (entropy_rates, stationary_distributions)
            - entropy_rates: list of entropy rates for each matrix
            - stationary_distributions: list of stationary distributions for each matrix
    """
    entropy_rates = []
    stationary_distributions = []

    for P in transition_matrices:
        # Calculate stationary distribution
        eigvals, eigvecs = np.linalg.eig(P.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        stationary = np.real(eigvecs[:, idx])
        stationary = stationary / stationary.sum()
        stationary_distributions.append(stationary)

        # Calculate entropy for each state's transition probabilities
        state_entropies = []
        for i in range(len(P)):
            row = P[i]
            # Entropy of this state: -∑_j P_ij * log(P_ij)
            # Use natural log (base-e) to match PyTorch's cross-entropy implementation
            state_entropy = -np.sum(
                row * np.log(row + 1e-10)
            )  # Adding small epsilon to avoid log(0)
            state_entropies.append(state_entropy)

        # Entropy rate is the weighted average of state entropies according to stationary distribution
        entropy_rate = np.sum(stationary * np.array(state_entropies))
        entropy_rates.append(entropy_rate)

    return entropy_rates, stationary_distributions


def calculate_unigram_cross_entropy(transition_matrix):
    # Find the stationary distribution
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    stationary_idx = np.argmin(np.abs(eigenvalues - 1.0))
    stationary_dist = np.real(eigenvectors[:, stationary_idx])
    stationary_dist = stationary_dist / stationary_dist.sum()  # Normalize

    # Compute the entropy of the unigram distribution
    unigram_entropy = -np.sum(stationary_dist * np.log(stationary_dist + 1e-10))

    return unigram_entropy


def generate_sequence(
    model, start_token, max_length=30, temperature=1.0, top_k=None, num_beams=None
):
    model.eval()
    input_ids = torch.tensor([[start_token]], device=model.device)

    if num_beams:
        # Beam Search
        output = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )
    else:
        # Temperature Sampling
        output = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
        )

    return output[0].tolist()


def sample_markov_chain(P, pi, T):
    """
    Sample a sequence of length T from a Markov chain with transition matrix P and initial distribution pi.
    """
    N = P.shape[0]
    x = np.zeros(T, dtype=int)
    # Sample initial state
    x[0] = np.random.choice(N, p=pi)
    for t in range(1, T):
        x[t] = np.random.choice(N, p=P[x[t - 1]])
    return x


def compute_sequence_prob(P, pi, x):
    """
    Compute the probability of a sequence x under a Markov chain defined by P and pi.
    """
    prob = pi[x[0]]
    for t in range(1, len(x)):
        prob *= P[x[t - 1], x[t]]
    return prob


def estimate_entropy_mixture(P_list, pi_list, T, M=10000):
    """
    Estimate the entropy H(X_{1:T}) of the mixture of Markov chains by Monte Carlo sampling.
    P_list: list of transition matrices (each shape N x N)
    pi_list: list of initial distributions (each length N)
    T: sequence length
    M: number of Monte Carlo samples per component
    """
    K = len(P_list)
    logs = []
    for k in range(K):
        Pk = P_list[k]
        pik = pi_list[k]
        for _ in range(M):
            x = sample_markov_chain(Pk, pik, T)
            # Compute mixture probability p(x) = (1/K) sum_j p_j(x)
            p_mix = 0.0
            for Pj, pij in zip(P_list, pi_list):
                p_mix += compute_sequence_prob(Pj, pij, x)
            p_mix /= K
            logs.append(-np.log(p_mix))
    return np.mean(logs)


def compute_sequence_logprob(P, pi, x):
    """
    Compute the log-probability of a sequence x under a Markov chain defined by P and pi,
    returning log p(x). Uses natural log. If probability is zero, returns -inf.
    """
    # Initial state log-prob
    p0 = pi[x[0]]
    if p0 <= 0:
        return -math.inf
    logp = math.log(p0)
    # Transitions
    for t in range(1, len(x)):
        p_trans = P[x[t - 1], x[t]]
        if p_trans <= 0:
            return -math.inf
        logp += math.log(p_trans)
    return logp


def estimate_entropy_mixture_stable(P_list, pi_list, T, M=10000, base=np.e):
    """
    Estimate the entropy H(X_{1:T}) of the mixture of Markov chains by Monte Carlo sampling,
    in a numerically stable way using log-domain computations.

    P_list: list of transition matrices (each shape N x N)
    pi_list: list of initial distributions (each length N)
    T: sequence length
    M: number of Monte Carlo samples per component
    base: log base for output entropy (default np.e for nats; use 2 for bits)

    Returns: estimated entropy in the chosen log base.
    """
    K = len(P_list)
    # Precompute log of K once
    log_K = math.log(K)
    # Factor to convert natural-log result to desired base: if base=2, divide by ln 2
    log_base_factor = math.log(base)

    total = 0.0
    count = 0
    for k in range(K):
        Pk = P_list[k]
        pik = pi_list[k]
        for _ in range(M):
            x = sample_markov_chain(Pk, pik, T)
            # Compute log p_k(x) for each component
            log_p_components = []
            for Pj, pij in zip(P_list, pi_list):
                lp = compute_sequence_logprob(Pj, pij, x)
                log_p_components.append(lp)
            # Compute log p_mix(x) via log-sum-exp: log(1/K * sum_j exp(log p_j(x)))
            logsum = logsumexp(log_p_components)
            log_p_mix = logsum - log_K
            # accumulate -log p_mix; since expectation over mixture: each component sampled with weight 1/K
            # But since we sample equally per component and average, we approximate E_{k~Unif, x~p_k}[ -log p_mix ]
            if log_p_mix == -math.inf:
                # sequence has zero probability under all components? unlikely unless P_list incorrect
                # treat contribution as zero? Actually if mixture prob zero, entropy diverges; but skip here
                continue
            total += -log_p_mix / log_base_factor
            count += 1
    # The expected value over K*M samples approximates the entropy
    # Because each component is sampled M times, total samples = K*M; average is total/count
    return total / count if count > 0 else float("nan")


def compute_entropy_exhaustive(P_list, pi_list, T, log_base=np.e):
    """
    Compute the expected negative log-likelihood (NLL) for next-token prediction
    under the mixture of Markov chains, matching the model's loss.

    P_list: list of transition matrices (each shape N x N)
    pi_list: list of initial distributions (each length N)
    T: sequence length
    log_base: base of logarithm (default np.e for nats)

    Returns: expected NLL (in chosen log base) for predicting x_1,...,x_{T-1} given x_0,...,x_{T-2}
    """
    K = len(P_list)
    N = P_list[0].shape[0]
    assert all(P.shape == (N, N) for P in P_list), "All P must be N x N"
    assert all(len(pi) == N for pi in pi_list), "All pi must length N"
    nll = 0.0
    for x in itertools.product(range(N), repeat=T):
        # Compute p_k(x) for each component
        p_x_components = []
        for P, pi in zip(P_list, pi_list):
            prob = pi[x[0]]
            for t in range(1, T):
                prob *= P[x[t - 1], x[t]]
            p_x_components.append(prob)
        # Mixture probability
        p_mix = sum(p_x_components) / K
        if p_mix > 0:
            # Now compute the sum of -log p(x_t | x_{<t}) under the mixture
            log_p_mix = np.log(p_mix)
            # For each t, compute p(x_{0:t}) and p(x_{0:t-1}) under the mixture
            for t in range(1, T):
                # p(x_{0:t}) = sum_k pi_k(x_0) * prod_{s=1}^t P_k(x_{s-1}, x_s)
                p_prefix = []
                for P, pi in zip(P_list, pi_list):
                    prob = pi[x[0]]
                    for s in range(1, t + 1):
                        prob *= P[x[s - 1], x[s]]
                    p_prefix.append(prob)
                p_prefix_mix = sum(p_prefix) / K
                # p(x_{0:t-1}) = sum_k pi_k(x_0) * prod_{s=1}^{t-1} P_k(x_{s-1}, x_s)
                p_prev = []
                for P, pi in zip(P_list, pi_list):
                    prob = pi[x[0]]
                    for s in range(1, t):
                        prob *= P[x[s - 1], x[s]]
                    p_prev.append(prob)
                p_prev_mix = sum(p_prev) / K
                # p(x_t | x_{<t}) = p(x_{0:t}) / p(x_{0:t-1})
                if p_prev_mix > 0 and p_prefix_mix > 0:
                    cond_prob = p_prefix_mix / p_prev_mix
                    nll -= p_mix * np.log(cond_prob) / np.log(log_base)
    return nll


def compute_entropy_exhaustive_wta(P_list, pi_list, T, log_base=np.e):
    """
    Compute the expected negative log-likelihood (NLL) for next-token prediction
    under the mixture of Markov chains, matching the model's loss.

    P_list: list of transition matrices (each shape N x N)
    pi_list: list of initial distributions (each length N)
    T: sequence length
    log_base: base of logarithm (default np.e for nats)

    Returns: expected NLL (in chosen log base) for predicting x_1,...,x_{T-1} given x_0,...,x_{T-2}
    """
    K = len(P_list)
    N = P_list[0].shape[0]
    assert all(P.shape == (N, N) for P in P_list), "All P must be N x N"
    assert all(len(pi) == N for pi in pi_list), "All pi must length N"
    nll = 0.0
    for x in itertools.product(range(N), repeat=T):
        # Compute p_k(x) for each component
        p_x_components = []
        for P, pi in zip(P_list, pi_list):
            prob = pi[x[0]]
            for t in range(1, T):
                prob *= P[x[t - 1], x[t]]
            p_x_components.append(prob)
        # Mixture probability
        k_star = np.argmax(p_x_components)
        p_k_star = p_x_components[k_star]
        p_mix = sum(p_x_components) / K
        if p_mix > 0:
            # For each t, compute p(x_{0:t}) and p(x_{0:t-1}) under the mixture
            for t in range(1, T):
                # p(x_{0:t}) = sum_k pi_k(x_0) * prod_{s=1}^t P_k(x_{s-1}, x_s)
                p_prefix = []
                P, pi = P_list[k_star], pi_list[k_star]
                prob = pi[x[0]]
                for s in range(1, t + 1):
                    prob *= P[x[s - 1], x[s]]
                p_prefix.append(prob)
                p_prefix_mix = sum(p_prefix)
                # p(x_{0:t-1}) = sum_k pi_k(x_0) * prod_{s=1}^{t-1} P_k(x_{s-1}, x_s)
                p_prev = []
                # for P, pi in zip(P_list, pi_list):
                P, pi = P_list[k_star], pi_list[k_star]
                prob = pi[x[0]]
                for s in range(1, t):
                    prob *= P[x[s - 1], x[s]]
                p_prev.append(prob)
                p_prev_mix = sum(p_prev)
                # p(x_t | x_{<t}) = p(x_{0:t}) / p(x_{0:t-1})
                if p_prev_mix > 0 and p_prefix_mix > 0:
                    cond_prob = p_prefix_mix / p_prev_mix
                    nll -= p_mix * np.log(cond_prob) / np.log(log_base)
    return nll


def logsumexp(log_vals):
    """
    Compute log(sum(exp(log_vals))) in a numerically stable way.
    log_vals: list or array of log-values
    """
    m = max(log_vals)
    if m == -math.inf:
        return -math.inf
    sum_exp = sum(math.exp(lv - m) for lv in log_vals)
    return m + math.log(sum_exp)


def compute_entropy_exhaustive_stable(P_list, pi_list, T, log_base=np.e):
    """
    Compute the entropy H(X_{1:T}) of the mixture of Markov chains by exhaustive enumeration
    in a numerically stable way using log-sum-exp.

    P_list: list of transition matrices (each shape N x N)
    pi_list: list of initial distributions (each length N)
    T: sequence length
    log_base: base of logarithm (default np.e for nats; use 2 for bits)

    Returns: entropy value (in chosen log base)
    """
    K = len(P_list)
    N = P_list[0].shape[0]
    assert all(P.shape == (N, N) for P in P_list), "All P must be N x N"
    assert all(len(pi) == N for pi in pi_list), "All pi must length N"

    entropy = 0.0
    log_K = math.log(K)
    log_base_factor = math.log(log_base)

    for x in itertools.product(range(N), repeat=T):
        # Compute log p_k(x) for each component
        log_p_components = []
        for P, pi in zip(P_list, pi_list):
            # If initial pi[x[0]] is zero, log_pi is -inf
            if pi[x[0]] == 0:
                log_p = -math.inf
            else:
                log_p = math.log(pi[x[0]])
            for t in range(1, T):
                p_trans = P[x[t - 1], x[t]]
                if p_trans == 0:
                    log_p = -math.inf
                    break
                log_p += math.log(p_trans)
            log_p_components.append(log_p)

        # Mixture log-prob: log p_mix(x) = log(1/K * sum_k exp(log p_k(x)))
        # = -log K + logsumexp(log_p_components)
        logsum = logsumexp(log_p_components)
        log_p_mix = logsum - log_K
        # If mixture prob is zero (all components zero), skip
        if log_p_mix == -math.inf:
            continue

        # p_mix = exp(log_p_mix), safe unless extremely small; for enumeration size small it's okay
        p_mix = math.exp(log_p_mix)
        # Accumulate entropy: -p_mix * (log p_mix / log_base)
        entropy -= p_mix * (log_p_mix / log_base_factor)
    return entropy


def compute_entropy_monte_carlo(
    P_list, pi_list, T, M=50000, log_base=np.e, random_state=None
):
    """
    Monte-Carlo approximation of the expected negative log-likelihood (NLL) returned by
    ``compute_entropy_exhaustive``.

    Instead of exhaustively enumerating all \(N^{T}\) sequences, this variant draws
    *M* sequences from the *mixture* of Markov chains and averages the per-token
    surprise.

    Args
    -----
    P_list : list[np.ndarray]
        Transition matrices \(P_c\) of shape ``[N, N]`` for each latent class.
    pi_list : list[np.ndarray]
        Initial distributions \(\pi_c\) for each latent class.  Must be the same
        length as *P_list*.
    T : int
        Sequence length (number of symbols) to sample.
    M : int, default 10000
        Number of Monte-Carlo samples to draw.
    log_base : float, default ``np.e``
        Base of the logarithm used to measure the NLL (e.g. set to ``2`` for bits).
    random_state : int | numpy.random.Generator | None
        Optional seed or ``Generator`` for reproducible sampling.

    Returns
    -------
    float
        Estimated expected NLL (same units as ``log_base``).
    """
    # Validate inputs
    K = len(P_list)
    assert K == len(pi_list), "P_list and pi_list must have the same length"
    N = P_list[0].shape[0]
    assert all(P.shape == (N, N) for P in P_list), "All P must be N x N"
    assert all(len(pi) == N for pi in pi_list), "All pi must length N"

    # Prepare RNG
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    log_base_factor = math.log(log_base)
    total_nll = 0.0

    # Precompute arrays for speed
    P_list_arr = [np.asarray(P, dtype=np.float64) for P in P_list]
    pi_list_arr = [np.asarray(pi, dtype=np.float64) for pi in pi_list]

    list_nll = []

    for _ in range(M):
        # 1. Sample a component uniformly and then a sequence from that component
        c = rng.integers(K)  # latent class index
        P_c, pi_c = P_list_arr[c], pi_list_arr[c]

        # Sample sequence x[0:T)
        x = np.empty(T, dtype=np.int64)
        x[0] = rng.choice(N, p=pi_c)
        for t in range(1, T):
            x[t] = rng.choice(N, p=P_c[x[t - 1]])

        # 2. Compute per-token mixture probabilities to obtain NLL
        #    Maintain running prefix probabilities for each component to avoid
        #    recomputation from scratch at every position.
        prefix_probs = np.array([pi[x[0]] for pi in pi_list_arr], dtype=np.float64)
        sum_prev = (
            prefix_probs.mean()
        )  # mixture probability of x_0 (uniform weights -> divide by K)

        nll_seq = 0.0
        for t in range(1, T):
            for k in range(K):
                prefix_probs[k] *= P_list_arr[k][x[t - 1], x[t]]
            sum_prefix = prefix_probs.mean()

            # Guard against zero probability 
            if sum_prev > 0 and sum_prefix > 0:
                cond_prob = sum_prefix / sum_prev
                nll_seq += -math.log(cond_prob) / log_base_factor
            sum_prev = sum_prefix

        total_nll += nll_seq
        list_nll.append(nll_seq)

    mean_nll = total_nll / M if M > 0 else float("nan")
    assert np.round(mean_nll, 4) == np.round(np.mean(list_nll), 4)
    std_nll = np.std(list_nll)

    return mean_nll, std_nll


def custom_collate_fn(batch):
    """
    Custom collation function that handles variable-length sequences by padding.
    """
    if isinstance(batch[0], dict):
        # Get all keys
        keys = batch[0].keys()
        result = {}

        for key in keys:
            if key == "latent_variable":
                continue

            # Skip if not tensors or if empty
            if not isinstance(batch[0][key], torch.Tensor):
                result[key] = [d[key] for d in batch]
                continue

            # Get max length for this key
            max_len = max([item[key].size(0) for item in batch])

            # Prepare padded tensors
            padded_tensors = []
            for item in batch:
                tensor = item[key]
                if tensor.size(0) < max_len:
                    # Pad with zeros (or whatever padding value is appropriate)
                    padding = torch.zeros(
                        max_len - tensor.size(0),
                        *tensor.size()[1:],
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )
                    padded_tensor = torch.cat([tensor, padding], dim=0)
                else:
                    padded_tensor = tensor
                padded_tensors.append(padded_tensor)

            # Stack padded tensors
            result[key] = torch.stack(padded_tensors)

        return result
    else:
        return default_collate(batch)


def train_and_save(
    n_hyps,
    my_args,
    plot=False,
    train_dataset=None,
    eval_dataset=None,
    data_collator=None,
    model_weights=None,
    log_transition_matrices_during_training=False,
    results_folder=None,
):

    if n_hyps == 2 and my_args["divide_rank_mcl"] is True:
        lora_r = 32
        lora_alpha = 32
    elif n_hyps == 1 and my_args["multiply_rank_1_hyp"] is True:
        lora_r = 128
        lora_alpha = 128
    else:
        lora_r = my_args["lora_r"]
        lora_alpha = my_args["lora_alpha"]

    seed = my_args["seed"]
    vocab_size = my_args["vocab_size"]
    hidden_size = my_args["hidden_size"]
    n_layer = my_args["n_layer"]
    n_epochs = my_args["n_epochs"]
    n_head = my_args["n_head"]
    N_it = my_args["N_it"]
    batch_size = my_args["batch_size"]
    wta_mode = my_args["wta_mode"]
    wta_params_epsilon = my_args["wta_params_epsilon"]
    use_latent_variables = my_args["use_latent_variables"]
    lr_scheduler_type = my_args.get(
        "lr_scheduler_type", "cosine"
    )  # Default to cosine if not specified
    if use_latent_variables is True:
        transition_matrices = my_args["transition_matrices"]
    else:
        transition_matrix = my_args["transition_matrix"]

    eval_steps = my_args["eval_steps"]
    evaluation_strategy = my_args["evaluation_strategy"]

    if n_hyps == 1:
        wta_mode = "wta"

    metrics_callback = MetricsCallback()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = build_gpt_model(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        n_layer=n_layer,
        n_head=n_head,
        use_lora=my_args["use_lora"],
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=my_args["lora_dropout"],
        window_size=my_args["window_size"],
        num_hyps=n_hyps,
    ).to(device)

    if model_weights is not None:
        models.load_state_dict(model_weights[n_hyps])

    training_args = TrainingArguments(
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,  # Change this to 1 for evaluation
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        save_strategy="no",
        logging_steps=5,
        learning_rate=my_args["learning_rate"],
        weight_decay=1e-3,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=0.1,
        report_to="none",
        disable_tqdm=False,
        logging_first_step=True,
        logging_dir="./logs",
        # Add LoRA-specific arguments
        gradient_checkpointing=False,
    )

    trainer = MyTrainer(
        model=models,
        wta_mode=wta_mode,
        wta_params_epsilon=wta_params_epsilon,
        lr_scheduler_type=lr_scheduler_type,  # Pass the scheduler type to MyTrainer
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[metrics_callback],
        seq_len=my_args["seq_len"],
        normalize_loss_by_T=my_args["normalize_loss_by_T"],
        log_transition_matrices_during_training=log_transition_matrices_during_training,
        results_folder=results_folder,
        transition_matrices=transition_matrices if use_latent_variables else None,
        my_args=my_args,
    )

    # Train
    train = True
    if train is True:
        trainer.train()

        # Plot training and validation losses
        if plot is True:
            plt.figure(figsize=(10, 6))
            plt.plot(
                metrics_callback.steps,
                metrics_callback.train_losses,
                label="Training Loss",
            )

            # Plot evaluation loss on the same steps as it was recorded
            if metrics_callback.eval_losses:
                plt.plot(
                    metrics_callback.eval_steps,
                    metrics_callback.eval_losses,
                    "ro-",
                    label="Validation Loss",
                )

        if use_latent_variables is False:
            unigram_entropy = calculate_unigram_cross_entropy(transition_matrix)
            if plot is True:
                plt.axhline(
                    y=unigram_entropy,
                    color="b",
                    linestyle="--",
                    label=f"Unigram Entropy: {unigram_entropy:.4f}",
                )

            # Get the optimal loss
            optimal_loss = calculate_entropy_rate(transition_matrix)
            # print(f"Optimal Loss (Entropy Rate): {optimal_loss:.4f}")
            # Add horizontal line for optimal loss
            if plot is True:
                plt.axhline(
                    y=optimal_loss,
                    color="g",
                    linestyle="--",
                    label=f"Optimal Loss: {optimal_loss:.4f}",
                )

        elif use_latent_variables is True:
            entropy_rates, stationary_distributions = calculate_entropy_rates(
                train_dataset.transition_matrices
            )
            for i, (entropy_rate, stationary_distribution) in enumerate(
                zip(entropy_rates, stationary_distributions)
            ):
                if plot is True:
                    plt.axhline(
                        y=entropy_rate,
                        color="m",
                        linestyle="--",
                        label=f"Entropy Rate {i}: {entropy_rate:.4f}",
                    )

        #######
        # Create a dummy model that uses the transition matrix directly
        # Create and evaluate the Markov model
        # print("\n--- Evaluating Markov Model with Exact Transition Matrix and Stationnary Distribution ---")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if use_latent_variables is False:
            markov_model = MarkovModel(transition_matrix)
            markov_model_mh = MarkovMHModel(train_dataset.transition_matrices)
            stationnary_distribution_model = StationnaryDistributionModel(
                transition_matrix
            )
            markov_model.to(device)
            markov_model_mh.to(device)
            stationnary_distribution_model.to(device)
            # Compute loss on evaluation dataset
            eval_dataloader = torch.utils.data.DataLoader(
                eval_dataset, batch_size=batch_size, collate_fn=data_collator
            )
            markov_model.eval()
            markov_model_mh.eval()
            stationnary_distribution_model.eval()
            total_loss = 0
            total_loss_mh = 0
            total_samples = 0
            total_stationnary_distribution_loss = 0
            with torch.no_grad():
                for batch in eval_dataloader:
                    # Move batch to the right device
                    batch = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }

                    # Extract input_ids and labels
                    input_ids = batch["input_ids"]
                    labels = batch["labels"]

                    outputs = markov_model(input_ids, labels)
                    outputs_mh = markov_model_mh(input_ids, labels)
                    stationnary_distribution_outputs = (
                        stationnary_distribution_model(input_ids, labels)
                    )
                    loss = outputs.loss
                    loss_mh = outputs_mh.loss

                    stationnary_distribution_loss = (
                        stationnary_distribution_outputs.loss
                    )
                    total_loss_mh += loss_mh.item() * input_ids.size(0)
                    total_loss += loss.item() * input_ids.size(0)
                    total_stationnary_distribution_loss += (
                        stationnary_distribution_loss.item() * input_ids.size(0)
                    )
                    total_samples += input_ids.size(0)

            markov_model_loss = total_loss / total_samples
            markov_model_mh_loss = total_loss_mh / total_samples
            stationnary_distribution_loss = (
                total_stationnary_distribution_loss / total_samples
            )
            #######

            if plot is True:
                plt.axhline(
                    y=markov_model_loss,
                    color="m",
                    linestyle="--",
                    label=f"Markov Model Loss: {markov_model_loss:.4f}",
                )
                plt.axhline(
                    y=stationnary_distribution_loss,
                    color="c",
                    linestyle="--",
                    label=f"Stationnary Distribution Loss: {stationnary_distribution_loss:.4f}",
                )

        elif use_latent_variables is True:

            transition_matrices_dict = {}
            transition_matrices_dict["mean"] = sum(
                train_dataset.transition_matrices
            ) / len(train_dataset.transition_matrices)

            losses = {}
            losses_mh = {}
            losses_wta = {}

            for key, transition_matrix in transition_matrices_dict.items():

                trained_model = models
                markov_model = MarkovModel(transition_matrix)
                markov_model_mh = MarkovMHModel(train_dataset.transition_matrices)
                markov_model.to(device)
                markov_model_mh.to(device)
                trained_model.to(device)

                # Compute loss on evaluation dataset
                eval_dataloader = torch.utils.data.DataLoader(
                    eval_dataset, batch_size=1, collate_fn=data_collator
                )
                markov_model.eval()
                markov_model_mh.eval()
                trained_model.eval()

                total_loss = 0
                total_loss_mh = 0
                total_samples = 0
                total_wta_loss = 0
                total_stationnary_distribution_loss = 0

                with torch.no_grad():
                    for batch in eval_dataloader:
                        # Move batch to the right device
                        batch = {
                            k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()
                        }

                        # Extract input_ids and labels
                        input_ids = batch["input_ids"]
                        labels = batch["labels"]

                        outputs = markov_model(input_ids, labels)
                        outputs_mh = markov_model_mh(input_ids, labels)

                        loss = outputs.loss
                        loss_mh = outputs_mh.loss

                        # Compute loss of trained model
                        stacked_losses = []
                        batch_size = batch["input_ids"].size(0)
                        for i in range(trained_model.num_hyps):
                            trained_model.set_adapter(f"lora{i}")
                            loss_wta_item = trainer.compute_loss(
                                trained_model, batch
                            )
                            stacked_losses.append(loss_wta_item)
                        stacked_losses = torch.stack(stacked_losses)
                        wta_loss = trainer.compute_loss_wta(
                            stacked_losses, mode="wta", epsilon=0.0
                        )

                        total_wta_loss += wta_loss.item() * input_ids.size(0)
                        total_loss += loss.item() * input_ids.size(0)
                        total_loss_mh += loss_mh.item() * input_ids.size(0)
                        total_samples += input_ids.size(0)

                markov_model_loss = total_loss / total_samples
                markov_model_mh_loss = total_loss_mh / total_samples

                wta_loss = total_wta_loss / total_samples
                losses[key] = markov_model_loss
                losses_mh[key] = markov_model_mh_loss
                losses_wta[key] = wta_loss

            if plot is True:
                plt.axhline(
                    y=markov_model_loss,
                    color="m",
                    linestyle="--",
                    label=f"Markov Model Loss: {markov_model_loss:.4f}",
                )

        if plot is True:
            plt.xlabel("Training Steps")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss vs. Optimal Loss")
            plt.legend()
            plt.grid(True)
            plt.show()
        print("Training finished for n_hyps = ", n_hyps)

    if use_latent_variables is True:
        return (
            metrics_callback.steps,
            metrics_callback.train_losses,
            metrics_callback.eval_steps,
            metrics_callback.eval_losses,
            metrics_callback.model_eval_losses,
            metrics_callback.model_eval_steps,
            models,
            entropy_rates,
            stationary_distributions,
            losses,
            losses_mh,
            losses_wta,
        )
    else:
        entropy_rates = []
        stationary_distributions = []
        return (
            metrics_callback.steps,
            metrics_callback.train_losses,
            metrics_callback.eval_steps,
            metrics_callback.eval_losses,
            metrics_callback.model_eval_losses,
            metrics_callback.model_eval_steps,
            markov_model_loss,
            stationnary_distribution_loss,
            unigram_entropy,
            optimal_loss,
            models,
            entropy_rates,
            stationary_distributions,
        )


def prepare_model_with_lora(model, r=8, alpha=32, dropout=0.1, num_hyps=1):
    """
    Prepare a model with multiple LoRA adapters.

    Args:
        model: The base model to add LoRA to
        r: LoRA attention dimension
        alpha: LoRA alpha parameter
        dropout: LoRA dropout
        num_hyps: Number of hypotheses (adapters) to create
    """
    # Create multiple LoRA configurations
    for i in range(num_hyps):
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            modules_to_save=None,
            init_lora_weights=True,
            layers_to_transform=None,
            layers_pattern=None,
        )
        peft_config.use_group_lora = False

        if i == 0:
            # First adapter
            model = get_peft_model(model, peft_config, adapter_name=f"lora{i}")
        else:
            # Add additional adapters
            model.add_adapter(f"lora{i}", peft_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model
