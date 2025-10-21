from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from local_transformers.activations import ACT2FN

def _mixtral_load_balancing_loss_func(
    gate_logits: torch.Tensor,
    num_experts: int,
    top_k: int,
    attention_mask: Optional[torch.Tensor] = None,
) -> float:
    routing_weights = torch.nn.functional.softmax(gate_logits, dim=-1) # [B*num_layers*S, E]

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1) # [B*num_layers*S, top_k]

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts) # [B*num_layers*S, top_k, E]

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = routing_weights.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand(
                (num_hidden_layers, batch_size, sequence_length, top_k, num_experts)
            )
            .reshape(-1, top_k, num_experts)
            .to(routing_weights.device) # [B*num_layers*S, top_k, E]
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(
            expert_mask.float() * expert_attention_mask, dim=0
        ) / torch.sum(expert_attention_mask, dim=0) # [top_k, E]

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(routing_weights.device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(
            routing_weights * router_per_expert_attention_mask, dim=0
        ) / torch.sum(router_per_expert_attention_mask, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class MixtralRouterLoss(torch.nn.Module):
    def __init__(self, router_aux_loss_coef, num_experts, top_k) -> None:
        super().__init__()
        self.aux_loss_coef = router_aux_loss_coef
        self.experts = num_experts
        self.topk = top_k

    def forward(self, gate_logits, attention_mask) -> torch.Tensor:
        return self.aux_loss_coef * _mixtral_load_balancing_loss_func(
            gate_logits, self.experts, self.topk, attention_mask
        )


def _dynamic_top_p(router_logits: torch.Tensor, top_p: float, temperature: float = 0.0):
    if temperature > 0.0:
        router_logits = router_logits / temperature
    sorted_logits, sorted_indices = torch.sort(router_logits, dim=-1, descending=True)
    cumulative_probs = sorted_logits.cumsum(dim=-1)
    expert_mask = cumulative_probs > top_p
    threshold_indices = expert_mask.long().argmax(dim=-1)
    threshold_mask = torch.nn.functional.one_hot(
        threshold_indices, num_classes=sorted_indices.size(-1)
    ).bool()
    expert_mask = expert_mask & ~threshold_mask
    sorted_logits = sorted_logits.masked_fill(expert_mask, 0.0)
    sorted_indices = sorted_indices.masked_fill(expert_mask, -1)
    return sorted_logits, sorted_indices


def _dynamic_load_balancing_loss_func(
    routing_weights: torch.Tensor,
    num_experts: int,
    top_p: float,
    temperature: float,
) -> float:
    _, selected_experts = _dynamic_top_p(routing_weights, top_p, temperature)

    expert_mask = torch.empty(
        (num_experts, num_experts, routing_weights.size(0)),
        dtype=routing_weights.dtype,
        device=routing_weights.device,
    )

    for expert_idx in range(num_experts):
        expert_mask[expert_idx] = (selected_experts == expert_idx).transpose(0, 1)

    expert_mask = expert_mask.permute(2, 1, 0)

    # Compute the percentage of tokens routed to each experts
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    # Compute the average probability of routing to these experts
    router_prob_per_expert = torch.mean(routing_weights, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class DynamicRouterLoss(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.aux_loss_coef = config.router_aux_loss_coef
        self.experts = config.num_experts
        self.top_p = config.top_p_
        self.temperature = config.temperature_

    def forward(self, gate_logits, attention_mask) -> torch.Tensor:
        routing_weights = torch.nn.functional.softmax(gate_logits, dim=-1)
        return self.aux_loss_coef * _dynamic_load_balancing_loss_func(
            routing_weights,
            self.experts,
            self.top_p,
            self.temperature,
        )

def _switch_router_z_loss_func(router_logits: torch.Tensor) -> float:
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = log_z**2
    return torch.sum(z_loss) / (router_logits.size(0))


def _switch_load_balancing_loss_func(router_probs: torch.Tensor) -> float:
    num_experts = router_probs.size(-1)

    expert_mask = torch.argmax(router_probs, dim=-1)
    expert_mask = torch.nn.functional.one_hot(expert_mask, num_classes=num_experts)

    tokens_per_group_and_expert = torch.mean(expert_mask.float(), dim=0)

    router_prob_per_group_and_expert = torch.mean(router_probs, dim=0)
    return torch.mean(
        tokens_per_group_and_expert * router_prob_per_group_and_expert
    ) * (num_experts**2)


class SwitchRouterLoss(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.experts = config.num_experts
        self.expert_capacity_ = config.expert_capacity_
        self.z_loss_coef = config.router_z_loss_coef_
        self.aux_loss_coef = config.router_aux_loss_coef

    def forward(self, router_logits, attention_mask) -> torch.Tensor:
        z_loss = _switch_router_z_loss_func(router_logits)
        router_probs = F.softmax(router_logits, dim=-1)
        # recompute expert indexes due to MoE-PEFT constraints
        aux_loss = _switch_load_balancing_loss_func(router_probs)
        return self.z_loss_coef * z_loss + self.aux_loss_coef * aux_loss