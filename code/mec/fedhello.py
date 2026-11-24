"""Fed-HeLLo-style utilities for the RoBERTa/GLUE baseline.

The reference method reserves a small server proxy set, uses randomized
geometrically-defined HLA during a warm-start period, and periodically
refreshes layer allocation probabilities with empirical Fisher information.
This module implements the proxy/FIM portion without depending on a specific
Transformers or PEFT version.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


def reserve_proxy_indices(
    num_examples: int,
    num_proxy: int,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Return deterministic, disjoint proxy and evaluation index lists."""
    n = int(num_examples)
    requested = int(num_proxy)
    if n < 2:
        raise ValueError("Fed-HeLLo proxy/evaluation split requires at least two examples")
    if requested < 1:
        raise ValueError("Fed-HeLLo proxy sample count must be positive")
    proxy_size = min(requested, n - 1)
    perm = np.random.RandomState(int(seed)).permutation(n).tolist()
    proxy = [int(x) for x in perm[:proxy_size]]
    evaluation = [int(x) for x in perm[proxy_size:]]
    return proxy, evaluation


def should_refresh_fim(
    round_num: int,
    rgd_rounds: int,
    fim_every: int,
    has_scores: bool,
) -> bool:
    """Whether the current 1-based round starts a new FIM-HLA interval."""
    t = int(round_num)
    warm = max(0, int(rgd_rounds))
    every = max(1, int(fim_every))
    first_fim_round = warm + 1
    if t < first_fim_round:
        return False
    return (not bool(has_scores)) or ((t - first_fim_round) % every == 0)


def empirical_fim_scores(
    *,
    model: nn.Module,
    lora_layers: Mapping[str, object],
    dataloader: Iterable,
    device: torch.device,
    max_samples: int,
) -> Tuple[Dict[str, float], int]:
    """Compute per-LoRA-module empirical Fisher scores on a server proxy set.

    Each example contributes the squared L2 norm of the supervised-loss
    gradient for both LoRA factors.  The result is averaged over examples,
    matching the empirical-FIM definition used by Fed-HeLLo.
    """
    limit = int(max_samples)
    if limit < 1:
        raise ValueError("max_samples must be positive")
    layers = dict(lora_layers)
    if len(layers) == 0:
        raise ValueError("No LoRA layers were provided for FIM computation")

    parameters = list(model.parameters())
    previous_requires_grad = [bool(p.requires_grad) for p in parameters]
    was_training = bool(model.training)
    for p in parameters:
        p.requires_grad = False
    for layer in layers.values():
        layer.lora_A.requires_grad = True
        layer.lora_B.requires_grad = True

    model.eval()
    criterion = nn.CrossEntropyLoss()
    totals = {str(name): 0.0 for name in layers}
    seen = 0
    try:
        for inputs, targets in dataloader:
            batch_size = int(targets.shape[0])
            for row in range(batch_size):
                if seen >= limit:
                    break
                if isinstance(inputs, Mapping):
                    sample_inputs = {
                        str(key): value[row : row + 1].to(device)
                        for key, value in dict(inputs).items()
                    }
                else:
                    sample_inputs = inputs[row : row + 1].to(device)
                sample_target = targets[row : row + 1].to(device)
                model.zero_grad(set_to_none=True)
                output = model(**sample_inputs) if isinstance(sample_inputs, Mapping) else model(sample_inputs)
                logits = getattr(output, "logits", output)
                loss = criterion(logits, sample_target)
                loss.backward()
                for name, layer in layers.items():
                    score = 0.0
                    for factor in (layer.lora_A, layer.lora_B):
                        grad = factor.grad
                        if grad is not None:
                            score += float(torch.sum(grad.detach().float() ** 2).item())
                    totals[str(name)] += score
                seen += 1
            if seen >= limit:
                break
    finally:
        model.zero_grad(set_to_none=True)
        for p, flag in zip(parameters, previous_requires_grad):
            p.requires_grad = bool(flag)
        model.train(was_training)

    if seen == 0:
        raise ValueError("Fed-HeLLo proxy dataloader yielded no examples")
    return {name: value / float(seen) for name, value in totals.items()}, int(seen)
