"""Edge-prioritized stochastic factor sampling for raw-wire LoRA updates."""
from __future__ import annotations

import math
import re
from typing import Mapping, Sequence

import numpy as np
import torch


EDGE_PRIOR_TAU = 0.22
EDGE_PRIOR_FLOOR = 0.25


def edge_prior_weights(
    names: Sequence[str],
    tau: float = EDGE_PRIOR_TAU,
    floor: float = EDGE_PRIOR_FLOOR,
) -> dict:
    """Return mean-one U-shaped depth weights.

    Transformer depth is extracted from common ``layers.N``/``blocks.N``/
    ``layer.N`` names.  Unknown backends fall back to sorted block order.  The
    fixed two-sided exponential prior favors both input- and output-adjacent
    blocks without reading accuracy or validation data.
    """
    ordered = sorted(str(name) for name in names)
    if not ordered:
        return {}
    depths = {}
    for ordinal, name in enumerate(ordered):
        match = re.search(r"(?:layers?|blocks?|h)\.(\d+)", name.lower())
        depths[name] = int(match.group(1)) if match else ordinal
    lo, hi = min(depths.values()), max(depths.values())
    span = float(max(1, hi - lo))
    tau = max(1e-6, float(tau))
    raw = {}
    for name in ordered:
        position = (float(depths[name]) - float(lo)) / span
        raw[name] = float(floor) + 0.5 * (
            math.exp(-position / tau) + math.exp(-(1.0 - position) / tau)
        )
    normalizer = sum(raw.values()) / float(len(raw))
    return {name: value / max(1e-12, normalizer) for name, value in raw.items()}


def sample_edge_weighted_atoms(
    *,
    delta_a: Mapping[str, torch.Tensor],
    delta_b: Mapping[str, torch.Tensor],
    budget_bytes: int,
    header_reserve: int,
    quantum_bytes: int,
    seed: int,
    client_id: int,
    round_num: int,
):
    """Weighted random without-replacement A/B atom selection under a cap."""
    names = sorted(set(delta_a).intersection(delta_b))
    weights = edge_prior_weights(names)
    rng = np.random.default_rng(
        int(seed) * 1000003 + int(client_id) * 997 + int(round_num) * 7919 + 104729
    )
    remaining = max(0, (int(budget_bytes) - int(header_reserve)) // int(quantum_bytes))
    atoms = []
    for factor, tensors in (("A", delta_a), ("B", delta_b)):
        for name in names:
            tensor = tensors[name]
            units = int(math.ceil(
                float(tensor.numel() * tensor.element_size()) / float(quantum_bytes)
            ))
            atoms.append((factor, name, units, float(weights[name])))
    selected_a, selected_b = {}, {}
    while atoms:
        feasible = [atom for atom in atoms if atom[2] <= remaining]
        if not feasible:
            break
        probabilities = np.asarray(
            [max(1e-12, atom[3]) for atom in feasible], dtype=np.float64
        )
        probabilities /= probabilities.sum()
        index = int(rng.choice(len(feasible), p=probabilities))
        factor, name, units, weight = feasible[index]
        tensor = delta_a[name] if factor == "A" else delta_b[name]
        (selected_a if factor == "A" else selected_b)[name] = tensor
        remaining -= units
        atoms.remove((factor, name, units, weight))
    choices = {
        name: (
            "ab" if name in selected_a and name in selected_b
            else "a" if name in selected_a
            else "b" if name in selected_b else "none"
        )
        for name in names
    }
    return selected_a, selected_b, choices, weights
