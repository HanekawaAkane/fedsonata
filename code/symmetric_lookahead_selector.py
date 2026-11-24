"""Symmetric two-step complementarity-aware LoRA factor selector.

The selector scores an action by the expected delivered descent of the current
product update plus the next-step complementary product.  A small dual service
debt keeps repeatedly transmitting one factor from starving its counterpart.
The latter is a constraint surrogate, not a random action generator.
"""
from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Tuple, Union

import torch

from mec.loss_aware_factor_selector import LossAwareSelection


def _bytes(x: torch.Tensor) -> int:
    return int(x.numel()) * int(x.element_size())


def _action_z(before_a, before_b, delta_a, delta_b, mode):
    if mode == "none":
        return torch.zeros_like(before_b @ before_a)
    if mode == "a":
        return before_b @ delta_a
    if mode == "b":
        return delta_b @ before_a
    if mode == "ab":
        return before_b @ delta_a + delta_b @ before_a + delta_b @ delta_a
    raise ValueError(mode)


def _future_z(delta_a, delta_b, mode):
    """One-step persistence forecast of the omitted complementary factor."""
    if mode == "a":
        return delta_b @ delta_a
    if mode == "b":
        return delta_b @ delta_a
    return torch.zeros_like(delta_b @ delta_a)


def _utility(
    *, before_a, before_b, delta_a, delta_b, grad_w, arrival_probability,
    step_size, future_weight, curvature, mode, balance_debt, balance_scale,
):
    z_now = _action_z(before_a, before_b, delta_a, delta_b, mode)
    z_future = _future_z(delta_a, delta_b, mode)
    z = z_now + float(future_weight) * z_future
    descent = (
        -float(step_size) * torch.sum(grad_w.detach().float() * z.detach().float())
        -0.5 * float(curvature) * float(step_size) ** 2
        * float(torch.sum(z.detach().float() ** 2))
    )
    delivered = max(0.0, float(arrival_probability)) * max(0.0, float(descent))
    # Factor service debt is positive after excess A service and negative after
    # excess B service.  A/B actions change the debt by +1/-1; AB resolves a
    # per-block debt in one transmission.
    delta_debt = 1 if mode == "a" else -1 if mode == "b" else 0
    debt = int(balance_debt)
    post_debt = 0 if mode == "ab" and debt != 0 else debt + delta_debt
    dual = -float(balance_scale) * (abs(float(post_debt)) - abs(float(debt)))
    return delivered + dual


def select_symmetric_lookahead_updates(
    *,
    before_A: Mapping[str, torch.Tensor],
    before_B: Mapping[str, torch.Tensor],
    delta_A: Mapping[str, torch.Tensor],
    delta_B: Mapping[str, torch.Tensor],
    grad_W: Mapping[str, torch.Tensor],
    arrival_probability: Mapping[str, Mapping[str, float]],
    budget_bytes: int,
    step_size: float,
    quantum_bytes: int = 4096,
    future_weight: float = 1.0,
    balance_debt: Union[int, Mapping[str, int]] = 0,
    curvature: Optional[Mapping[str, float]] = None,
) -> LossAwareSelection:
    if budget_bytes < 0 or quantum_bytes <= 0:
        raise ValueError("invalid budget or quantum")
    names = sorted(set(before_A) & set(before_B) & set(delta_A) & set(delta_B) & set(grad_W))
    if not names:
        raise ValueError("symmetric lookahead selector received no candidates")
    curvature = curvature or {}
    # Normalize the dual term from the observed action-utility scale.  This is
    # invariant to tensor dimension and learning-rate units, while making one
    # unit of accumulated factor imbalance comparable to one typical action.
    base_values = []
    for name in names:
        for mode in ("a", "b", "ab"):
            base_values.append(abs(_utility(
                before_a=before_A[name], before_b=before_B[name],
                delta_a=delta_A[name], delta_b=delta_B[name], grad_w=grad_W[name],
                arrival_probability=arrival_probability.get(name, {}).get(mode, 1.0),
                step_size=step_size, future_weight=future_weight,
                curvature=float(curvature.get(name, 0.0)), mode=mode,
                balance_debt=0, balance_scale=0.0,
            )))
    base_values.sort()
    balance_scale = base_values[len(base_values) // 2] if base_values else 0.0
    options = {}
    for name in names:
        options[name] = []
        for mode in ("none", "a", "b", "ab"):
            cost = (
                0 if mode == "none" else _bytes(delta_A[name]) if mode == "a"
                else _bytes(delta_B[name]) if mode == "b"
                else _bytes(delta_A[name]) + _bytes(delta_B[name])
            )
            util = _utility(
                before_a=before_A[name], before_b=before_B[name],
                delta_a=delta_A[name], delta_b=delta_B[name], grad_w=grad_W[name],
                arrival_probability=arrival_probability.get(name, {}).get(mode, 1.0),
                step_size=step_size, future_weight=future_weight,
                curvature=float(curvature.get(name, 0.0)), mode=mode,
                balance_debt=(
                    int(balance_debt.get(name, 0))
                    if isinstance(balance_debt, Mapping)
                    else int(balance_debt)
                ), balance_scale=balance_scale,
            )
            options[name].append((mode, cost, util))
    budget_units = int(budget_bytes) // int(quantum_bytes)
    states = {0: (0.0, [])}
    for name in names:
        nxt = {}
        for used, (score, choices) in states.items():
            for mode, cost, util in options[name]:
                units = used + (math.ceil(cost / quantum_bytes) if cost else 0)
                if units > budget_units:
                    continue
                candidate = (score + util, choices + [(name, mode)])
                if units not in nxt or candidate[0] > nxt[units][0]:
                    nxt[units] = candidate
        if not nxt:
            raise ValueError(f"symmetric lookahead policy infeasible at {name}")
        states = nxt
    used_units, (score, choices) = max(states.items(), key=lambda item: (item[1][0], -item[0]))
    selected = {name: mode for name, mode in choices}
    counts = {mode: 0 for mode in ("none", "a", "b", "ab")}
    for mode in selected.values():
        counts[mode] += 1
    selected_bytes = sum(
        0 if mode == "none" else _bytes(delta_A[name]) if mode == "a"
        else _bytes(delta_B[name]) if mode == "b"
        else _bytes(delta_A[name]) + _bytes(delta_B[name])
        for name, mode in choices
    )
    full_ab_bytes = sum(_bytes(delta_A[name]) + _bytes(delta_B[name]) for name in names)
    return LossAwareSelection(
        updates_A={name: delta_A[name] for name, mode in choices if mode in {"a", "ab"}},
        updates_B={name: delta_B[name] for name, mode in choices if mode in {"b", "ab"}},
        choices=selected,
        selected_bytes=int(selected_bytes), budget_bytes=int(budget_bytes),
        expected_descent=float(score), cap_optimal_descent=float(score),
        overdue_layers=(), captured_gain=float(score), oracle_gain=float(score),
        captured_gain_ratio=1.0, option_counts=counts,
        expected_captured_gain=float(score), cap_optimal_expected_gain=float(score),
        full_ab_bytes=int(full_ab_bytes), budget_utilization=(selected_bytes / budget_bytes if budget_bytes else 0.0),
        reliability=1.0,
    )
