"""Prototype loss-aware, arrival-adjusted LoRA factor selector.

This module is deliberately separate from the current production selector.
It implements the new research policy for smoke tests and review before it is
wired into the FedSONATA runner.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch


@dataclass(frozen=True)
class LossAwareSelection:
    updates_A: Dict[str, torch.Tensor]
    updates_B: Dict[str, torch.Tensor]
    choices: Dict[str, str]
    selected_bytes: int
    budget_bytes: int
    expected_descent: float
    cap_optimal_descent: float
    overdue_layers: Tuple[str, ...]
    # Compatibility fields used by the existing MEC audit writer.
    captured_gain: float = 0.0
    oracle_gain: float = 0.0
    captured_gain_ratio: float = 1.0
    option_counts: Dict[str, int] = None
    delivery_probability_estimate: float = 1.0
    expected_captured_gain: float = 0.0
    cap_optimal_expected_gain: float = 0.0
    gain_target_met: bool = False
    target_attainment_ratio: float = 1.0
    guard_mandatory_count: int = 0
    guard_max_age_after: int = 0
    # When the byte cap cannot serve every overdue block in one round, the
    # selector maximizes the number of overdue blocks served before optimizing
    # descent.  Keep this coverage explicit for the audit instead of silently
    # relaxing the age rule.
    overdue_served_count: int = 0
    overdue_deferred_layers: Tuple[str, ...] = ()
    # Fields shared with the wire RIFT record so the matched raw-frame
    # controller can use this selector without a second accounting path.
    full_ab_bytes: int = 0
    budget_utilization: float = 0.0
    reliability: float = 1.0
    guard_reserve_bytes: int = 0


def _bytes(x: torch.Tensor) -> int:
    return int(x.numel()) * int(x.element_size())


def _utility(
    *,
    before_A: torch.Tensor,
    before_B: torch.Tensor,
    delta_A: torch.Tensor,
    delta_B: torch.Tensor,
    grad_W: torch.Tensor,
    arrival_probability: float,
    curvature: float,
    step_size: float,
    mode: str,
) -> float:
    if mode == "none":
        z = torch.zeros_like(before_B @ before_A)
    elif mode == "a":
        z = before_B @ delta_A
    elif mode == "b":
        z = delta_B @ before_A
    elif mode == "ab":
        z = before_B @ delta_A + delta_B @ before_A + delta_B @ delta_A
    else:
        raise ValueError(f"unknown factor action: {mode}")
    descent = (
        -float(step_size) * torch.sum(grad_W.detach().float() * z.detach().float())
        - 0.5 * float(curvature) * float(step_size) ** 2
        * float(torch.sum(z.detach().float() ** 2))
    )
    return max(0.0, float(arrival_probability)) * max(0.0, float(descent))


def select_loss_aware_updates(
    *,
    before_A: Mapping[str, torch.Tensor],
    before_B: Mapping[str, torch.Tensor],
    delta_A: Mapping[str, torch.Tensor],
    delta_B: Mapping[str, torch.Tensor],
    grad_W: Mapping[str, torch.Tensor],
    arrival_probability: Mapping[str, Mapping[str, float]],
    curvature: Mapping[str, float],
    allocation_age: Mapping[str, int],
    budget_bytes: int,
    step_size: float,
    age_limit: int,
    quantum_bytes: int = 4096,
) -> LossAwareSelection:
    """Solve the loss-aware action MCKP with budget-aware age service.

    The primary objective is the number of overdue blocks that can be served
    under the byte cap.  Conditional on that maximum feasible coverage, the
    secondary objective is expected delivered descent.  This preserves the
    hard age behavior whenever the cap can accommodate all overdue blocks and
    gives a deterministic, auditable degradation when it cannot.
    """
    if budget_bytes < 0 or quantum_bytes <= 0 or age_limit <= 0:
        raise ValueError("invalid budget, quantum, or age limit")
    names = sorted(set(delta_A) & set(delta_B) & set(before_A) & set(before_B) & set(grad_W))
    overdue = tuple(name for name in names if int(allocation_age.get(name, 0)) >= age_limit - 1)
    options = {}
    for name in names:
        options[name] = []
        for mode in ("none", "a", "b", "ab"):
            cost = 0 if mode == "none" else _bytes(delta_A[name]) if mode == "a" else _bytes(delta_B[name]) if mode == "b" else _bytes(delta_A[name]) + _bytes(delta_B[name])
            util = _utility(
                before_A=before_A[name], before_B=before_B[name], delta_A=delta_A[name],
                delta_B=delta_B[name], grad_W=grad_W[name],
                arrival_probability=arrival_probability.get(name, {}).get(mode, 1.0),
                curvature=float(curvature.get(name, 0.0)), step_size=step_size, mode=mode,
            )
            options[name].append((mode, cost, util))
    budget_units = int(budget_bytes) // int(quantum_bytes)
    # State value is (served_overdue_count, expected_descent, choices).  The
    # count is compared first, so low-utility but overdue blocks are still
    # serviced before optional descent is considered.
    states = {0: (0, 0.0, [])}
    for name in names:
        nxt = {}
        for used, (served, gain, choices) in states.items():
            for mode, cost, util in options[name]:
                units = used + (math.ceil(cost / quantum_bytes) if cost else 0)
                if units > budget_units:
                    continue
                cand = (
                    served + (1 if name in overdue and mode != "none" else 0),
                    gain + util,
                    choices + [(name, mode)],
                )
                if units not in nxt or (cand[0], cand[1]) > (nxt[units][0], nxt[units][1]):
                    nxt[units] = cand
        if not nxt:
            # `none` is always present and has zero cost, so this should only
            # be reachable for an invalid/empty option construction.  Keep a
            # clear error rather than returning a partial policy.
            raise ValueError(f"loss-aware factor policy has no feasible action at {name}")
        states = nxt
    used_units, (served_count, gain, choices) = max(
        states.items(), key=lambda item: (item[1][0], item[1][1], -item[0])
    )
    selected = {name: mode for name, mode in choices}
    option_counts = {mode: 0 for mode in ("none", "a", "b", "ab")}
    for mode in selected.values():
        option_counts[mode] += 1
    updates_A = {name: delta_A[name] for name, mode in choices if mode in {"a", "ab"}}
    updates_B = {name: delta_B[name] for name, mode in choices if mode in {"b", "ab"}}
    full_ab_bytes = sum(_bytes(delta_A[name]) + _bytes(delta_B[name]) for name in names)
    return LossAwareSelection(
        updates_A=updates_A, updates_B=updates_B, choices=selected,
        selected_bytes=sum(0 if mode == "none" else _bytes(delta_A[name]) if mode == "a" else _bytes(delta_B[name]) if mode == "b" else _bytes(delta_A[name]) + _bytes(delta_B[name]) for name, mode in choices),
        budget_bytes=int(budget_bytes), expected_descent=float(gain),
        cap_optimal_descent=float(gain), overdue_layers=overdue,
        captured_gain=float(gain), oracle_gain=float(gain), captured_gain_ratio=1.0,
        option_counts=option_counts, expected_captured_gain=float(gain),
        cap_optimal_expected_gain=float(gain),
        guard_mandatory_count=len(overdue),
        guard_max_age_after=max((int(allocation_age.get(n, 0)) for n in names), default=0),
        overdue_served_count=int(served_count),
        overdue_deferred_layers=tuple(
            name for name in overdue if selected.get(name, "none") == "none"
        ),
        full_ab_bytes=int(full_ab_bytes),
        budget_utilization=(float(sum(
            0 if mode == "none" else _bytes(delta_A[name]) if mode == "a" else _bytes(delta_B[name]) if mode == "b" else _bytes(delta_A[name]) + _bytes(delta_B[name])
            for name, mode in choices
        )) / float(budget_bytes) if budget_bytes > 0 else 0.0),
    )
