"""FedSONATA: product-space, reliability-aware LoRA factor selection.

RIFT chooses one of {none, A, B, AB} for every allocated LoRA block.  Unlike a
raw delta-norm threshold, its value is the reduction in effective-weight error
relative to the client's full local LoRA update.  A conservative quantized
multiple-choice knapsack enforces the byte budget exactly.

FedSONATA additionally supports a convergence-gain target.  It first finds the
maximum expected captured gain under the reliability-scaled byte cap, then
returns the smallest conservative quantized payload that retains a declared
fraction of that cap-optimal gain.
"""

from __future__ import annotations

import os as _os
import random as _random_mod

_RANDOM_RNG = None


def _rift_random_rng():
    """Deterministic RNG for the D061 randact ablation (RIFT_SELECTION=random)."""
    global _RANDOM_RNG
    if _RANDOM_RNG is None:
        _RANDOM_RNG = _random_mod.Random(int(_os.environ.get("RIFT_SELECTION_RANDOM_SEED", "0")))
    return _RANDOM_RNG

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import torch


@dataclass(frozen=True)
class RiftSelection:
    updates_A: Dict[str, torch.Tensor]
    updates_B: Dict[str, torch.Tensor]
    choices: Dict[str, str]
    full_ab_bytes: int
    budget_bytes: int
    selected_bytes: int
    budget_utilization: float
    captured_gain: float
    oracle_gain: float
    captured_gain_ratio: float
    reliability: float
    option_counts: Dict[str, int]
    guard_mandatory_count: int = 0
    guard_reserve_bytes: int = 0
    delivery_probability_estimate: float = 1.0
    expected_captured_gain: float = 0.0
    gain_target_ratio: Optional[float] = None
    gain_target_met: bool = False
    cap_optimal_expected_gain: float = 0.0
    target_attainment_ratio: float = 1.0


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def oldest_guard_layers(
    names: List[str], ages: Mapping[str, int], max_age: int
) -> List[str]:
    """Choose a feasible-rate oldest-first service set for RIFT-Guard."""
    if int(max_age) <= 0:
        raise ValueError("max_age must be positive")
    ordered = sorted(set(names), key=lambda name: (-int(ages.get(name, 0)), name))
    if not ordered:
        return []
    service_count = int(math.ceil(len(ordered) / float(max_age)))
    return ordered[:service_count]


def _product_inner(
    x1: torch.Tensor,
    y1: torch.Tensor,
    x2: torch.Tensor,
    y2: torch.Tensor,
) -> torch.Tensor:
    """Frobenius inner product <X1 Y1, X2 Y2> without materializing products."""
    left = x1.detach().float().transpose(0, 1) @ x2.detach().float()
    right = y2.detach().float() @ y1.detach().float().transpose(0, 1)
    return torch.sum(left * right.transpose(0, 1))


def _sum_product_sq(components: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    value = None
    for i, (xi, yi) in enumerate(components):
        term = _product_inner(xi, yi, xi, yi)
        value = term if value is None else value + term
        for xj, yj in components[i + 1:]:
            value = value + 2.0 * _product_inner(xi, yi, xj, yj)
    if value is None:
        raise ValueError("at least one low-rank product component is required")
    return torch.clamp(value, min=0.0)


def _layer_options(
    name: str,
    a_after: torch.Tensor,
    b_after: torch.Tensor,
    delta_a: torch.Tensor,
    delta_b: torch.Tensor,
    staleness: float,
    staleness_weight: float,
) -> List[Tuple[str, int, float]]:
    a_before = a_after - delta_a
    b_before = b_after - delta_b
    if b_before.shape[1] != a_before.shape[0]:
        raise ValueError(f"{name}: incompatible LoRA shapes B={tuple(b_before.shape)} A={tuple(a_before.shape)}")

    # Z = B dA + dB A + dB dA.  All norms are evaluated through r-by-r
    # Gram products, avoiding large out_dim-by-in_dim materializations and the
    # thousands of GPU synchronization points of the direct implementation.
    effect_a = (b_before, delta_a)
    effect_b = (delta_b, a_before)
    cross = (delta_b, delta_a)
    target_sq_t = _sum_product_sq([effect_a, effect_b, cross])
    error_a_sq_t = _sum_product_sq([effect_b, cross])
    error_b_sq_t = _sum_product_sq([effect_a, cross])
    target_sq, error_a_sq, error_b_sq = (
        float(value.item()) for value in (target_sq_t, error_a_sq_t, error_b_sq_t)
    )
    weight = 1.0 + max(0.0, float(staleness_weight)) * max(0.0, float(staleness))
    bytes_a = _tensor_bytes(delta_a)
    bytes_b = _tensor_bytes(delta_b)
    options = [
        ("none", 0, 0.0),
        ("a", bytes_a, weight * max(0.0, target_sq - error_a_sq)),
        ("b", bytes_b, weight * max(0.0, target_sq - error_b_sq)),
        ("ab", bytes_a + bytes_b, weight * target_sq),
    ]
    return options


def _solve_quantized_layers(
    options_by_layer: List[Tuple[str, List[Tuple[str, int, float]]]],
    *,
    budget_units: int,
    quantum: int,
    require_positive: bool = False,
    delivery_rate_bps: Optional[float] = None,
    duration_hazard_rate: float = 0.0,
    gain_target_ratio: Optional[float] = None,
) -> Tuple[float, List[Tuple[str, str]], int, int, float, bool]:
    """Solve one quantized MCKP and return gain, choices, real bytes, units."""
    if not options_by_layer:
        return 0.0, [], 0, 0, 0.0, gain_target_ratio is not None
    states: Dict[int, Tuple[float, List[Tuple[str, str]], int]] = {0: (0.0, [], 0)}
    for name, options in options_by_layer:
        next_states: Dict[int, Tuple[float, List[Tuple[str, str]], int]] = {}
        allowed = [item for item in options if not require_positive or item[0] != "none"]
        for used_units, (base_gain, base_choices, base_bytes) in states.items():
            for mode, real_cost, gain in allowed:
                cost_units = int(math.ceil(real_cost / quantum)) if real_cost > 0 else 0
                total_units = used_units + cost_units
                if total_units > budget_units:
                    continue
                candidate = (
                    base_gain + float(gain),
                    base_choices + [(name, mode)],
                    base_bytes + int(real_cost),
                )
                previous = next_states.get(total_units)
                if previous is None or candidate[0] > previous[0] + 1e-18:
                    next_states[total_units] = candidate
        if not next_states:
            raise ValueError(
                f"quantized RIFT subproblem infeasible at layer {name}; "
                f"budget_units={budget_units}, require_positive={require_positive}"
            )
        pruned: Dict[int, Tuple[float, List[Tuple[str, str]], int]] = {}
        best_gain = -1.0
        for units in sorted(next_states):
            state = next_states[units]
            if state[0] > best_gain + 1e-18:
                pruned[units] = state
                best_gain = state[0]
        states = pruned
    hazard = float(duration_hazard_rate)
    if hazard > 0.0:
        if (
            delivery_rate_bps is None
            or not math.isfinite(float(delivery_rate_bps))
            or float(delivery_rate_bps) <= 0.0
        ):
            raise ValueError(
                "positive duration_hazard_rate requires positive delivery_rate_bps"
            )
        rate = float(delivery_rate_bps)

        def expected_gain(units: int, gain: float) -> float:
            conservative_bits = 8.0 * float(units * quantum)
            return float(gain) * math.exp(-hazard * conservative_bits / rate)
    else:
        def expected_gain(units: int, gain: float) -> float:
            del units
            return float(gain)

    cap_used_units, cap_best = max(
        states.items(),
        key=lambda item: (expected_gain(item[0], item[1][0]), -item[0]),
    )
    cap_optimal_expected_gain = expected_gain(cap_used_units, cap_best[0])
    target_met = False
    if gain_target_ratio is not None:
        ratio = float(gain_target_ratio)
        threshold = ratio * cap_optimal_expected_gain
        feasible = [
            (units, state)
            for units, state in states.items()
            if expected_gain(units, state[0]) + 1e-18 >= threshold
        ]
        if feasible:
            # The primary key proves minimum conservative quantized bytes.
            # Within the same byte bucket, retain the largest expected gain.
            used_units, best = min(
                feasible,
                key=lambda item: (
                    item[0],
                    -expected_gain(item[0], item[1][0]),
                    item[1][2],
                ),
            )
            target_met = True
        else:  # Numerically defensive: the cap optimum always meets ratio <= 1.
            used_units, best = cap_used_units, cap_best
    else:
        used_units, best = cap_used_units, cap_best
    gain, choices, real_bytes = best
    return (
        gain,
        choices,
        real_bytes,
        used_units,
        cap_optimal_expected_gain,
        target_met,
    )


def select_rift_updates(
    *,
    a_after: Mapping[str, torch.Tensor],
    b_after: Mapping[str, torch.Tensor],
    delta_a: Mapping[str, torch.Tensor],
    delta_b: Mapping[str, torch.Tensor],
    budget_ratio: float,
    reliability: float,
    staleness: Optional[Mapping[str, float]] = None,
    staleness_weight: float = 0.1,
    reliability_floor: float = 0.25,
    quantum_bytes: int = 4096,
    mandatory_layers: Optional[List[str]] = None,
    guard_reserve_ratio: float = 0.0,
    delivery_rate_bps: Optional[float] = None,
    duration_hazard_rate: float = 0.0,
    gain_target_ratio: Optional[float] = None,
) -> RiftSelection:
    """Select factor updates under a conservative reliability-scaled budget.

    Costs are rounded *up* to ``quantum_bytes`` while the budget is rounded
    down.  Therefore every returned selection respects the true byte budget.
    The dynamic program is exact for this quantized multiple-choice knapsack.
    """

    if not 0.0 < float(budget_ratio) <= 1.0:
        raise ValueError("budget_ratio must be in (0, 1]")
    if int(quantum_bytes) <= 0:
        raise ValueError("quantum_bytes must be positive")
    if not 0.0 <= float(reliability) <= 1.0:
        raise ValueError("reliability must be in [0, 1]")
    if not 0.0 < float(reliability_floor) <= 1.0:
        raise ValueError("reliability_floor must be in (0, 1]")
    if not 0.0 <= float(guard_reserve_ratio) <= 1.0:
        raise ValueError("guard_reserve_ratio must be in [0, 1]")
    if float(duration_hazard_rate) < 0.0:
        raise ValueError("duration_hazard_rate must be non-negative")
    if gain_target_ratio is not None and not 0.0 < float(gain_target_ratio) <= 1.0:
        raise ValueError("gain_target_ratio must be in (0, 1]")

    names = sorted(set(delta_a).intersection(delta_b))
    if not names:
        return RiftSelection(
            updates_A={}, updates_B={}, choices={}, full_ab_bytes=0,
            budget_bytes=0, selected_bytes=0, budget_utilization=0.0,
            captured_gain=0.0, oracle_gain=0.0, captured_gain_ratio=1.0,
            reliability=float(reliability),
            option_counts={"none": 0, "a": 0, "b": 0, "ab": 0},
            gain_target_ratio=(
                float(gain_target_ratio) if gain_target_ratio is not None else None
            ),
            gain_target_met=gain_target_ratio is not None,
        )
    missing = [name for name in names if name not in a_after or name not in b_after]
    if missing:
        raise KeyError(f"missing post-train factors for layers: {missing[:3]}")

    stale = staleness or {}
    options_by_layer = []
    full_ab_bytes = 0
    oracle_gain = 0.0
    for name in names:
        options = _layer_options(
            name,
            a_after[name],
            b_after[name],
            delta_a[name],
            delta_b[name],
            float(stale.get(name, 0.0)),
            float(staleness_weight),
        )
        options_by_layer.append((name, options))
        full_ab_bytes += next(cost for mode, cost, _ in options if mode == "ab")
        oracle_gain += max(gain for _, _, gain in options)

    rel_scale = max(float(reliability_floor), float(reliability))
    if _os.environ.get("RIFT_BUDGET_RELIABILITY", "1") == "0":
        rel_scale = 1.0
    budget_bytes = int(math.floor(full_ab_bytes * float(budget_ratio) * rel_scale))
    quantum = int(quantum_bytes)
    budget_units = budget_bytes // quantum

    if _os.environ.get("RIFT_SELECTION", "value") == "abornone":
        # Planner ablation: remove the factor-level {a, b} split and keep
        # only the block-level {none, ab} choice under the SAME byte budget.
        options_by_layer = [
            (name, [opt for opt in opts if opt[0] in {"none", "ab"}])
            for name, opts in options_by_layer
        ]

    mandatory = set(mandatory_layers or []).intersection(names)
    if mandatory and gain_target_ratio is not None:
        raise ValueError("gain-target selection cannot be combined with RIFT-Guard")
    if mandatory and float(guard_reserve_ratio) <= 0.0:
        raise ValueError("mandatory layers require a positive guard_reserve_ratio")
    guard_reserve_units = int(math.floor(budget_units * float(guard_reserve_ratio)))
    guard_reserve_bytes = guard_reserve_units * quantum
    if mandatory:
        mandatory_options = [item for item in options_by_layer if item[0] in mandatory]
        min_mandatory_units = sum(
            min(int(math.ceil(cost / quantum)) for mode, cost, _ in options if mode != "none")
            for _, options in mandatory_options
        )
        if min_mandatory_units > guard_reserve_units:
            raise ValueError(
                "RIFT-Guard reserve infeasible: "
                f"mandatory_min_units={min_mandatory_units} > reserve_units={guard_reserve_units}"
            )
        mandatory_gain, mandatory_choices, mandatory_bytes, mandatory_units, _, _ = _solve_quantized_layers(
            mandatory_options,
            budget_units=guard_reserve_units,
            quantum=quantum,
            require_positive=True,
        )
        residual_options = [item for item in options_by_layer if item[0] not in mandatory]
        residual_gain, residual_choices, residual_bytes, residual_units, _, _ = _solve_quantized_layers(
            residual_options,
            budget_units=budget_units - mandatory_units,
            quantum=quantum,
        )
        captured_gain = mandatory_gain + residual_gain
        choice_pairs = mandatory_choices + residual_choices
        selected_bytes = mandatory_bytes + residual_bytes
        selected_units = mandatory_units + residual_units
        cap_optimal_expected_gain = captured_gain
        gain_target_met = False
    else:
        (
            captured_gain,
            choice_pairs,
            selected_bytes,
            selected_units,
            cap_optimal_expected_gain,
            gain_target_met,
        ) = _solve_quantized_layers(
            options_by_layer,
            budget_units=budget_units,
            quantum=quantum,
            delivery_rate_bps=delivery_rate_bps,
            duration_hazard_rate=float(duration_hazard_rate),
            gain_target_ratio=gain_target_ratio,
        )
    if _os.environ.get("RIFT_SELECTION", "value") == "random":
        # D061 randact ablation: random feasible action allocation within the
        # SAME quantized byte budget (shuffle layers; uniform pick among
        # {none, a, b, ab}; drop actions that would exceed the budget).
        _rng = _rift_random_rng()
        _opt = dict(options_by_layer)
        _order = list(names)
        _rng.shuffle(_order)
        _used = 0
        choice_pairs = []
        for _name in _order:
            _pick = _rng.choice(("none", "a", "b", "ab"))
            if _pick == "none":
                continue
            _cost = next((c for m, c, _g in _opt[_name] if m == _pick), None)
            if _cost is None or _used + _cost > budget_bytes:
                continue
            choice_pairs.append((_name, _pick))
            _used += _cost
        selected_bytes = _used
        selected_units = (_used // quantum) if quantum > 0 else 0
        captured_gain = 0.0
        for _name, _pick in choice_pairs:
            captured_gain += next(g for m, c, g in _opt[_name] if m == _pick)
        cap_optimal_expected_gain = captured_gain
        gain_target_met = False
    choices = dict(choice_pairs)
    updates_A = {
        name: delta_a[name] for name, mode in choice_pairs if mode in {"a", "ab"}
    }
    updates_B = {
        name: delta_b[name] for name, mode in choice_pairs if mode in {"b", "ab"}
    }
    counts = {mode: 0 for mode in ("none", "a", "b", "ab")}
    for mode in choices.values():
        counts[mode] += 1
    if selected_bytes > budget_bytes:
        raise AssertionError(f"RIFT budget violated: {selected_bytes} > {budget_bytes}")
    if float(duration_hazard_rate) > 0.0:
        conservative_selected_bits = 8.0 * float(selected_units * quantum)
        delivery_probability_estimate = math.exp(
            -float(duration_hazard_rate)
            * conservative_selected_bits
            / float(delivery_rate_bps)
        )
    else:
        delivery_probability_estimate = 1.0
    expected_captured_gain = captured_gain * delivery_probability_estimate
    target_attainment_ratio = (
        expected_captured_gain / cap_optimal_expected_gain
        if cap_optimal_expected_gain > 0.0 else 1.0
    )
    return RiftSelection(
        updates_A=updates_A,
        updates_B=updates_B,
        choices=choices,
        full_ab_bytes=full_ab_bytes,
        budget_bytes=budget_bytes,
        selected_bytes=selected_bytes,
        budget_utilization=(selected_bytes / budget_bytes if budget_bytes > 0 else 0.0),
        captured_gain=captured_gain,
        oracle_gain=oracle_gain,
        captured_gain_ratio=(captured_gain / oracle_gain if oracle_gain > 0 else 1.0),
        reliability=float(reliability),
        option_counts=counts,
        guard_mandatory_count=len(mandatory),
        guard_reserve_bytes=guard_reserve_bytes if mandatory else 0,
        delivery_probability_estimate=delivery_probability_estimate,
        expected_captured_gain=expected_captured_gain,
        gain_target_ratio=(
            float(gain_target_ratio) if gain_target_ratio is not None else None
        ),
        gain_target_met=bool(gain_target_met),
        cap_optimal_expected_gain=float(cap_optimal_expected_gain),
        target_attainment_ratio=float(target_attainment_ratio),
    )
