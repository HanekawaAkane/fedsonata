"""Finite-horizon transmission scheduling for FedSONATA.

The scheduler is deliberately small and deterministic.  It optimizes the
binary high/low-fidelity allocation induced by a cumulative trajectory bound;
it does not claim to solve the non-convex federated-learning problem itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class HorizonAllocation:
    """Exact solution of the quantized binary horizon subproblem."""

    high_round_indices: Tuple[int, ...]
    weighted_error_reduction: float
    used_budget_units: int


@dataclass(frozen=True)
class FedSonataRoundPolicy:
    """Effective FedSONATA controls for one one-based training round."""

    schedule: str
    budget_ratio: float
    gain_target_ratio: Optional[float]
    high_fidelity: bool
    trajectory_weight: float


def trajectory_weights(num_rounds: int, contraction_q: float) -> Tuple[float, ...]:
    """Return cumulative-error weights for ``Delta[t+1] <= q Delta[t] + e[t]``.

    Summing ``Delta[1], ..., Delta[T]`` gives the error at zero-based round
    ``t`` the coefficient ``sum(q**j, j=0..T-1-t)``.  For ``q=1`` the stable
    limiting expression is ``T-t``.
    """

    horizon = int(num_rounds)
    q = float(contraction_q)
    if horizon <= 0:
        raise ValueError("num_rounds must be positive")
    if not 0.0 <= q <= 1.0:
        raise ValueError("contraction_q must be in [0, 1]")
    if abs(q - 1.0) <= 1e-12:
        return tuple(float(horizon - t) for t in range(horizon))
    return tuple((1.0 - q ** (horizon - t)) / (1.0 - q) for t in range(horizon))


def solve_binary_horizon_knapsack(
    *,
    weights: Sequence[float],
    extra_cost_units: Sequence[int],
    normalized_error_reduction: Sequence[float],
    budget_units: int,
) -> HorizonAllocation:
    """Exactly maximize weighted compression-error reduction.

    Each round either uses the low-fidelity endpoint or pays ``extra_cost`` for
    the high-fidelity endpoint.  The dynamic program is exact for integer cost
    units and has ``O(T B)`` time and memory complexity.  Ties prefer fewer
    units and then earlier rounds, which makes the protocol reproducible.
    """

    if not (
        len(weights) == len(extra_cost_units) == len(normalized_error_reduction)
    ):
        raise ValueError("horizon vectors must have identical lengths")
    if not weights:
        raise ValueError("horizon vectors must be non-empty")
    capacity = int(budget_units)
    if capacity < 0:
        raise ValueError("budget_units must be non-negative")
    if any(float(value) < 0.0 for value in weights):
        raise ValueError("weights must be non-negative")
    if any(int(value) <= 0 for value in extra_cost_units):
        raise ValueError("extra_cost_units must be positive integers")
    if any(float(value) < 0.0 for value in normalized_error_reduction):
        raise ValueError("normalized_error_reduction must be non-negative")

    # used units -> (weighted reduction, one-based selected rounds)
    states = {0: (0.0, ())}
    for index, (weight, cost, reduction) in enumerate(
        zip(weights, extra_cost_units, normalized_error_reduction), start=1
    ):
        next_states = dict(states)
        cost_i = int(cost)
        value_i = float(weight) * float(reduction)
        for used, (value, chosen) in states.items():
            total = used + cost_i
            if total > capacity:
                continue
            candidate = (value + value_i, chosen + (index,))
            previous = next_states.get(total)
            if previous is None or candidate[0] > previous[0] + 1e-15:
                next_states[total] = candidate
            elif abs(candidate[0] - previous[0]) <= 1e-15 and candidate[1] < previous[1]:
                next_states[total] = candidate
        states = next_states

    used, (value, chosen) = min(
        states.items(), key=lambda item: (-item[1][0], item[0], item[1][1])
    )
    return HorizonAllocation(
        high_round_indices=tuple(chosen),
        weighted_error_reduction=float(value),
        used_budget_units=int(used),
    )


def trajectory_optimal_high_rounds(
    num_rounds: int,
    high_round_count: int,
    contraction_q: float,
) -> Tuple[int, ...]:
    """Solve the equal-increment finite-horizon allocation exactly."""

    horizon = int(num_rounds)
    count = int(high_round_count)
    if not 0 <= count <= horizon:
        raise ValueError("high_round_count must be in [0, num_rounds]")
    solution = solve_binary_horizon_knapsack(
        weights=trajectory_weights(horizon, contraction_q),
        extra_cost_units=[1] * horizon,
        normalized_error_reduction=[1.0] * horizon,
        budget_units=count,
    )
    return solution.high_round_indices


def resolve_round_policy(
    *,
    schedule: str,
    round_number: int,
    num_rounds: int,
    high_round_count: int,
    high_budget_ratio: float,
    high_gain_target: float,
    low_budget_ratio: float,
    contraction_q: float,
) -> FedSonataRoundPolicy:
    """Resolve the effective high/low action for one round."""

    name = str(schedule).strip().lower()
    horizon = int(num_rounds)
    current = int(round_number)
    if not 1 <= current <= horizon:
        raise ValueError("round_number must be in [1, num_rounds]")
    if not 0.0 < float(high_budget_ratio) <= 1.0:
        raise ValueError("high_budget_ratio must be in (0, 1]")
    if not 0.0 < float(low_budget_ratio) <= 1.0:
        raise ValueError("low_budget_ratio must be in (0, 1]")
    if float(low_budget_ratio) > float(high_budget_ratio):
        raise ValueError("low_budget_ratio cannot exceed high_budget_ratio")
    if not 0.0 < float(high_gain_target) <= 1.0:
        raise ValueError("high_gain_target must be in (0, 1]")
    weights = trajectory_weights(horizon, contraction_q)

    if name == "fixed":
        high = True
    elif name == "trajectory_optimal":
        high = current in set(
            trajectory_optimal_high_rounds(horizon, high_round_count, contraction_q)
        )
    elif name == "backload_ablation":
        count = int(high_round_count)
        if not 0 <= count <= horizon:
            raise ValueError("high_round_count must be in [0, num_rounds]")
        high = current > horizon - count
    else:
        raise ValueError(f"unknown FedSONATA schedule: {schedule}")

    return FedSonataRoundPolicy(
        schedule=name,
        budget_ratio=(float(high_budget_ratio) if high else float(low_budget_ratio)),
        gain_target_ratio=(float(high_gain_target) if high else None),
        high_fidelity=bool(high),
        trajectory_weight=float(weights[current - 1]),
    )
