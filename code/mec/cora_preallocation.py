"""CORA predictor step for duration-aware RIFT selection.

RIFT must choose its transmitted LoRA factors before the final radio allocation
is known, while the duration-aware value of a message needs a link rate.  This
module removes the former equal-bandwidth shortcut.  It constructs a proven
upper bound on the selected RIFT payload, solves CORA for that upper-bound
payload, and exposes the resulting per-client rates as deterministic reference
rates for the RIFT multiple-choice knapsack.

After RIFT has selected the real payload, ``main.py`` runs CORA again.  Thus the
predictor and corrector radio subproblems are globally optimal at a fixed
deadline (and exact on the declared deadline grid), while RIFT is exact for its
quantized conditional subproblem.  This is not a claim of global optimality for
the coupled discrete/nonconvex federated-training problem.
"""

from __future__ import annotations

import os as _os

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping

from .optimal_resource_solver import solve_convex_round
from .profiles import PowerProfile
from .resource_solver import ResourcePlan
from .system_model import shannon_rate


REFERENCE_RATE_SOURCE = "cora_rift_budget_upper_bound_preallocation"


@dataclass(frozen=True)
class CoraRiftPreallocation:
    """A feasible CORA predictor and its deterministic RIFT reference rates."""

    plan: ResourcePlan
    payload_upper_bits: Dict[int, float]
    reference_rate_bps: Dict[int, float]
    rate_source: str = REFERENCE_RATE_SOURCE


def rift_payload_upper_bytes(
    full_ab_bytes: int,
    budget_ratio: float,
    reliability: float,
    reliability_floor: float,
    classifier_bytes: int = 0,
) -> int:
    """Return a true-byte upper bound for one RIFT-selected client message.

    RIFT defines ``budget=floor(rho*max(p_floor,p_k)*C_AB)`` and conservatively
    rounds option costs upward.  Its selected factor bytes therefore never
    exceed this value.  A server-global classifier, when used, is not selected
    by RIFT and is added separately.
    """

    if int(full_ab_bytes) < 0 or int(classifier_bytes) < 0:
        raise ValueError("payload byte counts must be non-negative")
    if not 0.0 < float(budget_ratio) <= 1.0:
        raise ValueError("budget_ratio must be in (0, 1]")
    if not 0.0 <= float(reliability) <= 1.0:
        raise ValueError("reliability must be in [0, 1]")
    if not 0.0 < float(reliability_floor) <= 1.0:
        raise ValueError("reliability_floor must be in (0, 1]")
    _rel_scale = (
        1.0 if _os.environ.get("RIFT_BUDGET_RELIABILITY", "1") == "0"
        else max(float(reliability_floor), float(reliability))
    )
    factor_budget = int(math.floor(
        int(full_ab_bytes)
        * float(budget_ratio)
        * _rel_scale
    ))
    return factor_budget + int(classifier_bytes)


def solve_rift_cora_preallocation(
    *,
    clients: List[int],
    full_ab_bytes: Mapping[int, int],
    classifier_bytes: Mapping[int, int],
    compute_time_s: Mapping[int, float],
    channel_gain: Mapping[int, float],
    reliability: Mapping[int, float],
    budget_ratio: float,
    reliability_floor: float,
    metadata_bits: float,
    total_bandwidth_hz: float,
    power: PowerProfile,
    n0_w_hz: float,
    d_min: float,
    d_max: float,
    lambda_e: float,
    lambda_u: float,
    omega_net: float,
    prev_staleness: Mapping[int, float],
    duration_hazard_rate: float,
    deadline_grid_size: int,
) -> CoraRiftPreallocation:
    """Solve the CORA predictor on the RIFT true-byte upper-bound payload."""

    client_ids = [int(client_id) for client_id in clients]
    if float(metadata_bits) < 0.0:
        raise ValueError("metadata_bits must be non-negative")
    payload_upper_bits = {
        client_id: float(metadata_bits) + 8.0 * float(rift_payload_upper_bytes(
            int(full_ab_bytes[client_id]),
            float(budget_ratio),
            float(reliability[client_id]),
            float(reliability_floor),
            int(classifier_bytes.get(client_id, 0)),
        ))
        for client_id in client_ids
    }
    plan = solve_convex_round(
        client_ids,
        payload_upper_bits,
        compute_time_s,
        channel_gain,
        total_bandwidth_hz,
        power,
        n0_w_hz,
        d_min,
        d_max,
        lambda_e=lambda_e,
        lambda_u=lambda_u,
        omega_net=omega_net,
        prev_staleness=prev_staleness,
        est_success=reliability,
        duration_hazard_rate=duration_hazard_rate,
        deadline_grid_size=deadline_grid_size,
    )
    if not plan.feasible:
        raise RuntimeError(
            "CORA RIFT preallocation is infeasible for the declared payload "
            "upper bounds and deadline interval"
        )
    reference_rate_bps = {}
    for client_id in client_ids:
        rate = shannon_rate(
            float(plan.bandwidth_hz[client_id]),
            float(plan.power_w[client_id]),
            float(channel_gain[client_id]),
            float(n0_w_hz),
        )
        if not math.isfinite(rate) or rate <= 0.0:
            raise RuntimeError(
                f"CORA RIFT preallocation produced invalid rate for client {client_id}"
            )
        reference_rate_bps[client_id] = float(rate)
    return CoraRiftPreallocation(
        plan=plan,
        payload_upper_bits=payload_upper_bits,
        reference_rate_bps=reference_rate_bps,
    )
