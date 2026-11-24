"""Convex fixed-deadline bandwidth/power allocation for FedSONATA.

For a declared deadline D, client k is assigned a constant bandwidth b_k and
the minimum power that makes its upload finish at D.  Eliminating power from
the Shannon equality gives a strictly convex RF-energy function of bandwidth.
The unique global allocation is obtained from its KKT conditions by nested
bisection.  Deadline selection is exact only over the explicitly evaluated
grid; no continuous or end-to-end training optimum is claimed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping

import numpy as np

from .profiles import PowerProfile
from .resource_solver import ResourcePlan, _min_bandwidth_for_power
from .system_model import shannon_rate


@dataclass(frozen=True)
class FixedDeadlineSolution:
    bandwidth_hz: Dict[int, float]
    power_w: Dict[int, float]
    upload_time_s: Dict[int, float]
    tx_energy_j: float
    feasible: bool


def _required_power(
    payload_bits: float,
    bandwidth_hz: float,
    channel_gain: float,
    n0_w_hz: float,
    upload_time_s: float,
) -> float:
    if payload_bits <= 0.0:
        return 0.0
    if bandwidth_hz <= 0.0 or channel_gain <= 0.0 or n0_w_hz <= 0.0:
        return float("inf")
    if upload_time_s <= 0.0:
        return float("inf")
    exponent = math.log(2.0) * payload_bits / (bandwidth_hz * upload_time_s)
    if exponent > 700.0:
        return float("inf")
    return (
        n0_w_hz
        * bandwidth_hz
        / channel_gain
        * math.expm1(exponent)
    )


def _marginal_energy_reduction(
    payload_bits: float,
    bandwidth_hz: float,
    channel_gain: float,
    n0_w_hz: float,
    upload_time_s: float,
    pa_efficiency: float,
) -> float:
    """Return -dE_rf/db for E_rf at the Shannon equality."""
    if payload_bits <= 0.0:
        return 0.0
    if bandwidth_hz <= 0.0:
        return float("inf")
    x = math.log(2.0) * payload_bits / (bandwidth_hz * upload_time_s)
    if x > 700.0:
        return float("inf")
    scale = upload_time_s * n0_w_hz / (pa_efficiency * channel_gain)
    return scale * (1.0 + math.exp(x) * (x - 1.0))


def _transmit_energy(
    payload_bits: float,
    bandwidth_hz: float,
    channel_gain: float,
    n0_w_hz: float,
    power: PowerProfile,
    transmit_time_s: float,
) -> float:
    """Energy for sending ``payload_bits`` over ``transmit_time_s``."""
    if payload_bits <= 0.0:
        return 0.0
    p_w = _required_power(
        payload_bits, bandwidth_hz, channel_gain, n0_w_hz, transmit_time_s
    )
    if not math.isfinite(p_w):
        return float("inf")
    p_w = min(power.p_max_w, max(0.0, p_w))
    return (p_w / power.pa_efficiency + power.circuit_power_w) * transmit_time_s


def _energy_optimal_transmit_time(
    payload_bits: float,
    bandwidth_hz: float,
    channel_gain: float,
    n0_w_hz: float,
    power: PowerProfile,
    t_min_s: float,
    t_max_s: float,
    steps: int = 80,
) -> float:
    """Energy-optimal upload duration in ``[t_min, t_max]``.

    The RF energy decreases with a longer window while the circuit energy grows
    linearly, so the total energy is unimodal in the transmit time.  A ternary
    search returns the interior minimiser (the *natural* transmit time), which
    may be strictly shorter than the deadline window -- this is what makes the
    deadline an upper bound instead of an equality.
    """
    if t_max_s <= t_min_s * (1.0 + 1e-12):
        return t_max_s
    lo, hi = float(t_min_s), float(t_max_s)
    for _ in range(steps):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        e1 = _transmit_energy(payload_bits, bandwidth_hz, channel_gain, n0_w_hz, power, m1)
        e2 = _transmit_energy(payload_bits, bandwidth_hz, channel_gain, n0_w_hz, power, m2)
        if e1 <= e2:
            hi = m2
        else:
            lo = m1
    return 0.5 * (lo + hi)


def _minimum_bandwidths(
    clients: List[int],
    payload_bits: Mapping[int, float],
    compute_time_s: Mapping[int, float],
    channel_gain: Mapping[int, float],
    total_bandwidth_hz: float,
    power: PowerProfile,
    n0_w_hz: float,
    deadline_s: float,
) -> tuple[Dict[int, float], Dict[int, float], bool]:
    minimum: Dict[int, float] = {}
    available: Dict[int, float] = {}
    for client_id in clients:
        upload_time = float(deadline_s) - float(compute_time_s[client_id])
        available[client_id] = upload_time
        if upload_time <= 0.0:
            return minimum, available, False
        bits = float(payload_bits[client_id])
        if bits <= 0.0:
            minimum[client_id] = 0.0
            continue
        required = _min_bandwidth_for_power(
            bits,
            power.p_max_w,
            float(channel_gain[client_id]),
            float(n0_w_hz),
            upload_time,
        )
        if not math.isfinite(required) or required > total_bandwidth_hz:
            return minimum, available, False
        minimum[client_id] = max(0.0, float(required))
    feasible = sum(minimum.values()) <= total_bandwidth_hz * (1.0 + 1e-10)
    return minimum, available, feasible


def solve_convex_bandwidth_power(
    clients: List[int],
    payload_bits: Mapping[int, float],
    compute_time_s: Mapping[int, float],
    channel_gain: Mapping[int, float],
    total_bandwidth_hz: float,
    power: PowerProfile,
    n0_w_hz: float,
    deadline_s: float,
    max_bisection_steps: int = 80,
    transmit_mode: str = "fill",
) -> FixedDeadlineSolution:
    """Return the unique minimum-energy allocation for a fixed deadline.

    Each positive-payload upload is constrained to finish at the deadline.
    With tau_k = D - T_cmp,k and c_k = ln(2) S_k / tau_k, its RF energy is

        E_k(b) = tau_k N0 b / (eta h_k) * (exp(c_k / b) - 1).

    E_k''(b) = tau_k N0 exp(c_k/b) c_k^2 /
               (eta h_k b^3) > 0,

    so the bandwidth problem with b_k >= b_k,min and sum b_k <= B is strictly
    convex.  The implementation solves the KKT multiplier by bisection.
    """
    clients = [int(client_id) for client_id in clients]
    if total_bandwidth_hz <= 0.0:
        raise ValueError("total_bandwidth_hz must be positive")
    if n0_w_hz <= 0.0:
        raise ValueError("n0_w_hz must be positive")
    if deadline_s <= 0.0:
        raise ValueError("deadline_s must be positive")
    if not clients:
        return FixedDeadlineSolution({}, {}, {}, 0.0, True)

    minimum, available, feasible = _minimum_bandwidths(
        clients,
        payload_bits,
        compute_time_s,
        channel_gain,
        total_bandwidth_hz,
        power,
        n0_w_hz,
        deadline_s,
    )
    if not feasible:
        return FixedDeadlineSolution({}, {}, {}, float("inf"), False)

    positive = [client_id for client_id in clients if payload_bits[client_id] > 0.0]
    if not positive:
        zeros = {client_id: 0.0 for client_id in clients}
        return FixedDeadlineSolution(zeros, zeros, zeros, 0.0, True)

    minimum_sum = sum(minimum.values())
    if minimum_sum > total_bandwidth_hz:
        return FixedDeadlineSolution({}, {}, {}, float("inf"), False)

    maximum = {
        client_id: total_bandwidth_hz
        - sum(minimum[other] for other in clients if other != client_id)
        for client_id in clients
    }

    def marginal(client_id: int, bandwidth: float) -> float:
        return _marginal_energy_reduction(
            float(payload_bits[client_id]),
            bandwidth,
            float(channel_gain[client_id]),
            float(n0_w_hz),
            available[client_id],
            power.pa_efficiency,
        )

    def allocation_for_multiplier(multiplier: float) -> Dict[int, float]:
        allocation = {client_id: minimum[client_id] for client_id in clients}
        for client_id in positive:
            low = max(minimum[client_id], 1e-12)
            high = max(low, maximum[client_id])
            if marginal(client_id, low) <= multiplier:
                allocation[client_id] = low
                continue
            if marginal(client_id, high) >= multiplier:
                allocation[client_id] = high
                continue
            for _ in range(max_bisection_steps):
                middle = 0.5 * (low + high)
                if marginal(client_id, middle) > multiplier:
                    low = middle
                else:
                    high = middle
            allocation[client_id] = 0.5 * (low + high)
        return allocation

    high_multiplier = max(
        marginal(client_id, max(minimum[client_id], 1e-12))
        for client_id in positive
    )
    low_multiplier = 0.0
    allocation = dict(minimum)
    for _ in range(max_bisection_steps):
        multiplier = 0.5 * (low_multiplier + high_multiplier)
        candidate = allocation_for_multiplier(multiplier)
        if sum(candidate.values()) > total_bandwidth_hz:
            low_multiplier = multiplier
        else:
            high_multiplier = multiplier
            allocation = candidate

    residual = total_bandwidth_hz - sum(allocation.values())
    if residual > 1e-6:
        recipient = max(positive, key=lambda client_id: marginal(client_id, allocation[client_id]))
        allocation[recipient] += residual
    elif residual < -1e-5:
        return FixedDeadlineSolution({}, {}, {}, float("inf"), False)

    powers: Dict[int, float] = {}
    upload_times: Dict[int, float] = {}
    total_energy = 0.0
    natural = str(transmit_mode) == "natural"
    for client_id in clients:
        bits = float(payload_bits[client_id])
        bandwidth = float(allocation[client_id])
        if bits <= 0.0:
            powers[client_id] = 0.0
            upload_times[client_id] = 0.0
            continue
        gain = float(channel_gain[client_id])
        if natural:
            # Deadline as an upper bound: pick the energy-optimal (natural)
            # transmit duration, which may be strictly shorter than the window.
            rate_max = shannon_rate(bandwidth, power.p_max_w, gain, n0_w_hz)
            if rate_max <= 0.0:
                return FixedDeadlineSolution({}, {}, {}, float("inf"), False)
            min_t = bits / rate_max
            if min_t > available[client_id] * (1.0 + 1e-7):
                # Cannot deliver within the deadline even at max power.
                return FixedDeadlineSolution({}, {}, {}, float("inf"), False)
            transmit_time = _energy_optimal_transmit_time(
                bits, bandwidth, gain, n0_w_hz, power, min_t, available[client_id]
            )
        else:
            transmit_time = available[client_id]
        p_w = _required_power(bits, bandwidth, gain, n0_w_hz, transmit_time)
        if not math.isfinite(p_w) or p_w > power.p_max_w * (1.0 + 1e-7):
            return FixedDeadlineSolution({}, {}, {}, float("inf"), False)
        p_w = min(power.p_max_w, max(0.0, p_w))
        rate = shannon_rate(bandwidth, p_w, gain, n0_w_hz)
        upload_time = bits / rate if rate > 0.0 else float("inf")
        if not math.isfinite(upload_time) or upload_time > available[client_id] * (1.0 + 1e-7):
            return FixedDeadlineSolution({}, {}, {}, float("inf"), False)
        powers[client_id] = p_w
        upload_times[client_id] = upload_time
        total_energy += (
            p_w / power.pa_efficiency + power.circuit_power_w
        ) * upload_time

    return FixedDeadlineSolution(
        bandwidth_hz=allocation,
        power_w=powers,
        upload_time_s=upload_times,
        tx_energy_j=total_energy,
        feasible=True,
    )


def solve_convex_round(
    clients: List[int],
    payload_bits: Mapping[int, float],
    compute_time_s: Mapping[int, float],
    channel_gain: Mapping[int, float],
    total_bandwidth_hz: float,
    power: PowerProfile,
    n0_w_hz: float,
    d_min: float,
    d_max: float,
    lambda_e: float = 0.0,
    lambda_u: float = 1.0,
    omega_net: float = 1.0,
    prev_staleness: Mapping[int, float] | None = None,
    est_success: Mapping[int, float] | None = None,
    duration_hazard_rate: float = 0.0,
    deadline_grid_size: int = 24,
    transmit_mode: str = "fill",
    allow_partial_arrival: bool = False,
) -> ResourcePlan:
    """Choose the exact best fixed-deadline solution on a declared grid."""
    if d_min <= 0.0 or d_max < d_min:
        raise ValueError("invalid deadline interval")
    if lambda_e < 0.0 or lambda_u < 0.0:
        raise ValueError("objective weights must be non-negative")
    if duration_hazard_rate < 0.0:
        raise ValueError("duration_hazard_rate must be non-negative")
    if deadline_grid_size < 2:
        raise ValueError("deadline_grid_size must be at least two")
    if not clients:
        return ResourcePlan({}, {}, d_min, True, 0.0, 0.0)

    start = time.perf_counter()
    stale = prev_staleness or {client_id: 0.0 for client_id in clients}
    reliability = est_success or {client_id: 1.0 for client_id in clients}

    at_max = solve_convex_bandwidth_power(
        clients,
        payload_bits,
        compute_time_s,
        channel_gain,
        total_bandwidth_hz,
        power,
        n0_w_hz,
        d_max,
        transmit_mode=transmit_mode,
    )
    if not at_max.feasible and not allow_partial_arrival:
        elapsed_ms = 1000.0 * (time.perf_counter() - start)
        return ResourcePlan({}, {}, d_max, False, elapsed_ms, float("inf"))

    lower = max(float(d_min), max(float(compute_time_s[k]) for k in clients) + 1e-9)
    _, _, lower_feasible = _minimum_bandwidths(
        clients,
        payload_bits,
        compute_time_s,
        channel_gain,
        total_bandwidth_hz,
        power,
        n0_w_hz,
        lower,
    )
    if not lower_feasible:
        low, high = lower, float(d_max)
        for _ in range(70):
            middle = 0.5 * (low + high)
            _, _, middle_feasible = _minimum_bandwidths(
                clients,
                payload_bits,
                compute_time_s,
                channel_gain,
                total_bandwidth_hz,
                power,
                n0_w_hz,
                middle,
            )
            if middle_feasible:
                high = middle
            else:
                low = middle
        lower = high

    if allow_partial_arrival:
        # Deadline as an upper bound *with* partial arrival: candidates span the
        # whole interval and clients that cannot meet a deadline simply miss it.
        # This exposes the horizon/arrival trade-off, so the optimum responds to
        # the utility weights instead of collapsing to the all-arrive feasibility
        # bound (the T5 degeneracy).
        def _min_bandwidth_for(client_id: int, deadline: float) -> float:
            available = deadline - float(compute_time_s[client_id])
            if available <= 0.0 or float(payload_bits[client_id]) <= 0.0:
                return float("inf")
            return _min_bandwidth_for_power(
                float(payload_bits[client_id]),
                power.p_max_w,
                float(channel_gain[client_id]),
                float(n0_w_hz),
                available,
            )

        candidates = sorted(set(
            float(value)
            for value in np.linspace(float(d_min), float(d_max), deadline_grid_size)
        ))
        best_deadline = float(d_max)
        best_solution = at_max
        best_objective = float("inf")
        for deadline in candidates:
            served = list(clients)
            while True:
                positive = [c for c in served if float(payload_bits[c]) > 0.0]
                needed = sum(_min_bandwidth_for(c, deadline) for c in positive)
                if not positive or needed <= total_bandwidth_hz * (1.0 + 1e-10):
                    break
                hardest = max(positive, key=lambda c: _min_bandwidth_for(c, deadline))
                served.remove(hardest)
            if not served:
                objective = deadline
                solution = None
            else:
                solution = solve_convex_bandwidth_power(
                    served,
                    payload_bits,
                    compute_time_s,
                    channel_gain,
                    total_bandwidth_hz,
                    power,
                    n0_w_hz,
                    deadline,
                    transmit_mode=transmit_mode,
                )
                if not solution.feasible:
                    continue
                utility = 0.0
                for client_id in served:
                    upload_time = solution.upload_time_s[client_id]
                    if not math.isfinite(upload_time):
                        continue
                    survival = math.exp(-duration_hazard_rate * upload_time)
                    utility += (
                        float(reliability.get(client_id, 1.0))
                        * survival
                        / (1.0 + omega_net * float(stale.get(client_id, 0.0)))
                    )
                objective = (
                    deadline + lambda_e * solution.tx_energy_j - lambda_u * utility
                )
            if objective < best_objective - 1e-12:
                best_objective = objective
                best_deadline = deadline
                if solution is not None:
                    best_solution = solution
        elapsed_ms = 1000.0 * (time.perf_counter() - start)
        return ResourcePlan(
            bandwidth_hz={c: float(best_solution.bandwidth_hz.get(c, 0.0)) for c in clients},
            power_w={c: float(best_solution.power_w.get(c, 0.0)) for c in clients},
            deadline_s=best_deadline,
            feasible=math.isfinite(best_objective),
            solver_ms=elapsed_ms,
            objective=best_objective,
        )

    candidates = sorted(set(
        float(value) for value in np.linspace(lower, float(d_max), deadline_grid_size)
    ))
    best_deadline = float(d_max)
    best_solution = at_max
    best_objective = float("inf")
    for deadline in candidates:
        solution = solve_convex_bandwidth_power(
            clients,
            payload_bits,
            compute_time_s,
            channel_gain,
            total_bandwidth_hz,
            power,
            n0_w_hz,
            deadline,
            transmit_mode=transmit_mode,
        )
        if not solution.feasible:
            continue
        utility = 0.0
        for client_id in clients:
            survival = math.exp(
                -duration_hazard_rate * solution.upload_time_s[client_id]
            )
            utility += (
                float(reliability.get(client_id, 1.0))
                * survival
                / (1.0 + omega_net * float(stale.get(client_id, 0.0)))
            )
        objective = (
            deadline
            + lambda_e * solution.tx_energy_j
            - lambda_u * utility
        )
        if objective < best_objective - 1e-12:
            best_objective = objective
            best_deadline = deadline
            best_solution = solution

    elapsed_ms = 1000.0 * (time.perf_counter() - start)
    return ResourcePlan(
        bandwidth_hz=best_solution.bandwidth_hz,
        power_w=best_solution.power_w,
        deadline_s=best_deadline,
        feasible=math.isfinite(best_objective),
        solver_ms=elapsed_ms,
        objective=best_objective,
    )
