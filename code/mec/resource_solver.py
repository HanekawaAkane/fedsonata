"""Resource solver: bandwidth / power / deadline allocation.

Implements SYSTEM_MODEL sec.8 B & C. Given a fixed block allocation and the
modeled compute time, the solver finds per-client (b_k, p_k) that satisfy the
rate superlevel constraint

    b_k * log2(1 + p_k h_k / (N0 b_k)) >= S_k / (D - T_cmp_k)

with sum(b_k) <= B and 0 <= p_k <= p_max, then picks an adaptive deadline.

Explicit infeasible handling: if even max power + full bandwidth cannot meet
the deadline for any client, return feasible=False. NEVER return NaN or
silently exceed p_max. (RESULT_AUDIT sec.4 "deadline infeasible").
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .profiles import PowerProfile, ChannelProfile, dbm_to_watt
from .system_model import uplink_payload_bits, shannon_rate


@dataclass
class ResourcePlan:
    """Per-round resource allocation returned by the solver."""

    bandwidth_hz: Dict[int, float]
    power_w: Dict[int, float]
    deadline_s: float
    feasible: bool
    solver_ms: float
    objective: float = float("nan")  # D + lambda_E*E - lambda_U*U (lower better)


def _min_power_for_bandwidth(
    s_up_bits: float,
    b_hz: float,
    h: float,
    n0_w_hz: float,
    t_avail: float,
    p_max_w: float,
) -> float:
    """Minimum transmit power (W) to deliver s_up within t_avail over b_hz.

    Closed form from R = b*log2(1+ph/(N0 b)) >= s_up/t_avail:

        p = (N0 b / h) * (2^(s_up/(b t_avail)) - 1)

    Raises ValueError if t_avail <= 0 (no time -> infeasible upstream).
    """
    if t_avail <= 0:
        raise ValueError("no available upload time")
    if b_hz <= 0:
        raise ValueError("bandwidth must be positive")
    required_rate = s_up_bits / t_avail
    # invert shannon rate analytically
    exponent = required_rate / b_hz
    if exponent > 1022:
        # 2^x overflow guard -> infeasible under this bandwidth
        return float("inf")
    p = (n0_w_hz * b_hz / h) * (2.0 ** exponent - 1.0)
    # NOTE: do NOT clamp to p_max here. Returning the true required power lets
    # the caller detect p > p_max as infeasible (explicit, never silent).
    return p


def _min_bandwidth_for_power(
    s_up_bits: float,
    p_w: float,
    h: float,
    n0_w_hz: float,
    t_avail: float,
) -> float:
    """Minimum bandwidth (Hz) to deliver s_up within t_avail at fixed power.

    Monotone in b (rate increases with b), so bisection works. Returns inf if
    even b -> +inf cannot meet the rate (i.e. required_rate > p*h/(N0)).
    """
    required_rate = s_up_bits / t_avail

    def rate_at(b: float) -> float:
        return shannon_rate(b, p_w, h, n0_w_hz)

    # Asymptotic rate as b->inf equals p*h/N0 (the high-SNR ceiling).
    ceiling = p_w * h / n0_w_hz
    if required_rate >= ceiling:
        return float("inf")

    lo, hi = 1e-3, 1e12
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if rate_at(mid) < required_rate:
            lo = mid
        else:
            hi = mid
    return hi


def solve_bandwidth_power(
    clients: List[int],
    s_up_bits: Dict[int, float],
    t_cmp: Dict[int, float],
    h: Dict[int, float],
    total_bw_hz: float,
    p_max_w: float,
    n0_w_hz: float,
    deadline_s: float,
    eps: float = 1e-3,
    max_iter: int = 30,
) -> tuple[Dict[int, float], Dict[int, float], bool]:
    """Iterative bandwidth<->power allocation for a fixed deadline.

    Strategy (SYSTEM_MODEL sec.8 B):
      1. Start from equal bandwidth share.
      2. For each client compute min power to meet deadline at current b.
      3. If some client needs > p_max, expand its bandwidth via bisection and
         re-normalise the sum to total_bw (proportional shrink of others).
      4. Iterate until feasible or max_iter.

    Returns (bandwidth_hz, power_w, feasible). feasible=False means even full
    bandwidth + p_max cannot satisfy the deadline for some client.
    """
    if deadline_s <= 0:
        raise ValueError("deadline must be positive")
    K = len(clients)
    if K == 0:
        return {}, {}, True

    bw = {k: total_bw_hz / K for k in clients}

    for _ in range(max_iter):
        # Step A: min power at current bandwidth.
        pw = {}
        infeasible_any = False
        for k in clients:
            t_avail = deadline_s - t_cmp[k]
            if t_avail <= 0:
                infeasible_any = True
                pw[k] = p_max_w
                continue
            p = _min_power_for_bandwidth(
                s_up_bits[k], bw[k], h[k], n0_w_hz, t_avail, p_max_w
            )
            pw[k] = p
            if p > p_max_w + 1e-9:
                infeasible_any = True

        if not infeasible_any:
            # Ensure sum(bw) <= total and each within bounds; renormalise.
            return _renormalise_bandwidth(bw, total_bw_hz), pw, True

        # Step B: expand bandwidth for clients that still exceed p_max.
        # Increase those clients' bandwidth until p fits, taking budget from
        # a proportional shrink of all clients, then check sum constraint.
        new_bw = dict(bw)
        extra_needed = 0.0
        for k in clients:
            t_avail = deadline_s - t_cmp[k]
            if t_avail <= 0:
                continue
            p = _min_power_for_bandwidth(
                s_up_bits[k], bw[k], h[k], n0_w_hz, t_avail, p_max_w
            )
            if p > p_max_w + 1e-9:
                # required bandwidth at p_max
                b_req = _min_bandwidth_for_power(
                    s_up_bits[k], p_max_w, h[k], n0_w_hz, t_avail
                )
                if not math.isfinite(b_req):
                    return bw, pw, False
                extra = max(0.0, b_req - bw[k])
                new_bw[k] = b_req
                extra_needed += extra

        if extra_needed <= 1e-9:
            # oscillation guard
            return _renormalise_bandwidth(new_bw, total_bw_hz), pw, True

        # Shrink others to free `extra_needed` while keeping sum <= total.
        others = [k for k in clients if new_bw[k] <= bw[k] + 1e-12]
        freeable = sum(bw[k] - new_bw[k] for k in others)
        if freeable + 1e-9 < extra_needed:
            return bw, pw, False
        scale = (freeable - extra_needed) / freeable if freeable > 0 else 1.0
        for k in others:
            new_bw[k] = new_bw[k] + scale * (bw[k] - new_bw[k])
        bw = new_bw

    # Converged to feasibility or hit max_iter.
    feasible = all(
        deadline_s - t_cmp[k] > 0
        and _min_power_for_bandwidth(
            s_up_bits[k], bw[k], h[k], n0_w_hz, deadline_s - t_cmp[k], p_max_w
        )
        <= p_max_w + 1e-9
        for k in clients
    )
    return _renormalise_bandwidth(bw, total_bw_hz), pw, feasible


def _renormalise_bandwidth(
    bw: Dict[int, float], total_bw_hz: float
) -> Dict[int, float]:
    s = sum(bw.values())
    if s <= 0:
        return {k: 0.0 for k in bw}
    if s > total_bw_hz:
        return {k: v * total_bw_hz / s for k, v in bw.items()}
    return dict(bw)


def solve_deadline(
    clients: List[int],
    s_up_bits: Dict[int, float],
    t_cmp: Dict[int, float],
    h: Dict[int, float],
    total_bw_hz: float,
    p_max_w: float,
    n0_w_hz: float,
    d_min: float,
    d_max: float,
    lambda_e: float,
    omega_net: float = 1.0,
    prev_staleness: Dict[int, float] = None,
    est_success: Dict[int, float] = None,
    n0_grid: int = 24,
    lambda_u: float = 1.0,
    pa_efficiency: float = 0.35,
    circuit_power_w: float = 0.1,
) -> tuple[float, bool]:
    """Adaptive deadline via 1-D search over candidate breakpoints.

    Objective (SYSTEM_MODEL sec.8 C):
        D + lambda_E * E(D) - lambda_U * U(D)
    only needs checking at breakpoints where completion times change plus the
    endpoints, because the per-client min-power (hence energy) is piecewise
    continuous between completion-time crossings. We evaluate a bounded grid
    between [d_min, d_max] plus the per-client t_cmp breakpoints.

    Returns (deadline_s, feasible). feasible=False if even d_max cannot be met
    by any client at full bandwidth + p_max.
    """
    if lambda_e < 0.0 or lambda_u < 0.0:
        raise ValueError("objective weights must be non-negative")
    if not 0.0 < pa_efficiency <= 1.0:
        raise ValueError("pa_efficiency must be in (0, 1]")
    if circuit_power_w < 0.0:
        raise ValueError("circuit_power_w must be non-negative")

    prev_staleness = prev_staleness or {k: 0.0 for k in clients}
    est_success = est_success or {k: 1.0 for k in clients}

    # Feasibility at d_max: each client needs finite bandwidth at p_max.
    for k in clients:
        t_avail = d_max - t_cmp[k]
        if t_avail <= 0:
            return d_max, False
        b_req = _min_bandwidth_for_power(
            s_up_bits[k], p_max_w, h[k], n0_w_hz, t_avail
        )
        if not math.isfinite(b_req):
            return d_max, False

    # Candidate deadlines: endpoints + per-client t_cmp crossings.
    candidates = sorted(set([d_min, d_max]))
    for k in clients:
        tc = t_cmp[k]
        if d_min < tc < d_max:
            candidates.append(tc)

    # HARD FLOOR (D015): any feasible deadline must be >= the time for the
    # worst client to finish compute + its uplink when bandwidth is SHARED
    # (worst case: every client competes for the budget). Using the shared
    # bandwidth (total_bw / n) as the conservative per-client rate guarantees
    # d_floor is never smaller than the true worst-case completion time, so the
    # solver can never pick a deadline that makes EVERY client miss. (The earlier
    # version estimated t_up from the *exclusive* full-bandwidth budget, which
    # understated the floor to ~t_cmp and let a too-tight deadline slip through,
    # still producing intermittent on_time=0 rounds.)
    n_clients = max(1, len(clients))
    shared_bw = total_bw_hz / n_clients
    worst_completion = 0.0
    for k in clients:
        rate_shared = shannon_rate(shared_bw, p_max_w, h[k], n0_w_hz)
        if rate_shared <= 0:
            # Cannot even complete at p_max over shared bw -> fall back to d_max.
            return d_max, False
        t_up_shared = s_up_bits[k] / rate_shared
        worst_completion = max(worst_completion, t_cmp[k] + t_up_shared)
    d_floor = max(d_min, worst_completion)
    if d_floor > d_max + 1e-9:
        # The requested deadline interval contains no point at which every
        # client can finish.  Do not append an out-of-range hard floor and
        # accidentally return a deadline larger than d_max as feasible.
        return d_max, False
    candidates.append(d_floor)

    candidates = sorted(set(candidates))

    # Also add a uniform grid for robustness of the objective landscape.
    grid = np.linspace(d_min, d_max, n0_grid)
    candidates = sorted(set(candidates) | set(float(x) for x in grid))

    best_d = d_max
    best_obj = float("inf")
    found = False
    for d in candidates:
        if d <= 0:
            continue
        if d > d_max + 1e-9:
            continue
        if d < d_floor - 1e-9:
            # (D015) Never consider a deadline below the hard floor: doing so
            # makes EVERY client miss and collapses on_time_rate to 0.
            continue
        bw, pw, ok = solve_bandwidth_power(
            clients, s_up_bits, t_cmp, h,
            total_bw_hz, p_max_w, n0_w_hz, d,
        )
        if not ok:
            continue
        # (D015) Verify every client can ACTUALLY complete within d at the
        # allocated bandwidth/power. solve_bandwidth_power may report feasible
        # while a client's real completion still exceeds d; reject such d.
        _all_arrive = True
        for k in clients:
            rate = shannon_rate(bw[k], pw[k], h[k], n0_w_hz)
            if rate <= 0:
                _all_arrive = False
                break
            t_up = s_up_bits[k] / rate
            if t_cmp[k] + t_up > d + 1e-9:
                _all_arrive = False
                break
        if not _all_arrive:
            continue
        found = True
        e_total = 0.0
        u_total = 0.0
        for k in clients:
            # transmit energy at this allocation
            rate = shannon_rate(bw[k], pw[k], h[k], n0_w_hz)
            if rate <= 0:
                continue
            t_up = s_up_bits[k] / rate
            e_tx = (pw[k] / pa_efficiency + circuit_power_w) * t_up
            e_total += e_tx
            # learning utility proxy (SYSTEM_MODEL sec.6) -- monotone in success
            tau = prev_staleness[k]
            u_k = est_success[k] / ((1.0 + omega_net * tau) ** 1.0)
            u_total += u_k
        obj = d + lambda_e * e_total - lambda_u * u_total
        if obj < best_obj:
            best_obj = obj
            best_d = d

    if not found:
        return d_max, False
    return best_d, True


def solve_round(
    clients: List[int],
    s_up_bits: Dict[int, float],
    t_cmp: Dict[int, float],
    h: Dict[int, float],
    profiles_total_bw_hz: float,
    power: PowerProfile,
    n0_w_hz: float,
    d_min: float,
    d_max: float,
    lambda_e: float = 0.0,
    omega_net: float = 1.0,
    prev_staleness: Dict[int, float] = None,
    est_success: Dict[int, float] = None,
    lambda_u: float = 1.0,
) -> ResourcePlan:
    """End-to-end per-round solve: deadline -> bandwidth/power.

    Wrapper that times the optimisation (proving it is not the bottleneck,
    SYSTEM_MODEL sec.10) and packages the result in a ResourcePlan.
    """
    t0 = time.perf_counter()
    deadline, ok = solve_deadline(
        clients, s_up_bits, t_cmp, h,
        profiles_total_bw_hz, power.p_max_w, n0_w_hz,
        d_min, d_max, lambda_e, omega_net, prev_staleness, est_success,
        lambda_u=lambda_u,
        pa_efficiency=power.pa_efficiency,
        circuit_power_w=power.circuit_power_w,
    )
    bandwidth, power_w, ok2 = solve_bandwidth_power(
        clients, s_up_bits, t_cmp, h,
        profiles_total_bw_hz, power.p_max_w, n0_w_hz, deadline,
    )
    feasible = ok and ok2
    elapsed_ms = 1000.0 * (time.perf_counter() - t0)
    return ResourcePlan(
        bandwidth_hz=bandwidth,
        power_w=power_w,
        deadline_s=deadline,
        feasible=feasible,
        solver_ms=elapsed_ms,
        objective=float("nan"),
    )
