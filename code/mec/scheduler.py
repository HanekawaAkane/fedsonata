"""Reliability-aware block scheduling (SYSTEM_MODEL sec.6 & 8A).

Given per-client-block utility and staleness plus reliability estimates, the
scheduler picks a budget C_k of blocks per client with a deterministic top-C
greedy rule (interpretable, reproducible). The score folds in:

    score = u - mu_c * c - mu_s * s + mu_f * log(1 + tilde_tau)

where u is the learning-utility proxy, c the per-block compute cost, s the
per-block payload, and tilde_tau the effective staleness. The reliability
term multiplies utility inside u (see selection_utility), so weak clients are
deprioritised unless their blocks are very stale (coverage).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Set

import numpy as np

from .staleness import EffectiveStaleness


@dataclass
class StalenessState:
    """Lightweight holder used by the scheduler for utility computation.

    Wraps an EffectiveStaleness so schedulers can query effective tau without
    importing the staleness internals directly.
    """

    effective: EffectiveStaleness

    def tau(self, client_id: int, block: str) -> float:
        return self.effective.effective(client_id, block)


def selection_utility(
    est_success: float,
    sensitivity_sq: float,
    tau_eff: float,
    gamma: float,
    lambda_cov: float,
) -> float:
    """Learning-utility proxy (SYSTEM_MODEL sec.6):

        u = rhat * D^2 / (1 + tilde_tau)^gamma
            + lambda_cov * log(1 + tilde_tau)

    This is a *local descent proxy*, NOT the true joint A/B Adam loss decrease
    (SYSTEM_MODEL sec.6 explicit caveat). It can legitimately be ablation-
    removed (proposed-noReliability sets est_success contribution to constant).
    """
    if tau_eff < 0:
        raise ValueError("effective staleness must be non-negative")
    base = est_success * (sensitivity_sq ** 2) / ((1.0 + tau_eff) ** gamma)
    cov = lambda_cov * math.log1p(tau_eff)
    return base + cov


def block_score(
    utility: float,
    compute_cost: float,
    payload_cost: float,
    tau_eff: float,
    mu_c: float,
    mu_s: float,
    mu_f: float,
) -> float:
    """Net block score (SYSTEM_MODEL sec.8A)."""
    return (
        utility
        - mu_c * compute_cost
        - mu_s * payload_cost
        + mu_f * math.log1p(tau_eff)
    )


@dataclass
class SchedulerConfig:
    mu_c: float = 1.0
    mu_s: float = 1.0
    mu_f: float = 0.1
    gamma: float = 1.0
    lambda_cov: float = 0.05


def reliability_aware_selection(
    client_id: int,
    all_blocks: List[str],
    budget: int,
    state: StalenessState,
    est_success: Dict[int, float],
    sensitivity: Dict[str, float],
    compute_cost: Dict[str, float],
    payload_cost: Dict[str, float],
    cfg: SchedulerConfig,
    reliability_weight: float = 1.0,
) -> Set[str]:
    """Deterministic top-C greedy block selection for one client.

    reliability_weight == 0 reproduces proposed-noReliability (utility no longer
    scales with success probability). The greedy is reproducible and scores
    are computed independently of training. No submodularity claim is made.
    """
    if budget <= 0:
        return set()
    if budget >= len(all_blocks):
        return set(all_blocks)

    rhat = float(est_success.get(client_id, 1.0)) ** (reliability_weight)
    scores = []
    for block in all_blocks:
        tau = state.tau(client_id, block)
        sens = sensitivity.get(block, 0.0)
        util = selection_utility(
            rhat, sens, tau, cfg.gamma, cfg.lambda_cov
        )
        sc = block_score(
            util,
            compute_cost.get(block, 0.0),
            payload_cost.get(block, 0.0),
            tau,
            cfg.mu_c,
            cfg.mu_s,
            cfg.mu_f,
        )
        scores.append((sc, block))
    scores.sort(key=lambda x: x[0], reverse=True)
    return {block for _, block in scores[:budget]}


def est_success_default(clients: List[int], value: float = 1.0) -> Dict[int, float]:
    return {k: value for k in clients}


def update_success_ewma(
    est_success: Dict[int, float],
    client_id: int,
    arrived: bool,
    beta: float = 0.8,
) -> float:
    """Update and return a client's end-to-end delivery estimate."""
    if not 0.0 <= float(beta) < 1.0:
        raise ValueError("beta must be in [0, 1)")
    old = min(1.0, max(0.0, float(est_success.get(client_id, 1.0))))
    observation = 1.0 if bool(arrived) else 0.0
    updated = float(beta) * old + (1.0 - float(beta)) * observation
    est_success[int(client_id)] = min(1.0, max(0.0, updated))
    return est_success[int(client_id)]
