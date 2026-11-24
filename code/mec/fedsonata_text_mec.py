"""FedSONATA text-MEC controller: RIFT-LEF selection plus CORA radio solve.

This module ports the vision-loop FedSONATA-CORA closed loop
(docs/FEDSONATA_METHOD.md) to the embedded RoBERTa/GLUE entry:

  1. **CORA predictor** (``pre_round``): before local training, solve the
     convex bandwidth/power/deadline problem on the RIFT true-byte upper
     bound and expose per-client reference rates for the duration-aware
     RIFT value.
  2. **RIFT-LEF selection** (``select_factors``): after local training,
     fold ACK-pending residuals into the deltas, then solve the quantized
     product-space multiple-choice knapsack per client.
  3. **CORA corrector + arrival gate** (``run_round``): re-solve the convex
     radio problem on the *actual* selected payload, apply the duration
     hazard and deadline gate, ACK delivered factors, and update effective
     staleness / reliability EWMA.

factor_policy:
  - "fedsonata_cora": full method (RIFT-LEF + CORA predictor/corrector);
  - "cora":           CORA radio only, factors transmitted as trained
                      (Fixed-A / Fixed-AB endpoints set by matrix_mode);
  - "equal_fixed":    identical radio to FedHelloMecController (regression
                      anchor; normally the D051 controller is used instead).

Traces depend only on (seed, num_clients, channel, dropout, hazard), so a
D052 run shares the trace realisation of the D051 equal-fixed baselines at
the same seed: paired comparisons are trace-matched.

No claim of global optimality of federated training: exactness holds only
for the quantized RIFT subproblem, the fixed-deadline convex radio problem,
and the declared deadline grid.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import torch

from .cora_preallocation import (
    REFERENCE_RATE_SOURCE,
    solve_rift_cora_preallocation,
)
from .error_feedback import FactorResidualState
from .factor_selector import RiftSelection, select_rift_updates
from .metrics import MecMetrics
from .optimal_resource_solver import solve_convex_round
from .profiles import ProfileSet, build_default_profiles
from .staleness import EffectiveStaleness, update_effective_staleness
from .system_model import compute_cycles, compute_time_energy, shannon_rate
from .traces import TraceGenerator, compute_trace_hash


@dataclass(frozen=True)
class FedSonataTextMecConfig:
    num_clients: int
    num_rounds: int
    seed: int
    factor_policy: str = "fedsonata_cora"  # fedsonata_cora | cora | equal_fixed
    heterogeneity_type: str = "6-3-1"
    allocation_ratio: float = 0.5
    channel_model: str = "correlated"
    channel_correlation: float = 0.9
    channel_gain_mean: float = 1.0e-10
    dropout_model: str = "gilbert_elliott"
    dropout_rate: float = 0.2
    ge_good_to_bad: float = 0.1
    duration_hazard_rate: float = 0.1
    total_bandwidth_mhz: float = 5.0
    power_max_dbm: float = 23.0
    # CORA deadline grid (vision D050 main-table setting)
    d_min_s: float = 0.5
    d_max_s: float = 10.0
    deadline_grid_size: int = 24
    lambda_e: float = 0.1
    lambda_u: float = 1.0
    omega_net: float = 1.0
    reliability_beta: float = 0.8
    # RIFT-LEF (vision D050 main-table setting)
    rift_budget_ratio: float = 0.35
    rift_ef_decay: float = 0.5
    rift_reliability_floor: float = 0.25
    rift_quantum_bytes: int = 4096
    rift_staleness_weight: float = 0.1


@dataclass
class FedSonataTextMecRound:
    client_results: List[dict]
    arrived_client_ids: List[int]
    diagnostics: Dict[str, object]


def _update_success_ewma(
    est_success: Dict[int, float], client_id: int, arrived: bool, beta: float
) -> float:
    if not 0.0 <= float(beta) < 1.0:
        raise ValueError("beta must be in [0, 1)")
    old = min(1.0, max(0.0, float(est_success.get(client_id, 1.0))))
    observation = 1.0 if bool(arrived) else 0.0
    updated = float(beta) * old + (1.0 - float(beta)) * observation
    est_success[int(client_id)] = min(1.0, max(0.0, updated))
    return est_success[int(client_id)]


class FedSonataTextMecController:
    """Deterministic RIFT-LEF + CORA controller for the text entry."""

    def __init__(self, config: FedSonataTextMecConfig):
        self.config = config
        policy = str(config.factor_policy)
        if policy not in {"fedsonata_cora", "cora", "equal_fixed"}:
            raise ValueError(f"unknown factor policy: {policy}")
        if int(config.num_clients) < 1 or int(config.num_rounds) < 1:
            raise ValueError("num_clients and num_rounds must be positive")
        if not 0.0 <= float(config.channel_correlation) < 1.0:
            raise ValueError("channel_correlation must be in [0,1)")
        if float(config.channel_gain_mean) <= 0.0:
            raise ValueError("channel_gain_mean must be positive")
        if float(config.d_min_s) <= 0.0 or float(config.d_max_s) < float(config.d_min_s):
            raise ValueError("invalid deadline interval")
        if not 0.0 < float(config.rift_budget_ratio) <= 1.0:
            raise ValueError("rift_budget_ratio must be in (0, 1]")
        if not 0.0 < float(config.rift_ef_decay) <= 1.0:
            raise ValueError("rift_ef_decay must be in (0, 1]")

        self.factor_policy = policy
        self.rift_enabled = policy == "fedsonata_cora"

        self.profiles: ProfileSet = build_default_profiles(
            num_clients=int(config.num_clients),
            heterogeneity_type=str(config.heterogeneity_type),
            base_ratio=float(config.allocation_ratio),
            total_bandwidth_mhz=float(config.total_bandwidth_mhz),
            power_max_dbm=float(config.power_max_dbm),
            channel_model=str(config.channel_model),
            dropout_model=str(config.dropout_model),
            dropout_rate=float(config.dropout_rate),
        )
        self.profiles.channel.correlation = float(config.channel_correlation)
        self.profiles.channel.rayleigh_mean = float(config.channel_gain_mean)
        if str(config.dropout_model) == "gilbert_elliott":
            p = float(config.ge_good_to_bad)
            if not 0.0 <= p <= 1.0:
                raise ValueError("ge_good_to_bad must be in [0,1]")
            if float(config.dropout_rate) <= 0.0:
                p, q = 0.0, 1.0
            else:
                q = p * (1.0 - float(config.dropout_rate)) / float(config.dropout_rate)
            if not 0.0 < q <= 1.0:
                raise ValueError("Gilbert-Elliott parameters are infeasible")
            self.profiles.dropout.ge_good_to_bad = p
            self.profiles.dropout.ge_bad_to_good = q

        self.trace = TraceGenerator(
            int(config.seed),
            int(config.num_clients),
            self.profiles.channel,
            self.profiles.dropout,
        )
        # Same trace inputs as FedHelloMecController => identical hash => the
        # D052 rows are trace-matched with the D051 equal-fixed baselines.
        self.trace_hash = compute_trace_hash(
            seed=int(config.seed),
            num_clients=int(config.num_clients),
            channel=self.profiles.channel,
            dropout=self.profiles.dropout,
            rounds=int(config.num_rounds),
            duration_hazard_rate=float(config.duration_hazard_rate),
        )
        self.metrics = MecMetrics()
        self.staleness = EffectiveStaleness(omega_net=float(config.omega_net))
        self.est_success: Dict[int, float] = {
            cid: 1.0 for cid in range(int(config.num_clients))
        }
        self.rift_ef_state = (
            FactorResidualState(decay=float(config.rift_ef_decay))
            if self.rift_enabled else None
        )
        # Per-round buffers filled by select_factors, consumed by run_round.
        self._round_corrected: Dict[int, Tuple[dict, dict]] = {}
        self._round_choices: Dict[int, Dict[str, str]] = {}
        self._round_allocated: Dict[int, Set[str]] = {}
        self._reference_rates: Dict[int, float] = {}
        self._pre_round_degraded: bool = False

    # ------------------------------------------------------------------
    # 1. CORA predictor (before local training; RIFT mode only)
    # ------------------------------------------------------------------
    def pre_round(
        self,
        *,
        round_num: int,
        selected_client_ids: Sequence[int],
        full_ab_bytes: Mapping[int, int],
        allocated_layers: Mapping[int, Set[str]],
        train_samples: Mapping[int, int],
        layer_names: Sequence[str],
    ) -> Optional[Dict[int, float]]:
        clients = [int(cid) for cid in selected_client_ids]
        if not clients:
            raise ValueError("selected_client_ids must not be empty")
        if not self.rift_enabled:
            return None
        if not layer_names:
            raise ValueError("layer_names must not be empty")
        compute_time_s: Dict[int, float] = {}
        active_blocks = {
            cid: len(allocated_layers.get(cid, set())) for cid in clients
        }
        for cid in clients:
            cycles = compute_cycles(
                n_samples=int(train_samples.get(cid, 0)),
                c0=float(self.profiles.c0_cycles_per_sample),
                active_blocks=active_blocks[cid],
                c_lora=float(self.profiles.c_lora_cycles_per_sample),
            )
            compute_time_s[cid] = cycles / self.profiles.device(cid).freq_hz
        channel = self.trace.channel_vector(int(round_num))
        fallback = str(layer_names[0])
        prev_stale = {
            cid: float(self.staleness.effective(
                cid,
                next(iter(allocated_layers.get(cid, set())), fallback),
            ))
            for cid in clients
        }
        pre = None
        try:
            pre = solve_rift_cora_preallocation(
                clients=clients,
                full_ab_bytes={cid: int(full_ab_bytes.get(cid, 0)) for cid in clients},
                classifier_bytes={cid: 0 for cid in clients},
                compute_time_s=compute_time_s,
                channel_gain={cid: float(channel[cid]) for cid in clients},
                reliability=self.est_success,
                budget_ratio=float(self.config.rift_budget_ratio),
                reliability_floor=float(self.config.rift_reliability_floor),
                metadata_bits=float(self.profiles.s_meta_bits),
                total_bandwidth_hz=float(self.profiles.total_bandwidth_hz),
                power=self.profiles.power_profile,
                n0_w_hz=float(self.profiles.channel.n0_w_hz),
                d_min=float(self.config.d_min_s),
                d_max=float(self.config.d_max_s),
                lambda_e=float(self.config.lambda_e),
                lambda_u=float(self.config.lambda_u),
                omega_net=float(self.config.omega_net),
                prev_staleness=prev_stale,
                duration_hazard_rate=float(self.config.duration_hazard_rate),
                deadline_grid_size=int(self.config.deadline_grid_size),
            )
        except RuntimeError:
            # Degraded-round semantics: an overloaded client (compute time
            # above the deadline ceiling under an extreme Dirichlet split)
            # cannot arrive this round under ANY radio allocation.  The
            # equal-fixed controller handles the same situation implicitly as
            # a per-client deadline miss; mirror that instead of aborting the
            # whole run.  RIFT then runs without predictor reference rates
            # (duration value disabled for this round only, like the legacy
            # vision protocol) and the corrector below re-detects the miss.
            pre = None
        self._reference_rates = dict(pre.reference_rate_bps) if pre else {}
        self._pre_round_degraded = pre is None
        return dict(self._reference_rates) if pre else None

    def reference_rate(self, client_id: int) -> Optional[float]:
        rate = self._reference_rates.get(int(client_id))
        return None if rate is None else float(rate)

    # ------------------------------------------------------------------
    # 2. RIFT-LEF factor selection (after local training)
    # ------------------------------------------------------------------
    def select_factors(
        self,
        *,
        round_num: int,
        client_id: int,
        a_after: Mapping[str, torch.Tensor],
        b_after: Mapping[str, torch.Tensor],
        delta_a: Mapping[str, torch.Tensor],
        delta_b: Mapping[str, torch.Tensor],
        allocated_layers: Set[str],
    ) -> Tuple[dict, dict, Optional[RiftSelection]]:
        cid = int(client_id)
        self._round_allocated[cid] = set(allocated_layers)
        if not self.rift_enabled:
            self._round_corrected[cid] = (dict(delta_a), dict(delta_b))
            self._round_choices[cid] = {}
            return dict(delta_a), dict(delta_b), None

        assert self.rift_ef_state is not None
        corrected_a, corrected_b, before_l2 = self.rift_ef_state.corrected(
            cid, delta_a, delta_b
        )
        selection_a_after = {
            name: a_after[name] + (corrected_a[name] - delta_a[name])
            for name in corrected_a
        }
        selection_b_after = {
            name: b_after[name] + (corrected_b[name] - delta_b[name])
            for name in corrected_b
        }
        stale = {
            name: float(self.staleness.effective(cid, name))
            for name in allocated_layers
        }
        rate = self.reference_rate(cid)
        # In a degraded round there is no predictor rate; the duration-aware
        # value must then be disabled (hazard=0) rather than crash, mirroring
        # the legacy vision protocol's un-rated RIFT rounds.
        hazard = (
            float(self.config.duration_hazard_rate) if rate is not None else 0.0
        )
        selection = select_rift_updates(
            a_after=selection_a_after,
            b_after=selection_b_after,
            delta_a=corrected_a,
            delta_b=corrected_b,
            budget_ratio=float(self.config.rift_budget_ratio),
            reliability=float(self.est_success.get(cid, 1.0)),
            staleness=stale,
            staleness_weight=float(self.config.rift_staleness_weight),
            reliability_floor=float(self.config.rift_reliability_floor),
            quantum_bytes=int(self.config.rift_quantum_bytes),
            delivery_rate_bps=rate,
            duration_hazard_rate=hazard,
        )
        self._round_corrected[cid] = (corrected_a, corrected_b)
        self._round_choices[cid] = dict(selection.choices)
        return selection.updates_A, selection.updates_B, selection

    # ------------------------------------------------------------------
    # 3. CORA corrector + arrival gate + state updates
    # ------------------------------------------------------------------
    def run_round(
        self,
        *,
        round_num: int,
        selected_client_ids: Sequence[int],
        payload_bytes: Mapping[int, int],
        active_blocks: Mapping[int, int],
        train_samples: Mapping[int, int],
        wall_clock_s: float,
        download_bytes: float,
        all_layer_names: Sequence[str],
        rift_records: Optional[Mapping[int, RiftSelection]] = None,
    ) -> FedSonataTextMecRound:
        clients = [int(cid) for cid in selected_client_ids]
        if not clients:
            raise ValueError("selected_client_ids must not be empty")
        n0 = float(self.profiles.channel.n0_w_hz)
        channel = self.trace.channel_vector(int(round_num))
        exogenous = self.trace.dropout_vector(int(round_num))

        s_up_bits: Dict[int, float] = {}
        t_cmp: Dict[int, float] = {}
        h: Dict[int, float] = {}
        cycles_by_client: Dict[int, float] = {}
        for cid in clients:
            raw_bytes = int(payload_bytes.get(cid, 0))
            if raw_bytes < 0:
                raise ValueError("payload bytes must be non-negative")
            s_up_bits[cid] = float(self.profiles.s_meta_bits) + 8.0 * float(raw_bytes)
            cycles = compute_cycles(
                n_samples=int(train_samples.get(cid, 0)),
                c0=float(self.profiles.c0_cycles_per_sample),
                active_blocks=int(active_blocks.get(cid, 0)),
                c_lora=float(self.profiles.c_lora_cycles_per_sample),
            )
            cycles_by_client[cid] = cycles
            t_cmp[cid] = cycles / self.profiles.device(cid).freq_hz
            h[cid] = float(channel[cid])

        prev_stale = {
            cid: float(self.staleness.effective(
                cid,
                next(iter(self._round_allocated.get(cid, set(all_layer_names))), next(iter(all_layer_names))),
            ))
            for cid in clients
        }

        solver_ms = 0.0
        deadline = float(self.config.d_max_s)
        bandwidth = {
            cid: float(self.profiles.total_bandwidth_hz) / float(len(clients))
            for cid in clients
        }
        power_w = {
            cid: float(self.profiles.power_profile.p_max_w) for cid in clients
        }
        resource_objective = float("nan")
        cora_predictor_deadline = None
        cora_degraded = False
        if self.factor_policy in {"fedsonata_cora", "cora"}:
            t0 = time.perf_counter()
            try:
                plan = solve_convex_round(
                    clients,
                    s_up_bits,
                    t_cmp,
                    h,
                    float(self.profiles.total_bandwidth_hz),
                    self.profiles.power_profile,
                    n0,
                    float(self.config.d_min_s),
                    float(self.config.d_max_s),
                    lambda_e=float(self.config.lambda_e),
                    lambda_u=float(self.config.lambda_u),
                    omega_net=float(self.config.omega_net),
                    prev_staleness=prev_stale,
                    est_success=self.est_success,
                    duration_hazard_rate=float(self.config.duration_hazard_rate),
                    deadline_grid_size=int(self.config.deadline_grid_size),
                )
                if not plan.feasible:
                    raise RuntimeError("text CORA corrector reported infeasible")
            except RuntimeError:
                # Degraded round: fall back to the equal-fixed allocation and
                # let the per-client arrival gate mark overloaded clients as
                # deadline misses (mirrors the equal-fixed controller).
                plan = None
                cora_degraded = True
            solver_ms = (time.perf_counter() - t0) * 1000.0
            if plan is not None:
                deadline = float(plan.deadline_s)
                bandwidth = {cid: float(plan.bandwidth_hz[cid]) for cid in clients}
                power_w = {cid: float(plan.power_w[cid]) for cid in clients}
                resource_objective = float(plan.objective)

        results: List[dict] = []
        for cid in clients:
            dev = self.profiles.device(cid)
            _, e_cmp = compute_time_energy(cycles_by_client[cid], dev)
            rate = shannon_rate(bandwidth[cid], power_w[cid], h[cid], n0)
            full_t_up = s_up_bits[cid] / rate if rate > 0.0 else float("inf")
            base_dropped = bool(exogenous[cid])
            duration_probability = (
                1.0 - math.exp(-float(self.config.duration_hazard_rate) * full_t_up)
                if math.isfinite(full_t_up) else 1.0
            )
            duration_dropped = bool(
                (not base_dropped)
                and self.trace.duration_failure(
                    int(round_num), cid, full_t_up,
                    float(self.config.duration_hazard_rate),
                )
            )
            dropped = bool(base_dropped or duration_dropped)
            if base_dropped:
                attempted_t_up = 0.0
            elif duration_dropped:
                attempted_t_up = min(
                    full_t_up,
                    self.trace.duration_failure_time(
                        int(round_num), cid, float(self.config.duration_hazard_rate)
                    ),
                )
            else:
                attempted_t_up = full_t_up
            transmitted_bits = min(
                s_up_bits[cid], max(0.0, rate * attempted_t_up)
            )
            e_tx = (
                power_w[cid] / float(self.profiles.power_profile.pa_efficiency)
                + float(self.profiles.power_profile.circuit_power_w)
            ) * attempted_t_up
            arrived = bool(
                (not dropped)
                and math.isfinite(full_t_up)
                and (t_cmp[cid] + full_t_up <= deadline + 1e-9)
            )
            failure_type = (
                "dropout" if base_dropped
                else "duration_dropout" if duration_dropped
                else "deadline_miss" if not arrived
                else "none"
            )
            # LEF ACK: clear only factors confirmed delivered this round.
            ef_stats = None
            if self.rift_enabled and self.rift_ef_state is not None:
                corrected = self._round_corrected.get(cid, ({}, {}))
                ef_stats = self.rift_ef_state.acknowledge(
                    cid,
                    corrected[0] or {},
                    corrected[1] or {},
                    self._round_choices.get(cid, {}),
                    arrived=arrived,
                )
            # Effective staleness for the NEXT round's RIFT value.
            update_effective_staleness(
                self.staleness,
                cid,
                self._round_allocated.get(cid, set()),
                list(all_layer_names),
                arrived,
                failure_type,
            )
            _update_success_ewma(
                self.est_success, cid, arrived,
                float(self.config.reliability_beta),
            )
            record = (rift_records or {}).get(cid)
            row = {
                "client_id": cid,
                "arrived": arrived,
                "dropped": dropped,
                "base_dropped": base_dropped,
                "duration_dropped": duration_dropped,
                "failure_type": failure_type,
                "active_blocks": int(active_blocks.get(cid, 0)),
                "payload_bits": s_up_bits[cid],
                "payload_tensor_bytes": int(payload_bytes.get(cid, 0)),
                "transmitted_bits": transmitted_bits,
                "t_cmp_s": float(t_cmp[cid]),
                "t_up_s": float(attempted_t_up),
                "full_t_up_s": float(full_t_up),
                "e_cmp_j": float(e_cmp),
                "e_tx_j": float(e_tx),
                "channel_gain": h[cid],
                "bandwidth_hz": float(bandwidth[cid]),
                "power_w": float(power_w[cid]),
                "deadline_s": float(deadline),
                "duration_failure_probability": float(duration_probability),
                "reliability": float(self.est_success[cid]),
            }
            if record is not None:
                row.update({
                    "rift_selected_bytes": int(record.selected_bytes),
                    "rift_budget_bytes": int(record.budget_bytes),
                    "rift_captured_gain": float(record.captured_gain),
                    "rift_oracle_gain": float(record.oracle_gain),
                    "rift_expected_captured_gain": float(record.expected_captured_gain),
                    "rift_choices": dict(record.choices),
                    "rift_option_counts": dict(record.option_counts),
                })
            if ef_stats is not None:
                row.update({
                    "rift_ef_cleared_A": int(ef_stats["cleared_a"]),
                    "rift_ef_cleared_B": int(ef_stats["cleared_b"]),
                    "rift_ef_residual_l2": float(ef_stats["residual_l2"]),
                    "rift_ef_residual_factors": int(ef_stats["residual_factors"]),
                })
            results.append(row)

        self.metrics.add_round(
            round_num=int(round_num),
            client_results=results,
            wall_clock_s=float(wall_clock_s),
            download_bytes=float(download_bytes),
            lpc_events=0,
            solver_ms=float(solver_ms),
        )
        arrived_ids = [int(row["client_id"]) for row in results if row["arrived"]]
        gains = [float(row["channel_gain"]) for row in results]
        diagnostics: Dict[str, object] = {
            "network_policy": (
                "convex_joint_text" if self.factor_policy in {"fedsonata_cora", "cora"}
                else "equal_fixed"
            ),
            "factor_policy": self.factor_policy,
            "mec_method": (
                "proposed_joint_rift_ef_cora_text" if self.rift_enabled
                else "proposed_joint_cora_text" if self.factor_policy == "cora"
                else "equal_fixed"
            ),
            "scheduled_clients": len(results),
            "arrived_clients": len(arrived_ids),
            "base_dropout_clients": sum(bool(r["base_dropped"]) for r in results),
            "duration_dropout_clients": sum(bool(r["duration_dropped"]) for r in results),
            "deadline_miss_clients": sum(r["failure_type"] == "deadline_miss" for r in results),
            "scheduled_upload_bytes_round": float(sum(r["payload_bits"] for r in results)) / 8.0,
            "attempted_upload_bytes_round": float(sum(r["transmitted_bits"] for r in results)) / 8.0,
            "delivered_upload_bytes_round": float(sum(r["payload_bits"] for r in results if r["arrived"])) / 8.0,
            "compute_energy_round_j": float(sum(r["e_cmp_j"] for r in results)),
            "tx_energy_round_j": float(sum(r["e_tx_j"] for r in results)),
            "channel_gain_mean": float(sum(gains) / len(gains)),
            "channel_gain_min": float(min(gains)),
            "channel_gain_max": float(max(gains)),
            "deadline_s": float(deadline),
            "solver_ms": float(solver_ms),
            "resource_objective": (
                float(resource_objective) if math.isfinite(resource_objective) else None
            ),
            "bandwidth_hz": {str(r["client_id"]): r["bandwidth_hz"] for r in results},
            "power_w": {str(r["client_id"]): r["power_w"] for r in results},
            "bandwidth_sum_hz": float(sum(r["bandwidth_hz"] for r in results)),
            "reliability_mean": float(sum(r["reliability"] for r in results) / len(results)),
            "trace_hash": self.trace_hash,
            "energy_model": "model-based simulated",
            "cora_rate_source": (
                REFERENCE_RATE_SOURCE
                if self.rift_enabled and self._reference_rates
                else None
            ),
            "cora_degraded_round": bool(
                cora_degraded or self._pre_round_degraded
            ),
        }
        if self.rift_enabled:
            diagnostics["rift_budget_bytes"] = int(sum(
                r.get("rift_budget_bytes", 0) for r in results
            ))
            diagnostics["rift_selected_bytes"] = int(sum(
                r.get("rift_selected_bytes", 0) for r in results
            ))
            diagnostics["rift_captured_gain_ratio"] = float(
                sum(r.get("rift_captured_gain", 0.0) for r in results)
                / max(1e-30, sum(r.get("rift_oracle_gain", 0.0) for r in results))
            )
        # Clear per-round buffers.
        self._round_corrected = {}
        self._round_choices = {}
        self._round_allocated = {}
        self._reference_rates = {}
        self._pre_round_degraded = False
        return FedSonataTextMecRound(results, arrived_ids, diagnostics)
