"""MEC transport adapter for the Fed-HeLLo-style RoBERTa baseline.

This module reuses the CIFAR-100 system primitives but keeps the first text
experiment deliberately narrow: native Fed-HeLLo layer allocation plus equal
bandwidth, maximum-power transmission.  Actual uploaded A/B tensor bytes drive
Shannon latency, duration interruption, the arrival gate, and simulated energy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

from .metrics import MecMetrics
from .profiles import ProfileSet, build_default_profiles
from .system_model import compute_cycles, compute_time_energy, shannon_rate
from .traces import TraceGenerator, compute_trace_hash


@dataclass(frozen=True)
class FedHelloMecConfig:
    num_clients: int
    num_rounds: int
    seed: int
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
    deadline_s: float = 10.0


@dataclass
class FedHelloMecRound:
    client_results: List[dict]
    arrived_client_ids: List[int]
    diagnostics: Dict[str, object]


class FedHelloMecController:
    """Deterministic equal-fixed wireless controller with an arrival gate."""

    def __init__(self, config: FedHelloMecConfig):
        self.config = config
        if int(config.num_clients) < 1 or int(config.num_rounds) < 1:
            raise ValueError("num_clients and num_rounds must be positive")
        if float(config.duration_hazard_rate) < 0.0:
            raise ValueError("duration_hazard_rate must be non-negative")
        if float(config.deadline_s) <= 0.0:
            raise ValueError("deadline_s must be positive")
        if not 0.0 <= float(config.channel_correlation) < 1.0:
            raise ValueError("channel_correlation must be in [0,1)")
        if float(config.channel_gain_mean) <= 0.0:
            raise ValueError("channel_gain_mean must be positive")

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
        self.trace_hash = compute_trace_hash(
            seed=int(config.seed),
            num_clients=int(config.num_clients),
            channel=self.profiles.channel,
            dropout=self.profiles.dropout,
            rounds=int(config.num_rounds),
            duration_hazard_rate=float(config.duration_hazard_rate),
        )
        self.metrics = MecMetrics()

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
    ) -> FedHelloMecRound:
        clients = [int(cid) for cid in selected_client_ids]
        if len(clients) == 0:
            raise ValueError("selected_client_ids must not be empty")
        bandwidth_hz = float(self.profiles.total_bandwidth_hz) / float(len(clients))
        power_w = float(self.profiles.power_profile.p_max_w)
        n0 = float(self.profiles.channel.n0_w_hz)
        channel = self.trace.channel_vector(int(round_num))
        exogenous = self.trace.dropout_vector(int(round_num))
        results: List[dict] = []

        for cid in clients:
            raw_bytes = int(payload_bytes.get(cid, 0))
            if raw_bytes < 0:
                raise ValueError("payload bytes must be non-negative")
            bits = float(self.profiles.s_meta_bits) + 8.0 * float(raw_bytes)
            cycles = compute_cycles(
                n_samples=int(train_samples.get(cid, 0)),
                c0=float(self.profiles.c0_cycles_per_sample),
                active_blocks=int(active_blocks.get(cid, 0)),
                c_lora=float(self.profiles.c_lora_cycles_per_sample),
            )
            t_cmp, e_cmp = compute_time_energy(cycles, self.profiles.device(cid))
            h = float(channel[cid])
            rate = shannon_rate(bandwidth_hz, power_w, h, n0)
            full_t_up = bits / rate if rate > 0.0 else float("inf")
            base_dropped = bool(exogenous[cid])
            duration_probability = (
                1.0 - math.exp(-float(self.config.duration_hazard_rate) * full_t_up)
                if math.isfinite(full_t_up) else 1.0
            )
            duration_dropped = bool(
                (not base_dropped)
                and self.trace.duration_failure(
                    int(round_num), cid, full_t_up, float(self.config.duration_hazard_rate)
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
            transmitted_bits = min(bits, max(0.0, rate * attempted_t_up))
            e_tx = (
                power_w / float(self.profiles.power_profile.pa_efficiency)
                + float(self.profiles.power_profile.circuit_power_w)
            ) * attempted_t_up
            arrived = bool(
                (not dropped)
                and math.isfinite(full_t_up)
                and (t_cmp + full_t_up <= float(self.config.deadline_s))
            )
            failure_type = (
                "dropout" if base_dropped
                else "duration_dropout" if duration_dropped
                else "deadline_miss" if not arrived
                else "none"
            )
            results.append(
                {
                    "client_id": cid,
                    "arrived": arrived,
                    "dropped": dropped,
                    "base_dropped": base_dropped,
                    "duration_dropped": duration_dropped,
                    "failure_type": failure_type,
                    "active_blocks": int(active_blocks.get(cid, 0)),
                    "payload_bits": bits,
                    "payload_tensor_bytes": raw_bytes,
                    "transmitted_bits": transmitted_bits,
                    "t_cmp_s": float(t_cmp),
                    "t_up_s": float(attempted_t_up),
                    "full_t_up_s": float(full_t_up),
                    "e_cmp_j": float(e_cmp),
                    "e_tx_j": float(e_tx),
                    "channel_gain": h,
                    "bandwidth_hz": bandwidth_hz,
                    "power_w": power_w,
                    "deadline_s": float(self.config.deadline_s),
                    "duration_failure_probability": float(duration_probability),
                }
            )

        self.metrics.add_round(
            round_num=int(round_num),
            client_results=results,
            wall_clock_s=float(wall_clock_s),
            download_bytes=float(download_bytes),
            lpc_events=0,
            solver_ms=0.0,
        )
        arrived_ids = [int(row["client_id"]) for row in results if bool(row["arrived"])]
        gains = [float(row["channel_gain"]) for row in results]
        diagnostics: Dict[str, object] = {
            "network_policy": "equal_fixed",
            "scheduled_clients": len(results),
            "arrived_clients": len(arrived_ids),
            "base_dropout_clients": sum(bool(row["base_dropped"]) for row in results),
            "duration_dropout_clients": sum(bool(row["duration_dropped"]) for row in results),
            "deadline_miss_clients": sum(row["failure_type"] == "deadline_miss" for row in results),
            "scheduled_upload_bytes_round": float(sum(row["payload_bits"] for row in results)) / 8.0,
            "attempted_upload_bytes_round": float(sum(row["transmitted_bits"] for row in results)) / 8.0,
            "delivered_upload_bytes_round": float(sum(row["payload_bits"] for row in results if row["arrived"])) / 8.0,
            "compute_energy_round_j": float(sum(row["e_cmp_j"] for row in results)),
            "tx_energy_round_j": float(sum(row["e_tx_j"] for row in results)),
            "channel_gain_mean": float(sum(gains) / len(gains)),
            "channel_gain_min": float(min(gains)),
            "channel_gain_max": float(max(gains)),
            "bandwidth_hz_per_client": bandwidth_hz,
            "bandwidth_sum_hz": bandwidth_hz * len(results),
            "power_w": power_w,
            "deadline_s": float(self.config.deadline_s),
            "trace_hash": self.trace_hash,
            "energy_model": "model-based simulated",
        }
        return FedHelloMecRound(results, arrived_ids, diagnostics)

