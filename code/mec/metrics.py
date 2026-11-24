"""Per-round and cumulative MEC system metrics.

Builds the `round_summary` dict consumed by scripts/audit_results.py. The 11
REQUIRED fields are produced exactly:

    round, accuracy_personalized, accuracy_global, mec_latency_s,
    wall_clock_s, compute_energy_j, tx_energy_j, on_time_rate, lpc_events,
    upload_bytes, download_bytes

Unit discipline (RESULT_AUDIT sec.4): seconds, Joule and bytes are kept
strictly separate and non-negative. mec_latency_s is the modeled parallel MEC
round time (max over clients of completion time, or the deadline used), fully
separate from wall_clock_s (the real sequential server elapsed time).

All energy is model-based simulated energy (D006) unless otherwise calibrated;
this flag is carried in the summary for honest reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MecMetrics:
    """Accumulates system diagnostics for one run across rounds."""

    total_compute_j: float = 0.0
    total_tx_j: float = 0.0
    total_upload_bytes: float = 0.0
    total_download_bytes: float = 0.0
    total_lpc_events: int = 0
    rounds: int = 0
    # per-round buffers (last round values)
    last_on_time_rate: float = 0.0
    last_mec_latency_s: float = 0.0
    last_wall_clock_s: float = 0.0
    last_dropout_rate: float = 0.0
    last_deadline_miss_rate: float = 0.0
    solver_ms_total: float = 0.0

    def add_round(
        self,
        round_num: int,
        client_results: List[dict],
        wall_clock_s: float,
        download_bytes: float,
        lpc_events: int,
        solver_ms: float,
    ) -> None:
        """Fold one round's per-client dicts into accumulators.

        Each client dict has keys: arrived(bool), t_cmp_s, t_up_s, e_cmp_j,
        e_tx_j, payload_bits, transmitted_bits, dropped(bool), failure_type(str).
        """
        self.rounds = round_num
        self.solver_ms_total += solver_ms
        self.last_wall_clock_s = wall_clock_s

        n = len(client_results)
        if n == 0:
            self.last_on_time_rate = 0.0
            self.last_mec_latency_s = 0.0
            return

        arrived = sum(1 for c in client_results if c.get("arrived", False))
        dropped = sum(1 for c in client_results if c.get("dropped", False))
        missed = sum(
            1 for c in client_results
            if not c.get("arrived", False) and not c.get("dropped", False)
        )
        self.last_on_time_rate = arrived / n
        self.last_dropout_rate = dropped / n
        self.last_deadline_miss_rate = missed / n

        # MEC latency: modeled parallel round time = max completion time among
        # clients that attempted upload; if none, the modeled deadline.
        completions = [
            c["t_cmp_s"] + c["t_up_s"]
            for c in client_results
            if c.get("t_cmp_s") is not None and c.get("t_up_s") is not None
        ]
        self.last_mec_latency_s = max(completions) if completions else 0.0

        for c in client_results:
            # Count attempted resource use. A duration interruption or deadline
            # miss still consumes airtime and energy before the upload aborts.
            self.total_compute_j += float(c.get("e_cmp_j", 0.0))
            self.total_tx_j += float(c.get("e_tx_j", 0.0))
            transmitted_bits = c.get("transmitted_bits")
            if transmitted_bits is None:
                transmitted_bits = (
                    c.get("payload_bits", 0.0) if c.get("arrived", False) else 0.0
                )
            self.total_upload_bytes += float(transmitted_bits) / 8.0

        self.total_download_bytes += download_bytes
        self.total_lpc_events += lpc_events

    def round_summary(
        self,
        round_num: int,
        accuracy_personalized: float,
        accuracy_global: float,
    ) -> Dict[str, float]:
        """Produce the exact audit-required round_summary row."""
        return {
            "row_type": "round_summary",
            "round": int(round_num),
            "accuracy_personalized": float(accuracy_personalized),
            "accuracy_global": float(accuracy_global),
            "mec_latency_s": float(self.last_mec_latency_s),
            "wall_clock_s": float(self.last_wall_clock_s),
            "compute_energy_j": float(self.total_compute_j),
            "tx_energy_j": float(self.total_tx_j),
            "on_time_rate": float(self.last_on_time_rate),
            "lpc_events": int(self.total_lpc_events),
            "upload_bytes": float(self.total_upload_bytes),
            "download_bytes": float(self.total_download_bytes),
            # honest labeling of energy source (D006)
            "energy_model": "model-based simulated",
        }


def make_round_summary(
    round_num: int,
    accuracy_personalized: float,
    accuracy_global: float,
    mec_latency_s: float,
    wall_clock_s: float,
    compute_energy_j: float,
    tx_energy_j: float,
    on_time_rate: float,
    lpc_events: int,
    upload_bytes: float,
    download_bytes: float,
    accuracy_in_percent: bool = True,
) -> Dict[str, float]:
    """Convenience builder with explicit unit discipline.

    accuracy_in_percent: if True, converts percentages in [0,100] to the
    audit-required [0,1] range.
    """
    if accuracy_in_percent:
        ap = accuracy_personalized / 100.0
        ag = accuracy_global / 100.0
    else:
        ap = accuracy_personalized
        ag = accuracy_global
    if not (0.0 <= ap <= 1.0 and 0.0 <= ag <= 1.0):
        raise ValueError("accuracy must map into [0,1]")
    if any(v < 0 for v in (
        mec_latency_s, wall_clock_s, compute_energy_j, tx_energy_j,
        upload_bytes, download_bytes,
    )):
        raise ValueError("seconds/joules/bytes must be non-negative")
    if not (0.0 <= on_time_rate <= 1.0):
        raise ValueError("on_time_rate must be in [0,1]")
    return {
        "row_type": "round_summary",
        "round": int(round_num),
        "accuracy_personalized": float(ap),
        "accuracy_global": float(ag),
        "mec_latency_s": float(mec_latency_s),
        "wall_clock_s": float(wall_clock_s),
        "compute_energy_j": float(compute_energy_j),
        "tx_energy_j": float(tx_energy_j),
        "on_time_rate": float(on_time_rate),
        "lpc_events": int(lpc_events),
        "upload_bytes": float(upload_bytes),
        "download_bytes": float(download_bytes),
        "energy_model": "model-based simulated",
    }
