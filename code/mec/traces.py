"""Deterministic MEC traces: channel and dropout realisations + hashing.

All randomness flows from numpy Generators seeded deterministically by
(seed, round, client, stream). Channel/dropout values are pure functions of
(seed, round, client): they are recomputed from round 1 on each call, so a
fresh TraceGenerator and a long-lived one produce identical traces. This makes
standalone use, persistent-loop use, and compute_trace_hash fully consistent.

Identical (seed, channel_model, dropout_rate, bandwidth_mhz, num_clients,
round) MUST produce identical traces -- a hard Phase-0 test and required for
fair paired comparison (BASELINES_AND_SOURCES sec.6).

The SHA256 over the realised trace bytes is recorded in run config so that
different dropouts/channels cannot be silently compared as if matched.

FS23 network scenarios (2026-11-17): optional ``ge_schedule`` /
``hazard_schedule`` support pre-declared phase-varying Gilbert-Elliott
transition probabilities and duration hazards (PROTOCOL_CN SHIFT / IDEAL).
The GE chain is rolled forward from round 1 with each round's own (p, q), so
state stays continuous across phase boundaries and the function remains pure
in (seed, round, client).  With no schedule the behaviour is byte-identical
to the archived Nov6 traces.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .profiles import ChannelProfile, DropoutProfile

# (round_start, round_end, p_good_to_bad, q_bad_to_good) 1-based inclusive.
GePhase = Tuple[int, int, float, float]
# (round_start, round_end, hazard_per_s) 1-based inclusive.
HazardPhase = Tuple[int, int, float]


def _phase_lookup(schedule: Optional[Sequence[Tuple]], round_num: int, kind: str):
    if not schedule:
        return None
    for phase in schedule:
        r0, r1 = int(phase[0]), int(phase[1])
        if r0 <= round_num <= r1:
            return phase
    raise ValueError(f"round {round_num} not covered by {kind} schedule")


@dataclass
class TraceGenerator:
    """Generates per-round channel and dropout traces deterministically."""

    seed: int
    num_clients: int
    channel: ChannelProfile
    dropout: DropoutProfile
    ge_schedule: Optional[Sequence[GePhase]] = None
    hazard_schedule: Optional[Sequence[HazardPhase]] = None
    default_hazard: float = 0.0

    def ge_params(self, round_num: int) -> Tuple[float, float]:
        """(p, q) for this round; profile values when no schedule."""
        phase = _phase_lookup(self.ge_schedule, round_num, "ge")
        if phase is None:
            return (
                float(self.dropout.ge_good_to_bad),
                float(self.dropout.ge_bad_to_good),
            )
        return float(phase[2]), float(phase[3])

    def hazard_rate(self, round_num: int) -> float:
        """Duration hazard for this round; ``default_hazard`` when no schedule."""
        phase = _phase_lookup(self.hazard_schedule, round_num, "hazard")
        if phase is None:
            return float(self.default_hazard)
        return float(phase[2])

    def channel_gain(self, round_num: int, client_id: int) -> float:
        """Channel power gain h_k^t, pure function of (seed, round, client)."""
        if self.channel.model == "constant":
            return float(self.channel.constant_gain)
        if self.channel.model == "rayleigh":
            rng = np.random.default_rng([self.seed, client_id, 0])
            h = self.channel.rayleigh_mean
            for _ in range(1, round_num + 1):
                g = rng.standard_normal() + 1j * rng.standard_normal()
                g *= np.sqrt(self.channel.rayleigh_mean / 2.0)
                h = abs(g) ** 2
            return float(h)
        if self.channel.model == "correlated":
            # Stationary complex Gauss-Markov fading.  Correlating the complex
            # coefficient (rather than positive power samples) preserves
            # E[|g_t|^2] = rayleigh_mean for every round.
            rng = np.random.default_rng([self.seed, client_id, 0])
            scale = np.sqrt(self.channel.rayleigh_mean / 2.0)
            g = (rng.standard_normal() + 1j * rng.standard_normal()) * scale
            rho = float(self.channel.correlation)
            innovation_scale = np.sqrt(max(0.0, 1.0 - rho ** 2))
            for _ in range(1, round_num + 1):
                innovation = (
                    rng.standard_normal() + 1j * rng.standard_normal()
                ) * scale
                g = rho * g + innovation_scale * innovation
            return float(abs(g) ** 2)
        raise ValueError(f"unknown channel model {self.channel.model}")

    def dropout_flag(self, round_num: int, client_id: int) -> bool:
        """Exogenous dropout d_k^t, pure function of (seed, round, client).

        - none: never drops.
        - bernoulli: d ~ Bernoulli(dropout_rate).
        - gilbert_elliott: two-state Markov chain tuned to dropout_rate;
          the chain is rolled forward from round 1 so the steady-state mean
          equals dropout_rate regardless of call order.  With a ge_schedule
          each round uses its own (p, q); state is NOT reset at phase
          boundaries (PROTOCOL_CN SHIFT: "阶段边界继续维护 GE 状态").
        """
        if self.dropout.model == "none":
            return False
        if self.dropout.model == "bernoulli":
            # Independent per-round Bernoulli draw at round t; pure in
            # (seed, client, round). Uses dropout_rate directly.
            rng = np.random.default_rng([self.seed, round_num, client_id, 1])
            return bool(rng.random() < self.dropout.dropout_rate)
        if self.dropout.model == "gilbert_elliott":
            rng = np.random.default_rng([self.seed, client_id, 1])
            bad = False
            for t in range(1, round_num + 1):
                p, q = self.ge_params(t)
                u = rng.random()
                if bad:
                    bad = bool(u < (1.0 - q))
                else:
                    bad = bool(u < p)
            return bad
        raise ValueError(f"unknown dropout model {self.dropout.model}")

    def channel_vector(self, round_num: int) -> List[float]:
        return [self.channel_gain(round_num, k) for k in range(self.num_clients)]

    def dropout_vector(self, round_num: int) -> List[bool]:
        return [self.dropout_flag(round_num, k) for k in range(self.num_clients)]

    def duration_failure_uniform(self, round_num: int, client_id: int) -> float:
        """Latent matched uniform for payload-duration-dependent interruption."""
        rng = np.random.default_rng([self.seed, round_num, client_id, 2])
        return float(rng.random())

    def duration_failure(
        self, round_num: int, client_id: int, upload_time_s: float, hazard_rate: float
    ) -> bool:
        if upload_time_s < 0.0 or hazard_rate < 0.0:
            raise ValueError("upload_time_s and hazard_rate must be non-negative")
        if hazard_rate == 0.0:
            # FS23: zero-hazard branch -- "no event", no division (PROTOCOL_CN).
            return False
        probability = 1.0 - np.exp(-float(hazard_rate) * float(upload_time_s))
        return bool(self.duration_failure_uniform(round_num, client_id) < probability)

    def duration_failure_time(
        self, round_num: int, client_id: int, hazard_rate: float
    ) -> float:
        """Matched exponential interruption time for partial-Tx accounting."""
        if hazard_rate < 0.0:
            raise ValueError("hazard_rate must be non-negative")
        if hazard_rate == 0.0:
            return float("inf")
        u = self.duration_failure_uniform(round_num, client_id)
        return -math.log1p(-u) / float(hazard_rate)


def generate_channel_trace(
    seed: int,
    num_clients: int,
    channel: ChannelProfile,
    round_num: int,
) -> np.ndarray:
    """Standalone deterministic channel gain vector (used by unit tests)."""
    gen = TraceGenerator(seed, num_clients, channel, DropoutProfile(model="none"))
    return np.array(gen.channel_vector(round_num), dtype=float)


def generate_dropout_trace(
    seed: int,
    num_clients: int,
    dropout: DropoutProfile,
    round_num: int,
) -> np.ndarray:
    """Standalone deterministic dropout vector (used by unit tests)."""
    gen = TraceGenerator(seed, num_clients, ChannelProfile(model="constant"), dropout)
    return np.array(gen.dropout_vector(round_num), dtype=bool)


def compute_trace_hash(
    seed: int,
    num_clients: int,
    channel: ChannelProfile,
    dropout: DropoutProfile,
    rounds: int,
    duration_hazard_rate: float = 0.0,
    ge_schedule: Optional[Sequence[GePhase]] = None,
    hazard_schedule: Optional[Sequence[HazardPhase]] = None,
) -> str:
    """SHA256 over the concatenated realised channel+dropout bytes.

    The hash captures exactly the random realisation used by a run, so two
    runs with differing traces are never silently compared as matched.
    FS23: when a scenario schedule is active it is hashed in as well and the
    per-round hazards select which duration uniforms enter the digest.
    """
    gen = TraceGenerator(
        seed, num_clients, channel, dropout,
        ge_schedule=ge_schedule, hazard_schedule=hazard_schedule,
        default_hazard=float(duration_hazard_rate),
    )
    h = hashlib.sha256()
    if ge_schedule:
        h.update(repr(
            [(int(a), int(b), float(c), float(d)) for a, b, c, d in ge_schedule]
        ).encode())
    if hazard_schedule:
        h.update(repr(
            [(int(a), int(b), float(c)) for a, b, c in hazard_schedule]
        ).encode())
    for t in range(1, rounds + 1):
        ch = np.array(gen.channel_vector(t), dtype="<f8")
        dr = np.packbits(np.array(gen.dropout_vector(t), dtype=bool))
        h.update(ch.tobytes())
        h.update(dr.tobytes())
        hazard_t = (
            gen.hazard_rate(t) if hazard_schedule
            else float(duration_hazard_rate)
        )
        if float(hazard_t) > 0.0:
            duration_u = np.array(
                [gen.duration_failure_uniform(t, k) for k in range(num_clients)],
                dtype="<f8",
            )
            h.update(duration_u.tobytes())
    if hazard_schedule:
        h.update(
            np.asarray(
                [float(hz) for _, _, hz in hazard_schedule], dtype="<f8"
            ).tobytes()
        )
    elif float(duration_hazard_rate) > 0.0:
        h.update(np.asarray([float(duration_hazard_rate)], dtype="<f8").tobytes())
    return h.hexdigest()
