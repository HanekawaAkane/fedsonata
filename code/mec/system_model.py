"""MEC system model: compute, payload, rate, latency, energy, arrival.

Pure functions implementing docs/SYSTEM_MODEL.md sec.2-4. No training-code
dependency. All inputs/outputs use explicit units:

  cycles   : dimensionless count
  seconds  : seconds
  joules   : Joule
  bytes    : bytes (bits = bytes * 8 exactly)
  hz       : Hertz
  watt     : Watt
  dbm      : dBm

Explicit infeasible conditions raise ValueError rather than returning NaN:
  - b <= 0  (zero allocated bandwidth)
  - p <= 0  (zero transmit power)
  - deadline <= compute_time (no time left for upload)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Set

from .profiles import DeviceProfile, PowerProfile, ChannelProfile, dbm_to_watt


# --------------------------------------------------------------------------
# Compute model (SYSTEM_MODEL sec.2)
# --------------------------------------------------------------------------
def compute_cycles(
    n_samples: int,
    c0: float,
    active_blocks: int,
    c_lora: float,
    lpc_blocks: int = 0,
) -> float:
    """Total CPU cycles for one client in one round.

        G = n * [c0 + active_blocks * c_lora + lpc_blocks * c_lora]

    Equal epochs folded into n_samples upstream. c_lora here bundles the
    LoRA backward+optimisation cost for one block; LPC correction reuses the
    same per-block B cost.
    """
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    if c0 < 0 or c_lora < 0:
        raise ValueError("cycle costs must be non-negative")
    return float(n_samples) * (c0 + active_blocks * c_lora + lpc_blocks * c_lora)


def compute_time_energy(
    cycles: float,
    device: DeviceProfile,
) -> tuple[float, float]:
    """Compute time (s) and energy (J) from cycles.

        T_cmp = G / f
        E_cmp = kappa * G * f^2 + P_idle * T_cmp

    model-based simulated energy (see D006 / SYSTEM_MODEL sec.7).
    """
    if cycles < 0:
        raise ValueError("cycles must be non-negative")
    t_cmp = cycles / device.freq_hz
    e_dyn = device.kappa * cycles * (device.freq_hz ** 2)
    e_idle = device.idle_power_w * t_cmp
    return t_cmp, e_dyn + e_idle


# --------------------------------------------------------------------------
# Communication model (SYSTEM_MODEL sec.3)
# --------------------------------------------------------------------------
def uplink_payload_bits(
    active_blocks: int,
    s_meta_bits: float,
    s_block_a_bits: float,
) -> float:
    """Uplink payload in bits.

        S_up = s_meta + active_blocks * s_block_a

    FedHARP uploads only the A updates of selected blocks plus control
    metadata/sensitivity scalar.
    """
    if active_blocks < 0:
        raise ValueError("active_blocks must be non-negative")
    if s_meta_bits < 0 or s_block_a_bits < 0:
        raise ValueError("payload sizes must be non-negative")
    return s_meta_bits + active_blocks * s_block_a_bits


def shannon_rate(bandwidth_hz: float, power_w: float, h: float, n0_w_hz: float) -> float:
    """Shannon uplink rate (bits/s) for FDMA sub-band.

        R = b * log2(1 + p*h / (N0 * b))

    Raises ValueError on b<=0 or p<=0 (explicit infeasible, never NaN).
    """
    if bandwidth_hz <= 0:
        raise ValueError("bandwidth must be positive")
    if power_w <= 0:
        raise ValueError("transmit power must be positive")
    if h < 0:
        raise ValueError("channel gain must be non-negative")
    if n0_w_hz < 0:
        raise ValueError("noise PSD must be non-negative")
    sinr = (power_w * h) / (n0_w_hz * bandwidth_hz)
    return bandwidth_hz * math.log2(1.0 + sinr)


def uplink_latency_energy(
    payload_bits: float,
    bandwidth_hz: float,
    power_w: float,
    h: float,
    n0_w_hz: float,
    power: PowerProfile,
) -> tuple[float, float]:
    """Uplink latency (s) and transmit energy (J).

        T_up = S_up / R
        E_up = (p / eta_PA + P_cir) * T_up

    payload_bits must be > 0 to have a meaningful transmission; a zero payload
    still requires the control metadata, handled by caller keeping s_meta > 0.
    """
    if payload_bits < 0:
        raise ValueError("payload_bits must be non-negative")
    rate = shannon_rate(bandwidth_hz, power_w, h, n0_w_hz)
    if rate <= 0:
        raise ValueError("non-positive rate")
    t_up = payload_bits / rate
    e_tx = (power_w / power.pa_efficiency + power.circuit_power_w) * t_up
    return t_up, e_tx


# --------------------------------------------------------------------------
# Arrival / aggregation indicator (SYSTEM_MODEL sec.4)
# --------------------------------------------------------------------------
def arrival_indicator(
    t_cmp: float,
    t_up: float,
    dropped: bool,
    deadline: float,
) -> bool:
    """Aggregation indicator a_k^t = d_k^t * 1{T_cmp + T_up <= D}.

    Dropped clients never upload. Clients whose modeled completion time
    exceeds the deadline miss the round.
    """
    if t_cmp < 0 or t_up < 0:
        raise ValueError("times must be non-negative")
    if deadline <= 0:
        raise ValueError("deadline must be positive")
    if dropped:
        return False
    return (t_cmp + t_up) <= deadline


@dataclass
class ClientSystemResult:
    """Per-client modeled quantities for one round."""

    client_id: int
    active_blocks: int
    lpc_blocks: int
    cycles: float
    t_cmp_s: float
    e_cmp_j: float
    payload_bits: float
    bandwidth_hz: float
    power_w: float
    h: float
    t_up_s: float
    e_tx_j: float
    dropped: bool
    deadline_s: float
    arrived: bool
    failure_type: str  # "none", "dropout", "deadline_miss"

    def completion_time(self) -> float:
        return self.t_cmp_s + self.t_up_s


_c0_default = 1.0e6
_c_lora_default = 5.0e4


def set_compute_costs(c0: float, c_lora: float) -> None:
    """Override the model-based default per-sample cycle costs.

    These are *model-based simulated* estimates; see SYSTEM_MODEL sec.7 and
    D006. Real device profiling is a later calibration step.
    """
    global _c0_default, _c_lora_default
    _c0_default = c0
    _c_lora_default = c_lora


def model_client(
    client_id: int,
    active_blocks: int,
    lpc_blocks: int,
    n_samples: int,
    device: DeviceProfile,
    power: PowerProfile,
    h: float,
    n0_w_hz: float,
    bandwidth_hz: float,
    tx_power_w: float,
    dropout_flag: bool,
    deadline_s: float,
    s_meta_bits: float,
    s_block_a_bits: float,
    t_cmp_s: float = None,
) -> ClientSystemResult:
    """Explicit per-client modeling entry point (no hidden globals).

    If `t_cmp_s` is provided it is used directly (the caller already modeled
    compute time from a profile); otherwise it is derived from `n_samples`.
    """
    if t_cmp_s is not None:
        t_cmp = float(t_cmp_s)
        cycles = t_cmp * device.freq_hz
        e_cmp = device.kappa * cycles * (device.freq_hz ** 2) + device.idle_power_w * t_cmp
    else:
        cycles = compute_cycles(n_samples, _c0_default, active_blocks, _c_lora_default, lpc_blocks)
        t_cmp, e_cmp = compute_time_energy(cycles, device)
    payload_bits = uplink_payload_bits(active_blocks, s_meta_bits, s_block_a_bits)
    t_up, e_tx = uplink_latency_energy(
        payload_bits, bandwidth_hz, tx_power_w, h, n0_w_hz, power
    )
    arrived = arrival_indicator(t_cmp, t_up, dropout_flag, deadline_s)
    if dropout_flag:
        failure_type = "dropout"
    elif not arrived:
        failure_type = "deadline_miss"
    else:
        failure_type = "none"
    return ClientSystemResult(
        client_id=client_id,
        active_blocks=active_blocks,
        lpc_blocks=lpc_blocks,
        cycles=cycles,
        t_cmp_s=t_cmp,
        e_cmp_j=e_cmp,
        payload_bits=payload_bits,
        bandwidth_hz=bandwidth_hz,
        power_w=tx_power_w,
        h=h,
        t_up_s=t_up,
        e_tx_j=e_tx,
        dropped=dropout_flag,
        deadline_s=deadline_s,
        arrived=arrived,
        failure_type=failure_type,
    )
