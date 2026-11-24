"""Device, power, channel and dropout profiles for the MEC system model.

All numeric parameters are taken from published device/RF literature and are
clearly marked. Energy computed elsewhere from these parameters is
*model-based simulated energy*, NOT a measurement on a real edge device.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DeviceProfile:
    """Per-device compute profile.

    Attributes:
        freq_hz: clock frequency in Hz (observed profile, not an optimisation
            variable in version 1 -- see D002 / SYSTEM_MODEL sec.2).
        kappa: effective switched-capacitance coefficient for dynamic energy,
            E_dyn = kappa * cycles * f^2 (Joule). Source: generic CMOS energy
            model E ~ C V^2 ; V ~ f, so E_dyn ~ kappa * cycles * f^2.
            Default 1e-27 J*s^2/cycle is a literature order-of-magnitude value
            for mobile SoCs and is explicitly a model-based estimate.
        idle_power_w: idle/circuit power while computing (Watt).
    """

    freq_hz: float = 1.5e9
    kappa: float = 1.0e-27
    idle_power_w: float = 0.5

    def __post_init__(self) -> None:
        if self.freq_hz <= 0:
            raise ValueError("freq_hz must be positive")
        if self.kappa < 0:
            raise ValueError("kappa must be non-negative")
        if self.idle_power_w < 0:
            raise ValueError("idle_power_w must be non-negative")


@dataclass
class PowerProfile:
    """RF transmit power and circuit profile for one client.

    Attributes:
        p_max_dbm: maximum transmit power (dBm). System default 23 dBm per
            experiment_matrix.json. Baselines MUST use the same p_max and must
            not silently exceed it.
        pa_efficiency: power amplifier efficiency eta in (0, 1]; PA dc draw =
            p_tx / eta.
        circuit_power_w: RF circuit power (Watt) consumed during transmission
            (LNA/mixer/oscillator etc.).
    """

    p_max_dbm: float = 23.0
    pa_efficiency: float = 0.35
    circuit_power_w: float = 0.1

    def __post_init__(self) -> None:
        if not -40.0 <= self.p_max_dbm <= 40.0:
            raise ValueError("p_max_dbm out of plausible range")
        if not 0.0 < self.pa_efficiency <= 1.0:
            raise ValueError("pa_efficiency must be in (0, 1]")
        if self.circuit_power_w < 0:
            raise ValueError("circuit_power_w must be non-negative")

    @property
    def p_max_w(self) -> float:
        return dbm_to_watt(self.p_max_dbm)


@dataclass
class ChannelProfile:
    """Channel model configuration.

    Attributes:
        model: one of "constant", "rayleigh", "correlated".
        noise_spectral_density_dbm_hz: N0 in dBm/Hz. Default -174 dBm/Hz
            (thermal noise at 290 K).
        constant_gain: channel power gain h used when model == "constant".
        rayleigh_mean: mean of the exponential |h|^2 distribution for i.i.d.
            Rayleigh fading (E[|h|^2] = rayleigh_mean).
        correlation: AR(1) coefficient for temporally correlated fading
            (used when model == "correlated").
    """

    model: str = "rayleigh"
    noise_spectral_density_dbm_hz: float = -174.0
    constant_gain: float = 1.0
    rayleigh_mean: float = 1.0
    correlation: float = 0.8

    def __post_init__(self) -> None:
        if self.model not in ("constant", "rayleigh", "correlated"):
            raise ValueError(f"unknown channel model: {self.model}")
        if self.rayleigh_mean <= 0:
            raise ValueError("rayleigh_mean must be positive")
        if not 0.0 <= self.correlation < 1.0:
            raise ValueError("correlation must be in [0, 1)")

    @property
    def n0_w_hz(self) -> float:
        return dbm_to_watt(self.noise_spectral_density_dbm_hz)


@dataclass
class DropoutProfile:
    """Exogenous dropout (battery / process / link interruption) profile.

    Attributes:
        model: "none", "bernoulli" or "gilbert_elliott".
        dropout_rate: target average dropout probability (bernoulli) or the
            average dropout rate the two-state Gilbert-Elliott chain is tuned
            to (so different models are comparable at the same mean).
        ge_good_to_bad: Gilbert-Elliott transition prob from good->bad state.
        ge_bad_to_good: Gilbert-Elliott transition prob from bad->good state.
    """

    model: str = "none"
    dropout_rate: float = 0.0
    ge_good_to_bad: float = 0.1
    ge_bad_to_good: float = 0.2

    def __post_init__(self) -> None:
        if self.model not in ("none", "bernoulli", "gilbert_elliott"):
            raise ValueError(f"unknown dropout model: {self.model}")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1)")
        if not 0.0 <= self.ge_good_to_bad <= 1.0:
            raise ValueError("ge_good_to_bad must be in [0, 1]")
        if not 0.0 <= self.ge_bad_to_good <= 1.0:
            raise ValueError("ge_bad_to_good must be in [0, 1]")
        if self.model == "gilbert_elliott":
            # Fix the steady-state bad probability to equal dropout_rate.  For a
            # two-state chain the steady-state bad prob is p / (p + q) with
            # p = good->bad and q = bad->good.  Keep the caller's good->bad rate
            # as the anchor (falling back to a positive default) and re-tune q so
            # the steady state exactly matches dropout_rate.  This mirrors the
            # validated logic in main.py / fedhello_mec.py and removes the former
            # hardcoded 0.5 anchor that produced q > 1 (an absorbing bad state)
            # whenever dropout_rate < 1/3.
            r = self.dropout_rate
            if dropout_rate_close(r, 0.0):
                self.ge_good_to_bad = 0.0
                self.ge_bad_to_good = 1.0
            else:
                p = self.ge_good_to_bad
                if p <= 0.0:
                    p = 0.5  # a zero good->bad rate can never reach r > 0
                q = p * (1.0 - r) / r
                if not 0.0 < q <= 1.0:
                    raise ValueError(
                        "Gilbert-Elliott parameters infeasible: choose a smaller "
                        "good->bad probability"
                    )
                self.ge_good_to_bad = p
                self.ge_bad_to_good = q


@dataclass
class ProfileSet:
    """Container for all client profiles + shared RF/channel config."""

    device_profiles: List[DeviceProfile] = field(default_factory=list)
    power_profile: PowerProfile = field(default_factory=PowerProfile)
    channel: ChannelProfile = field(default_factory=ChannelProfile)
    dropout: DropoutProfile = field(default_factory=DropoutProfile)
    total_bandwidth_hz: float = 20e6
    # Per-block LoRA compute/communication cost (shared across clients unless
    # overridden). These are *model-based* estimates; see system_model.py.
    c0_cycles_per_sample: float = 1.0e6
    c_lora_cycles_per_sample: float = 5.0e4
    s_meta_bits: float = 2.0e3
    s_block_a_bits: float = 8.0e4

    def __post_init__(self) -> None:
        if self.total_bandwidth_hz <= 0:
            raise ValueError("total_bandwidth_hz must be positive")
        if len(self.device_profiles) == 0:
            raise ValueError("device_profiles must not be empty")

    def num_clients(self) -> int:
        return len(self.device_profiles)

    def device(self, k: int) -> DeviceProfile:
        return self.device_profiles[k]


def dbm_to_watt(dbm: float) -> float:
    """Convert dBm to Watt. Exact: 10^(dbm/10) / 1000."""
    return 10.0 ** (dbm / 10.0) / 1000.0


def watt_to_dbm(watt: float) -> float:
    if watt <= 0:
        raise ValueError("watt must be positive")
    return 10.0 * math.log10(watt * 1000.0)


def dropout_rate_close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def build_default_profiles(
    num_clients: int,
    heterogeneity_type: str = "6-3-1",
    base_ratio: float = 0.5,
    total_bandwidth_mhz: float = 20.0,
    power_max_dbm: float = 23.0,
    channel_model: str = "rayleigh",
    dropout_model: str = "bernoulli",
    dropout_rate: float = 0.4,
) -> ProfileSet:
    """Build a ProfileSet with device heterogeneity matching main.py.

    Device frequencies are drawn across three tiers to emulate the
    high/mid/low resource split used by the '6-3-1' heterogeneity type.
    These frequencies are *observed profiles*, not an optimisation variable.
    """
    if heterogeneity_type == "uniform":
        ratios = [base_ratio] * num_clients
    elif heterogeneity_type == "1-1-1":
        n_low = int(0.33 * num_clients)
        n_mid = int(0.33 * num_clients)
        n_high = num_clients - n_low - n_mid
        ratios = (
            [base_ratio] * n_low
            + [min(1.0, base_ratio * 1.5)] * n_mid
            + [1.0] * n_high
        )
    elif heterogeneity_type == "6-3-1":
        n_low = int(0.6 * num_clients)
        n_mid = int(0.3 * num_clients)
        n_high = num_clients - n_low - n_mid
        ratios = (
            [base_ratio] * n_low
            + [min(1.0, base_ratio * 1.5)] * n_mid
            + [1.0] * n_high
        )
    else:
        ratios = [base_ratio] * num_clients

    # Map resource ratio to a representative device frequency (model-based).
    # Low/mid/high tiers ~ 1.0/1.5/2.0 GHz typical mobile/edge SoC bins.
    device_profiles: List[DeviceProfile] = []
    for r in ratios:
        if r <= base_ratio + 1e-9:
            freq = 1.0e9
        elif r < 1.0 - 1e-9:
            freq = 1.5e9
        else:
            freq = 2.0e9
        device_profiles.append(
            DeviceProfile(freq_hz=freq, kappa=1.0e-27, idle_power_w=0.5)
        )

    power = PowerProfile(p_max_dbm=power_max_dbm, pa_efficiency=0.35, circuit_power_w=0.1)
    channel = ChannelProfile(model=channel_model, noise_spectral_density_dbm_hz=-174.0)
    dropout = DropoutProfile(model=dropout_model, dropout_rate=dropout_rate)

    return ProfileSet(
        device_profiles=device_profiles,
        power_profile=power,
        channel=channel,
        dropout=dropout,
        total_bandwidth_hz=total_bandwidth_mhz * 1e6,
    )
