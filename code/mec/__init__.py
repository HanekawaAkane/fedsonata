"""FedHARP-MEC independent modules.

These modules implement the MEC compute/communication/dropout/LoRA-block
staleness closed loop described in docs/SYSTEM_MODEL.md. They are pure
Python/numpy with no dependency on the training code, so they can be unit
tested on synthetic inputs before being wired into main.py.

All energy reported through these modules is *model-based simulated energy*:
parameters are taken from device/RF literature and are clearly annotated as
such. Nothing here is a measurement on a real mobile edge device.
"""

from .profiles import (
    DeviceProfile,
    PowerProfile,
    ChannelProfile,
    DropoutProfile,
    ProfileSet,
    build_default_profiles,
    dbm_to_watt,
)
from .system_model import (
    compute_cycles,
    compute_time_energy,
    uplink_payload_bits,
    shannon_rate,
    uplink_latency_energy,
    arrival_indicator,
    model_client,
    set_compute_costs,
)
from .traces import (
    TraceGenerator,
    generate_channel_trace,
    generate_dropout_trace,
    compute_trace_hash,
)
from .resource_solver import (
    ResourcePlan,
    solve_bandwidth_power,
    solve_deadline,
    solve_round,
)
from .optimal_resource_solver import (
    FixedDeadlineSolution,
    solve_convex_bandwidth_power,
    solve_convex_round,
)
from .cora_preallocation import (
    REFERENCE_RATE_SOURCE,
    CoraRiftPreallocation,
    rift_payload_upper_bytes,
    solve_rift_cora_preallocation,
)
from .scheduler import (
    selection_utility,
    block_score,
    reliability_aware_selection,
    StalenessState,
    SchedulerConfig,
    est_success_default,
)
from .staleness import (
    EffectiveStaleness,
    update_effective_staleness,
    lpc_trigger_mask,
)
from .metrics import MecMetrics, make_round_summary
from .early_stopping import StopDecision, ValidationEarlyStopper
from .error_feedback import FactorResidualState
from .fedsonata_text_mec import (
    FedSonataTextMecConfig,
    FedSonataTextMecController,
    FedSonataTextMecRound,
)

__all__ = [
    "DeviceProfile",
    "PowerProfile",
    "ChannelProfile",
    "DropoutProfile",
    "ProfileSet",
    "build_default_profiles",
    "compute_cycles",
    "compute_time_energy",
    "uplink_payload_bits",
    "shannon_rate",
    "uplink_latency_energy",
    "arrival_indicator",
    "TraceGenerator",
    "generate_channel_trace",
    "generate_dropout_trace",
    "compute_trace_hash",
    "ResourcePlan",
    "solve_bandwidth_power",
    "solve_deadline",
    "solve_round",
    "FixedDeadlineSolution",
    "solve_convex_bandwidth_power",
    "solve_convex_round",
    "REFERENCE_RATE_SOURCE",
    "CoraRiftPreallocation",
    "rift_payload_upper_bytes",
    "solve_rift_cora_preallocation",
    "selection_utility",
    "block_score",
    "reliability_aware_selection",
    "StalenessState",
    "SchedulerConfig",
    "est_success_default",
    "EffectiveStaleness",
    "update_effective_staleness",
    "lpc_trigger_mask",
    "MecMetrics",
    "make_round_summary",
    "StopDecision",
    "ValidationEarlyStopper",
    "FactorResidualState",
    "FedSonataTextMecConfig",
    "FedSonataTextMecController",
    "FedSonataTextMecRound",
]
