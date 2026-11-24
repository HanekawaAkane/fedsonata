"""Effective LoRA-block staleness (allocation + delivery), bypass extension.

SYSTEM_MODEL sec.5 splits staleness into:
  - allocation staleness: block not selected for a client.
  - delivery staleness: block trained but its A-update did not arrive on time.

Effective staleness (drives LPC trigger / utility):
    tilde_tau = tau_alloc + omega_net * tau_net.

This module is a *bypass* extension: it does NOT modify server.staleness.
Instead it keeps its own dual counters keyed by (client, block) and reads the
allocation decision + arrival flag produced by the main loop each round.

If a_k * x_{k,l} == 1 (selected AND arrived) -> both counters reset to 0.
Otherwise the relevant counter increments by 1 according to the failure type:
  - not selected          -> allocation +1
  - selected but dropped  -> delivery +1
  - selected but deadline -> delivery +1

FS23 intervention view (2026-11-17): raw counters are ALWAYS updated from true
events. ``age_view`` only masks what consumers read through ``effective()``:
  - dual            : alloc + omega_net * delivery   (legacy / FULL)
  - allocation_only : alloc                          (m_a=1, m_n=0)
  - delivery_only   : delivery                       (m_a=0, m_n=1)
  - zero            : 0.0                            (AGE_OFF)
  - merged          : alloc + delivery               (single-counter
                      equivalent form; at omega_net=1 bit-identical to dual)

The visible composition uses the protocol delivery weight of 1.0 for the
masked views (PROTOCOL_CN "effective age 的 delivery 权重=1"); the legacy
``dual`` view keeps ``omega_net`` so default runs stay bit-identical.
``raw_effective()`` always exposes the unmasked dual value for logging.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Set, Tuple

AGE_VIEWS = ("dual", "allocation_only", "delivery_only", "zero", "merged")


@dataclass
class EffectiveStaleness:
    """Dual-counter staleness store keyed by (client_id, block_name)."""

    omega_net: float = 1.0
    age_view: str = "dual"
    alloc: Dict[Tuple[int, str], int] = field(default_factory=dict)
    net: Dict[Tuple[int, str], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.age_view not in AGE_VIEWS:
            raise ValueError(
                f"unknown age_view {self.age_view!r}; expected one of {AGE_VIEWS}"
            )

    def raw_effective(self, client_id: int, block: str) -> float:
        """Unmasked dual-counter value (alloc + omega_net * net)."""
        key = (client_id, block)
        tau_a = self.alloc.get(key, 0)
        tau_n = self.net.get(key, 0)
        return tau_a + self.omega_net * tau_n

    def visible(self, client_id: int, block: str) -> float:
        """Consumer-visible age under the configured ``age_view`` mask."""
        key = (client_id, block)
        tau_a = float(self.alloc.get(key, 0))
        tau_n = float(self.net.get(key, 0))
        v = self.age_view
        if v == "dual":
            return tau_a + self.omega_net * tau_n
        if v == "allocation_only":
            return tau_a
        if v == "delivery_only":
            return tau_n
        if v == "zero":
            return 0.0
        if v == "merged":
            return tau_a + tau_n
        raise ValueError(f"unknown age_view {v!r}")  # pragma: no cover

    def effective(self, client_id: int, block: str) -> float:
        """Consumer view used by P1 / LPC trigger / predictor / corrector.

        FS23: this is the masked view so that every existing consumer of
        ``effective()`` receives the intervention without per-site edits.
        Use ``raw_effective()`` for unmasked logging.
        """
        return self.visible(client_id, block)

    def update_block_arrival(
        self,
        client_id: int,
        block: str,
        selected: bool,
        arrived: bool,
        failure_type: str,
    ) -> None:
        """Update counters for a single (client, block).

        selected: whether the block was allocated to this client this round.
        arrived:  whether the A-update reached the server on time.
        failure_type: "none" | "dropout" | "deadline_miss".
        """
        key = (client_id, block)
        if selected and arrived:
            self.alloc[key] = 0
            self.net[key] = 0
            return
        if not selected:
            self.alloc[key] = self.alloc.get(key, 0) + 1
        else:
            # selected but not arrived -> delivery staleness
            self.net[key] = self.net.get(key, 0) + 1


@dataclass
class ReliabilityView:
    """Read-only consumer view over the raw reliability EWMA store.

    ``observed``          -> reads pass through to the raw store.
    ``constant_initial``  -> every read returns ``r_init`` (NO_RELIABILITY /
                             DELIVERY_CONTROL_OPEN arms).  Raw updates must be
                             written to the RAW store, never through this view
                             (PROTOCOL_CN: raw counters keep true updates).
    """

    raw: Mapping[int, float]
    view: str = "observed"
    r_init: float = 1.0

    def __post_init__(self) -> None:
        if self.view not in ("observed", "constant_initial"):
            raise ValueError(
                f"unknown reliability view {self.view!r}; "
                "expected 'observed' or 'constant_initial'"
            )

    def __getitem__(self, key: int) -> float:
        if self.view == "constant_initial":
            return float(self.r_init)
        return float(self.raw[key])

    def get(self, key: int, default=None):
        if self.view == "constant_initial":
            return float(self.r_init)
        return self.raw.get(key, default)

    def __contains__(self, key) -> bool:
        return key in self.raw

    def __iter__(self):
        return iter(self.raw)

    def __len__(self) -> int:
        return len(self.raw)


def update_effective_staleness(
    state: EffectiveStaleness,
    client_id: int,
    allocated: Set[str],
    all_blocks: List[str],
    arrived: bool,
    failure_type: str,
) -> None:
    """Update staleness for one client across all blocks for one round.

    Blocks not in `allocated` increment allocation staleness.
    Blocks in `allocated` but not arrived increment delivery staleness.
    """
    for block in all_blocks:
        selected = block in allocated
        state.update_block_arrival(
            client_id, block, selected, arrived, failure_type
        )


def lpc_trigger_mask(
    state: EffectiveStaleness,
    client_id: int,
    allocated: Set[str],
    b_train_every: int,
    gamma: float = 1.0,
) -> Set[str]:
    """Return the set of blocks that should trigger LPC this round.

    Mirrors the original FedHARP trigger (effective staleness == b_train_every)
    but uses the combined effective staleness so network failures also raise
    LPC demand (SYSTEM_MODEL sec.5, EXPERIMENT_PLAN RQ3). Never silently
    assumes network failures increase LPC -- that is an empirical ablation.

    FS23: reads the consumer-visible age (``state.effective``), so the
    ``age_view`` mask reaches the LPC trigger like every other consumer.
    Trigger semantics stay ``round(tau) == b_train_every`` (exact equality);
    overshoot is an explicit test case, not silently ``>=``.
    """
    mask: Set[str] = set()
    for block in allocated:
        tau = state.effective(client_id, block)
        effective_steps = int(round(tau))
        if effective_steps == b_train_every:
            mask.add(block)
    return mask
