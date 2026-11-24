"""ACK-driven factor error feedback for RIFT.

RIFT scores corrected A/B factor deltas in the gauge-invariant effective BA
product space, while this state keeps omitted or failed factor deltas until an
ACK confirms delivery.  The state is small (at most one LoRA A/B payload per
client/block) and does not materialize dense product-space matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Mapping, Tuple

import torch


TensorMap = Dict[str, torch.Tensor]


@dataclass
class FactorResidualState:
    decay: float = 1.0
    residual_a: Dict[int, TensorMap] = field(default_factory=dict)
    residual_b: Dict[int, TensorMap] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < float(self.decay) <= 1.0:
            raise ValueError("residual decay must be in (0, 1]")
        self.decay = float(self.decay)

    def corrected(
        self,
        client_id: int,
        delta_a: Mapping[str, torch.Tensor],
        delta_b: Mapping[str, torch.Tensor],
    ) -> Tuple[TensorMap, TensorMap, float]:
        """Add discounted pending factors without mutating the stored state."""
        names = sorted(set(delta_a).intersection(delta_b))
        pending_a = self.residual_a.get(int(client_id), {})
        pending_b = self.residual_b.get(int(client_id), {})
        corrected_a: TensorMap = {}
        corrected_b: TensorMap = {}
        before_sq = 0.0
        for name in names:
            da = delta_a[name]
            db = delta_b[name]
            ra = pending_a.get(name)
            rb = pending_b.get(name)
            if ra is not None:
                ra = ra.to(device=da.device, dtype=da.dtype)
                discounted_ra = self.decay * ra
                before_sq += float(torch.sum(discounted_ra.detach().float() ** 2).item())
                da = da + discounted_ra
            if rb is not None:
                rb = rb.to(device=db.device, dtype=db.dtype)
                discounted_rb = self.decay * rb
                before_sq += float(torch.sum(discounted_rb.detach().float() ** 2).item())
                db = db + discounted_rb
            corrected_a[name] = da
            corrected_b[name] = db
        return corrected_a, corrected_b, math.sqrt(max(0.0, before_sq))

    def acknowledge(
        self,
        client_id: int,
        corrected_a: Mapping[str, torch.Tensor],
        corrected_b: Mapping[str, torch.Tensor],
        choices: Mapping[str, str],
        arrived: bool,
    ) -> Dict[str, float]:
        """Clear only factors confirmed by ACK; retain omitted/failed factors."""
        cid = int(client_id)
        store_a = self.residual_a.setdefault(cid, {})
        store_b = self.residual_b.setdefault(cid, {})
        names = sorted(set(corrected_a).intersection(corrected_b))
        cleared_a = 0
        cleared_b = 0
        for name in names:
            mode = str(choices.get(name, "none")).lower()
            ack_a = bool(arrived and mode in {"a", "ab"})
            ack_b = bool(arrived and mode in {"b", "ab"})
            if ack_a:
                store_a.pop(name, None)
                cleared_a += 1
            else:
                store_a[name] = corrected_a[name].detach().clone()
            if ack_b:
                store_b.pop(name, None)
                cleared_b += 1
            else:
                store_b[name] = corrected_b[name].detach().clone()
        if not store_a:
            self.residual_a.pop(cid, None)
        if not store_b:
            self.residual_b.pop(cid, None)
        return {
            "cleared_a": float(cleared_a),
            "cleared_b": float(cleared_b),
            "residual_l2": self.total_l2(),
            "residual_bytes": float(self.storage_bytes()),
            "residual_factors": float(self.nonzero_factors()),
        }

    def total_l2(self) -> float:
        total = 0.0
        for clients in (self.residual_a, self.residual_b):
            for factors in clients.values():
                for tensor in factors.values():
                    total += float(torch.sum(tensor.detach().float() ** 2).item())
        return math.sqrt(max(0.0, total))

    def storage_bytes(self) -> int:
        total = 0
        for clients in (self.residual_a, self.residual_b):
            for factors in clients.values():
                for tensor in factors.values():
                    total += int(tensor.numel()) * int(tensor.element_size())
        return total

    def nonzero_factors(self) -> int:
        return sum(
            len(factors)
            for clients in (self.residual_a, self.residual_b)
            for factors in clients.values()
        )
