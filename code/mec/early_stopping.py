"""Validation-only early stopping for versioned MEC experiments.

The stopping decision must never consume the official test metric.  Callers
pass a client-holdout validation score and keep test accuracy reporting-only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


@dataclass(frozen=True)
class StopDecision:
    stop: bool
    improved: bool
    round_num: int
    best_round: int
    best_score: float
    bad_rounds: int
    reason: Optional[str]


class ValidationEarlyStopper:
    """Maximise a validation metric with a declared minimum horizon."""

    def __init__(self, min_rounds: int, max_rounds: int, patience: int, min_delta: float):
        if min_rounds < 1:
            raise ValueError("min_rounds must be >= 1")
        if max_rounds < min_rounds:
            raise ValueError("max_rounds must be >= min_rounds")
        if patience < 1:
            raise ValueError("patience must be >= 1")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        self.min_rounds = int(min_rounds)
        self.max_rounds = int(max_rounds)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_score = float("-inf")
        self.best_round = 0
        self.bad_rounds = 0
        self.last_round = 0

    def update(self, round_num: int, validation_score: float) -> StopDecision:
        if round_num != self.last_round + 1:
            raise ValueError("round_num must be consecutive and start at 1")
        if not math.isfinite(validation_score):
            raise ValueError("validation_score must be finite")

        improved = self.best_round == 0 or validation_score >= self.best_score + self.min_delta
        if improved:
            self.best_score = float(validation_score)
            self.best_round = int(round_num)
            self.bad_rounds = 0
        else:
            self.bad_rounds += 1
        self.last_round = int(round_num)

        reason = None
        stop = False
        if round_num >= self.max_rounds:
            stop = True
            reason = "max_rounds"
        elif round_num >= self.min_rounds and self.bad_rounds >= self.patience:
            stop = True
            reason = "validation_plateau"

        return StopDecision(
            stop=stop,
            improved=improved,
            round_num=int(round_num),
            best_round=self.best_round,
            best_score=self.best_score,
            bad_rounds=self.bad_rounds,
            reason=reason,
        )
