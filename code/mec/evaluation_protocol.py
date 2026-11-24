"""Leakage-safe split helpers for validation-selected experiments."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence


def split_client_indices(
    indices: Iterable[int],
    ratios: Sequence[float],
    names: Sequence[str],
) -> Dict[str, List[int]]:
    """Partition an ordered client index list without overlap or loss."""
    values = [int(x) for x in indices]
    ratios = [float(x) for x in ratios]
    names = [str(x) for x in names]
    if len(ratios) not in {2, 3} or len(names) != len(ratios):
        raise ValueError("expected train/test or train/validation/test")
    if any(x < 0.0 for x in ratios) or sum(ratios) <= 0.0:
        raise ValueError("split ratios must be non-negative with a positive sum")
    total_ratio = float(sum(ratios))
    normalized = [x / total_ratio for x in ratios]
    boundaries = []
    cumulative = 0.0
    for ratio in normalized[:-1]:
        cumulative += ratio
        boundaries.append(int(len(values) * cumulative))
    cuts = [0] + boundaries + [len(values)]
    result = {
        name: values[cuts[i]:cuts[i + 1]] for i, name in enumerate(names)
    }
    flattened = [x for name in names for x in result[name]]
    if len(flattened) != len(values) or len(set(flattened)) != len(values):
        raise ValueError("client split must be disjoint and exhaustive")
    return result
