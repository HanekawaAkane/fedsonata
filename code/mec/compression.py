"""Gradient-compression baselines for the D053 true-byte-budget comparison.

Implements the three compression families the v2 spec (T4) requires as
baselines against FedSONATA:

  * Top-k sparsification                 (k in {10%, 25%} of coordinates)
  * STC  (sparsify + ternary + error feedback)   (Sattler et al., 2019)
  * QSGD (uniform quantisation to ``levels``)    (Alistarh et al., 2017)

All three are *stateful* through an error-feedback residual so the compressed
updates are unbiased in the long run, which is what makes them fair
convergence baselines. The bit accounting is exposed explicitly so the D053
campaign can match a **true-byte budget** against FedSONATA instead of a
top-level sparsity ratio.

Everything operates on a single flat ``torch.Tensor``; dict helpers apply a
compressor per tensor (e.g. each LoRA factor).
"""

from __future__ import annotations

import math

import torch


def _num_bits_from_values(values: torch.Tensor) -> int:
    """Payload bits for a sparse (value, index) encoding."""
    if values.numel() == 0:
        return 0
    # 32 bits per value + 32 bits per index (int32 position).
    return int(values.numel()) * 64


def topk_compress(
    tensor: torch.Tensor,
    ratio: float,
    residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Top-k sparsification with error feedback.

    Returns (dense_decompressed_update, payload_bits).  ``ratio`` is the kept
    fraction in (0, 1].  ``residual`` carries the accumulated compression error
    from previous rounds; the returned update already includes it.
    """
    if not 0.0 < float(ratio) <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    flat = tensor.detach().reshape(-1)
    if residual is not None:
        flat = flat + residual.detach().reshape(-1)
    k = max(1, int(math.ceil(flat.numel() * float(ratio))))
    if k >= flat.numel():
        out = flat.clone()
        return out.reshape(tensor.shape), int(flat.numel()) * 32
    values, indices = torch.topk(flat.abs(), k)
    sparse = torch.zeros_like(flat)
    sparse[indices] = flat[indices]
    new_residual = flat - sparse
    return sparse.reshape(tensor.shape), _num_bits_from_values(values)


def stc_compress(
    tensor: torch.Tensor,
    ratio: float,
    residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Sparse ternary compression (STC) with error feedback.

    Keeps the top-``ratio`` magnitude coordinates, then quantises their values
    to {-1, 0, +1} * (mean magnitude of the kept set), matching the STC paper.
    """
    if not 0.0 < float(ratio) <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    flat = tensor.detach().reshape(-1)
    if residual is not None:
        flat = flat + residual.detach().reshape(-1)
    k = max(1, int(math.ceil(flat.numel() * float(ratio))))
    k = min(k, flat.numel())
    values, indices = torch.topk(flat.abs(), k)
    scale = float(values.mean().item()) if values.numel() > 0 else 0.0
    sparse = torch.zeros_like(flat)
    if scale > 0.0:
        kept = flat[indices]
        ternary = torch.sign(kept) * scale
        sparse[indices] = ternary
    new_residual = flat - sparse
    # sign + shared scale (32 bits) is the STC payload per kept coordinate.
    payload_bits = int(k) + 32
    return sparse.reshape(tensor.shape), payload_bits


def qsgd_compress(
    tensor: torch.Tensor,
    levels: int,
    residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """QSGD uniform quantisation with error feedback.

    ``levels`` is the number of quantisation levels per coordinate (s >= 2).
    The update is scaled by its L2 norm and rounded to the nearest of ``levels``
    bins; the (unbiased) norm scaling matches QSGD's encoding cost of
    ``log2(levels) + 32`` bits per coordinate.
    """
    if int(levels) < 2:
        raise ValueError("levels must be >= 2")
    s = int(levels)
    flat = tensor.detach().reshape(-1)
    if residual is not None:
        flat = flat + residual.detach().reshape(-1)
    norm = float(torch.norm(flat).item())
    if norm == 0.0:
        return torch.zeros_like(flat).reshape(tensor.shape), 0
    unit = flat / norm
    # Map [-1, 1] to [0, s-1] and round to the nearest integer level.
    scaled = (unit + 1.0) * 0.5 * (s - 1)
    quantized = torch.round(scaled)
    # Represent {-1, 0, +1} bins with a two-sided grid, then rescale by norm.
    dequantized = (quantized / (s - 1)) * 2.0 - 1.0
    dequantized = dequantized * norm
    new_residual = flat - dequantized
    # QSGD payload: ceil(log2(s)) bits per coordinate, plus ONE 32-bit norm per
    # tensor (not per coordinate).
    bits_per_coord = int(math.ceil(math.log2(s)))
    payload_bits = int(flat.numel()) * bits_per_coord + 32
    return dequantized.reshape(tensor.shape), payload_bits


COMPRESSORS = {
    "topk": topk_compress,
    "stc": stc_compress,
    "qsgd": qsgd_compress,
}


def compress_dict(
    tensors: dict[str, torch.Tensor],
    method: str,
    *,
    ratio: float | None = None,
    levels: int | None = None,
    residual: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], int, dict[str, torch.Tensor]]:
    """Apply a compressor per tensor and return (updates, bits, new_residual)."""
    if method not in COMPRESSORS:
        raise ValueError(f"unknown compression method: {method}")
    out: dict[str, torch.Tensor] = {}
    new_residual: dict[str, torch.Tensor] = {}
    total_bits = 0
    for name, tensor in tensors.items():
        res = None if residual is None else residual.get(name)
        if method == "topk":
            compressed, bits = topk_compress(tensor, ratio, res)
        elif method == "stc":
            compressed, bits = stc_compress(tensor, ratio, res)
        else:
            compressed, bits = qsgd_compress(tensor, levels, res)
        out[name] = compressed
        new_residual[name] = tensor.detach() - compressed + (
            torch.zeros_like(tensor) if res is None else res.detach()
        )
        total_bits += bits
    return out, total_bits, new_residual


def bit_budget(byte_budget: float) -> int:
    """Convert a true-byte budget into a bit budget."""
    return int(round(float(byte_budget) * 8.0))
