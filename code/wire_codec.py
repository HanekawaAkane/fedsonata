"""Actual byte codecs for a bounded, allocation-replayed LoRA experiment.

The model consumes decoded messages. QSGD uses stochastic norm quantization;
STC transmits explicit uint32 coordinates, packed signs, and a float32 scale.
This is a declared transport adaptation, not the official STC entropy coder.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
import struct
import zlib
from dataclasses import replace

import numpy as np
import torch

from mec.factor_selector import RiftSelection, select_rift_updates
from mec.loss_aware_factor_selector import select_loss_aware_updates
from mec.symmetric_lookahead_selector import select_symmetric_lookahead_updates
from mec.edge_random_factor_policy import (
    EDGE_PRIOR_FLOOR as EDGE_PRIOR_FLOOR_V2,
    EDGE_PRIOR_TAU as EDGE_PRIOR_TAU_V2,
    sample_edge_weighted_atoms as sample_edge_weighted_atoms_v2,
)

MAGIC = b"FSC1"

EDGE_PRIOR_TAU = 0.22
EDGE_PRIOR_FLOOR = 0.25


def edge_prior_weights(names, tau=EDGE_PRIOR_TAU, floor=EDGE_PRIOR_FLOOR):
    """Return normalized U-shaped depth weights for LoRA blocks.

    The depth is extracted from common transformer names such as
    ``layers.0``/``layer.11``.  If a backend uses another naming scheme, the
    sorted block order is used deterministically.  Both boundaries receive
    higher probability than the interior, with no data-dependent tuning.
    """
    names = sorted(str(name) for name in names)
    if not names:
        return {}
    depth = {}
    for ordinal, name in enumerate(names):
        match = re.search(r"(?:layers?|blocks?|h)\.(\d+)", name.lower())
        depth[name] = int(match.group(1)) if match else ordinal
    lo, hi = min(depth.values()), max(depth.values())
    span = float(max(1, hi - lo))
    tau = max(1e-6, float(tau))
    raw = {}
    for name in names:
        position = (float(depth[name]) - float(lo)) / span
        edge = 0.5 * (
            math.exp(-position / tau) + math.exp(-(1.0 - position) / tau)
        )
        raw[name] = float(floor) + edge
    mean_weight = sum(raw.values()) / float(len(raw))
    return {name: value / max(1e-12, mean_weight) for name, value in raw.items()}


def sample_edge_weighted_atoms(
    *, delta_a, delta_b, budget_bytes, header_reserve, quantum_bytes,
    seed, client_id, round_num,
):
    """Sample A/B atoms without replacement under the true quantized cap."""
    names = sorted(set(delta_a).intersection(delta_b))
    weights = edge_prior_weights(names)
    rng = np.random.default_rng(
        int(seed) * 1000003 + int(client_id) * 997 + int(round_num) * 7919 + 104729
    )
    remaining = max(0, (int(budget_bytes) - int(header_reserve)) // int(quantum_bytes))
    atoms = []
    for factor, tensors in (("A", delta_a), ("B", delta_b)):
        for name in names:
            tensor = tensors[name]
            units = int(math.ceil(
                float(tensor.numel() * tensor.element_size()) / float(quantum_bytes)
            ))
            atoms.append((factor, name, units, float(weights[name])))
    selected_a, selected_b = {}, {}
    while atoms:
        feasible = [atom for atom in atoms if atom[2] <= remaining]
        if not feasible:
            break
        probabilities = np.asarray([max(1e-12, atom[3]) for atom in feasible], dtype=np.float64)
        probabilities /= probabilities.sum()
        index = int(rng.choice(len(feasible), p=probabilities))
        factor, name, units, _weight = feasible[index]
        tensor = delta_a[name] if factor == "A" else delta_b[name]
        (selected_a if factor == "A" else selected_b)[name] = tensor
        remaining -= units
        atoms.remove((factor, name, units, _weight))
    choices = {
        name: (
            "ab" if name in selected_a and name in selected_b
            else "a" if name in selected_a
            else "b" if name in selected_b else "none"
        )
        for name in names
    }
    return selected_a, selected_b, choices, weights


def _pack_bits(values, width):
    values = np.asarray(values, dtype=np.uint32)
    bits = ((values[:, None] >> np.arange(width, dtype=np.uint32)) & 1).astype(np.uint8)
    return np.packbits(bits.ravel(), bitorder="little").tobytes()


def _unpack_bits(blob, n, width):
    bits = np.unpackbits(np.frombuffer(blob, dtype=np.uint8), bitorder="little")[:n * width]
    if len(bits) != n * width:
        raise ValueError("Truncated packed values")
    return (bits.reshape(n, width).astype(np.uint32) * (1 << np.arange(width, dtype=np.uint32))).sum(axis=1)


def _frame(header, payload):
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    body = MAGIC + struct.pack("<I", len(encoded)) + encoded + payload
    return body + struct.pack("<I", zlib.crc32(body))


def _parse(blob):
    if len(blob) < 12 or blob[:4] != MAGIC:
        raise ValueError("Invalid codec frame")
    if zlib.crc32(blob[:-4]) != struct.unpack("<I", blob[-4:])[0]:
        raise ValueError("CRC mismatch")
    size = struct.unpack("<I", blob[4:8])[0]
    return json.loads(blob[8:8 + size]), blob[8 + size:-4]


def _flatten(a, b):
    items, vectors = [], []
    for factor, tensors in (("A", a), ("B", b)):
        for name in sorted(tensors):
            tensor = tensors[name]
            if tensor.dtype != torch.float32:
                raise ValueError("This declared codec protocol requires FP32 LoRA factors")
            items.append([factor, name, list(tensor.shape)])
            vectors.append(tensor.detach().cpu().numpy().reshape(-1).astype("<f4", copy=False))
    vector = np.concatenate(vectors) if vectors else np.zeros(0, dtype="<f4")
    if not np.all(np.isfinite(vector)):
        raise ValueError("Nonfinite local increment")
    return items, vector


def decode_message(blob, templates_a, templates_b):
    header, payload = _parse(blob)
    n = int(header["n"])
    codec = header["codec"]
    if codec == "raw":
        vector = np.frombuffer(payload, dtype="<f4").copy()
        if len(vector) != n:
            raise ValueError("Invalid raw length")
    elif codec in {"topk", "stc"}:
        k = int(header["k"])
        indices = np.frombuffer(payload[:4 * k], dtype="<u4")
        if len(indices) != k or np.any(indices >= n) or len(np.unique(indices)) != k:
            raise ValueError("Invalid sparse coordinates")
        vector = np.zeros(n, dtype=np.float32)
        if codec == "topk":
            values = np.frombuffer(payload[4 * k:], dtype="<f4")
            if len(values) != k:
                raise ValueError("Invalid top-k values")
        else:
            sign_bytes = (k + 7) // 8
            signs = _unpack_bits(payload[4 * k:4 * k + sign_bytes], k, 1)
            tail = payload[4 * k + sign_bytes:]
            if len(tail) != 4:
                raise ValueError("Invalid STC scale")
            scale = struct.unpack("<f", tail)[0]
            values = (2 * signs.astype(np.float32) - 1) * scale
        vector[indices] = values
    elif codec == "qsgd":
        width = int(header["width"])
        packed_bytes = (n * width + 7) // 8
        codes = _unpack_bits(payload[:packed_bytes], n, width)
        norm = struct.unpack("<f", payload[packed_bytes:])[0]
        sign = (codes >> (width - 1)).astype(np.float32)
        magnitude = codes & ((1 << (width - 1)) - 1)
        vector = (2 * sign - 1) * magnitude.astype(np.float32) * (norm / int(header["levels"]))
    else:
        raise ValueError("Unknown codec")
    decoded_a, decoded_b = {}, {}
    offset = 0
    for factor, name, shape in header["items"]:
        count = int(np.prod(shape))
        templates = templates_a if factor == "A" else templates_b
        template = templates[name]
        tensor = torch.from_numpy(vector[offset:offset + count].copy()).reshape(shape)
        tensor = tensor.to(device=template.device, dtype=template.dtype)
        (decoded_a if factor == "A" else decoded_b)[name] = tensor
        offset += count
    if offset != n:
        raise ValueError("Invalid shape metadata")
    return decoded_a, decoded_b, header


def encode_raw(a, b):
    items, flat = _flatten(a, b)
    return _frame({"codec": "raw", "items": items, "n": len(flat)}, flat.astype("<f4").tobytes())


def encode_coordinates(a, b, codec, budget, rng):
    items, flat = _flatten(a, b)
    n = len(flat)
    base = {"codec": codec, "items": items, "n": n}
    if codec in {"topk", "stc"}:
        def size(k):
            header = dict(base, k=k)
            payload_size = 8 * k if codec == "topk" else 4 * k + (k + 7) // 8 + 4
            return len(_frame(header, b"")) + payload_size
        low, high = 0, n
        while low < high:
            mid = (low + high + 1) // 2
            if size(mid) <= budget:
                low = mid
            else:
                high = mid - 1
        k = low if np.any(flat) else 0
        if size(k) > budget:
            raise ValueError("Budget smaller than codec metadata")
        indices = np.sort(np.argpartition(np.abs(flat), n - k)[n - k:]) if k else np.zeros(0, dtype=np.int64)
        selected = flat[indices]
        payload = indices.astype("<u4").tobytes()
        if codec == "topk":
            payload += selected.astype("<f4").tobytes()
        else:
            scale = float(np.mean(np.abs(selected))) if k else 0.0
            payload += _pack_bits((selected >= 0).astype(np.uint32), 1)
            payload += struct.pack("<f", scale)
        blob = _frame(dict(base, k=k), payload)
    elif codec == "qsgd":
        width = None
        for candidate in range(16, 1, -1):
            levels = (1 << (candidate - 1)) - 1
            header = dict(base, width=candidate, levels=levels)
            if len(_frame(header, b"")) + (n * candidate + 7) // 8 + 4 <= budget:
                width = candidate
                break
        if width is None:
            raise ValueError("Budget cannot hold sign plus one stochastic magnitude bit per coordinate")
        levels = (1 << (width - 1)) - 1
        norm = float(np.linalg.norm(flat.astype(np.float64)))
        if not math.isfinite(norm):
            raise ValueError("Nonfinite QSGD norm")
        if norm:
            scaled = np.minimum(levels, np.abs(flat.astype(np.float64)) * levels / norm)
            lower = np.floor(scaled)
            magnitude = (lower + (rng.random(n) < scaled - lower)).astype(np.uint32)
        else:
            magnitude = np.zeros(n, dtype=np.uint32)
        codes = magnitude | ((flat >= 0).astype(np.uint32) << (width - 1))
        payload = _pack_bits(codes, width) + struct.pack("<f", norm)
        blob = _frame(dict(base, width=width, levels=levels), payload)
    else:
        raise ValueError(codec)
    if len(blob) > budget:
        raise AssertionError("Actual encoded byte constraint violated")
    return blob


def acknowledge_coordinates(state, cid, corrected_a, corrected_b, decoded_a, decoded_b, arrived):
    """Retain d-decoded(d) after successful delivery; retain d after failure."""
    for corrected, decoded, store in ((corrected_a, decoded_a, state.residual_a), (corrected_b, decoded_b, state.residual_b)):
        client_store = store.setdefault(int(cid), {})
        for name, increment in corrected.items():
            residual = increment - decoded.get(name, torch.zeros_like(increment)) if arrived else increment
            client_store[name] = residual.detach().clone()
    return {"cleared_a": 0, "cleared_b": 0, "residual_l2": state.total_l2(),
            "residual_bytes": state.storage_bytes(), "residual_factors": state.nonzero_factors()}


class WireController:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.mec_output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.rows = {}
        self.trace = {}
        if args.wire_replay:
            self.trace = {int(row["round"]): row for row in
                          map(json.loads, Path(args.wire_replay).read_text().splitlines())}
        self.round_record = None
        self.rng = {}
        self.factor_debt = {}
        source_hashes = {}
        for p in sorted(Path(__file__).parent.rglob("*.py")):
            if "__pycache__" not in p.parts:
                source_hashes[str(p.relative_to(Path(__file__).parent))] = hashlib.sha256(p.read_bytes()).hexdigest()
        protocol = {"version": "wire-compression-screen-v1", "codec": args.wire_codec,
                    "seed": args.seed, "source_hashes": source_hashes,
                    "allocation_budget_state_replay": args.wire_replay,
                    "replay_sha256": hashlib.sha256(Path(args.wire_replay).read_bytes()).hexdigest() if args.wire_replay else None,
                    "environment": {k: os.environ.get(k) for k in ["PYTHONHASHSEED", "RIFT_SELECTION", "RIFT_BUDGET_RELIABILITY"]},
                    "energy_model": "model-based simulated", "coordinate_residual": "d - a*decoded(d)",
                    "base_energy_accounting": "legacy vision frequency coefficient; comparisons confined to this campaign",
                    "legacy_A_byte_column": "whole coupled codec container, not factor-cost attribution"}
        if args.wire_codec == "edge_random_rift":
            protocol["edge_prior"] = {
                "type": "two-sided exponential depth prior",
                "tau": EDGE_PRIOR_TAU_V2,
                "floor": EDGE_PRIOR_FLOOR_V2,
                "sampling": "weighted without replacement under quantized raw-frame cap",
            }
        (self.out / "WIRE_PROTOCOL.json").write_text(json.dumps(protocol, indent=2))
        config_path = self.out / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            config.update(wire_codec=args.wire_codec, wire_replay_sha256=protocol["replay_sha256"],
                          wire_format_version=protocol["version"], source_sha256=source_hashes)
            config_path.write_text(json.dumps(config, indent=2))

    def before_round(self, round_num, allocation, reliability, server, staleness):
        if self.trace:
            row = self.trace[int(round_num)]
            allocation = {int(cid): set(names) for cid, names in row["allocation"].items()}
            reliability.update({int(cid): float(value) for cid, value in row["reliability"].items()})
            server.staleness = {int(cid): dict(values) for cid, values in row["server_age"].items()}
            staleness.alloc = {(int(cid), name): int(value) for cid, name, value in row["alloc_age"]}
            staleness.net = {(int(cid), name): int(value) for cid, name, value in row["net_age"]}
        self.round_record = {"round": int(round_num),
                             "allocation": {str(cid): sorted(names) for cid, names in allocation.items()},
                             "reliability": dict(reliability), "server_age": server.staleness,
                             "alloc_age": [[cid, name, v] for (cid, name), v in staleness.alloc.items()],
                             "net_age": [[cid, name, v] for (cid, name), v in staleness.net.items()],
                             "clients": {}}
        return allocation

    def select(self, *, round_num, client_id, a_after, b_after, delta_a, delta_b,
               budget_ratio, reliability, reliability_floor, **kwargs):
        full_bytes = sum(t.numel() * t.element_size() for maps in (delta_a, delta_b) for t in maps.values())
        # S3 I3 (DP-RIFT): expected-delivery pricing of the byte cap.
        _cap_signal = float(reliability)
        if str(getattr(self.args, "cap_policy", "linear")) == "expected_delivery":
            _rate = float(kwargs.get("delivery_rate_bps") or 0.0)
            _hz = float(kwargs.get("cap_hazard_rate") or 0.0)
            if _rate > 0.0 and math.isfinite(_rate) and _hz >= 0.0:
                _t_est = (full_bytes * 8.0) / _rate
                _cap_signal = float(reliability) * math.exp(-_hz * _t_est)
        cap = int(math.floor(full_bytes * budget_ratio * max(reliability_floor, _cap_signal)))
        if self.trace:
            expected = self.trace[int(round_num)]["clients"][str(client_id)]
            if cap != expected["cap_bytes"] or full_bytes != expected["full_ab_bytes"]:
                raise AssertionError("Replay failed to match allocation and true byte cap")
        codec = self.args.wire_codec
        edge_weights = {}
        if codec in {"rift", "random_rift", "edge_random_rift", "lossaware_rift", "lookahead_rift", "balanced_lookahead_rift", "counterfactual_rift"}:
            # Reserve a full-message header upper bound before the original DP.
            items, flat = _flatten(delta_a, delta_b)
            header_reserve = len(_frame({"codec": "raw", "items": items, "n": len(flat)}, b""))
            reduced_ratio = max(1e-8, (cap - header_reserve) / (full_bytes * max(reliability_floor, _cap_signal)))
            if codec == "rift":
                selection = select_rift_updates(a_after=a_after, b_after=b_after, delta_a=delta_a, delta_b=delta_b,
                                                budget_ratio=reduced_ratio, reliability=reliability,
                                                reliability_floor=reliability_floor, **kwargs)
                selected_a, selected_b = selection.updates_A, selection.updates_B
            elif codec == "edge_random_rift":
                selected_a, selected_b, choices, edge_weights = sample_edge_weighted_atoms_v2(
                    delta_a=delta_a,
                    delta_b=delta_b,
                    budget_bytes=cap,
                    header_reserve=header_reserve,
                    quantum_bytes=int(kwargs.get("quantum_bytes", 4096)),
                    seed=int(self.args.seed),
                    client_id=int(client_id),
                    round_num=int(round_num),
                )
                selection = RiftSelection(
                    updates_A=selected_a,
                    updates_B=selected_b,
                    choices=choices,
                    full_ab_bytes=full_bytes,
                    budget_bytes=cap,
                    selected_bytes=0,
                    budget_utilization=0.0,
                    captured_gain=0.0,
                    oracle_gain=0.0,
                    captured_gain_ratio=0.0,
                    reliability=reliability,
                    option_counts={
                        mode: sum(value == mode for value in choices.values())
                        for mode in ("none", "a", "b", "ab")
                    },
                )
            elif codec == "counterfactual_rift":
                score_fn = kwargs.pop("counterfactual_score", None)
                grad_w = kwargs.pop("grad_W", {}) or {}
                before_a = {name: a_after[name] - delta_a[name] for name in delta_a}
                before_b = {name: b_after[name] - delta_b[name] for name in delta_b}
                names = sorted(set(delta_a).intersection(delta_b))
                arrival = {name: {"none": 1.0, "a": float(reliability), "b": float(reliability), "ab": float(reliability)} for name in names}
                base = select_symmetric_lookahead_updates(
                    before_A=before_a,
                    before_B=before_b,
                    delta_A=delta_a,
                    delta_B=delta_b,
                    grad_W={name: grad_w.get(name, torch.zeros_like(before_b[name] @ before_a[name], dtype=torch.float32)) for name in names},
                    arrival_probability=arrival,
                    curvature={name: 0.0 for name in names},
                    budget_bytes=max(0, int(cap - header_reserve)),
                    step_size=1.0,
                    quantum_bytes=int(kwargs.get("quantum_bytes", 4096)),
                    future_weight=1.0,
                    balance_debt=kwargs.pop("factor_debt", {}),
                )
                candidates = [(base.updates_A, base.updates_B, base.choices)]
                quantum = int(kwargs.get("quantum_bytes", 4096))
                remaining_units = max(0, (cap - header_reserve) // quantum)
                for candidate_id in range(8):
                    rng = np.random.default_rng(
                        int(self.args.seed) * 1000003 + int(client_id) * 997 + candidate_id * 7919
                    )
                    atoms = [(factor, name) for factor, tensors in (("A", delta_a), ("B", delta_b)) for name in names]
                    rng.shuffle(atoms)
                    selected_a, selected_b = {}, {}
                    remaining = remaining_units
                    for factor, name in atoms:
                        units = math.ceil(
                            ((delta_a if factor == "A" else delta_b)[name].numel()
                             * (delta_a if factor == "A" else delta_b)[name].element_size()) / quantum
                        )
                        if units <= remaining:
                            (selected_a if factor == "A" else selected_b)[name] = (delta_a if factor == "A" else delta_b)[name]
                            remaining -= units
                    choices = {
                        name: "ab" if name in selected_a and name in selected_b
                        else "a" if name in selected_a else "b" if name in selected_b else "none"
                        for name in names
                    }
                    candidates.append((selected_a, selected_b, choices))
                if score_fn is not None:
                    scored = [(float(score_fn(ca, cb)), idx) for idx, (ca, cb, _choices) in enumerate(candidates)]
                    _best_score, best_idx = min(scored, key=lambda item: (item[0], item[1]))
                else:
                    best_idx = 0
                selected_a, selected_b, choices = candidates[best_idx]
                selection = RiftSelection(
                    updates_A=selected_a, updates_B=selected_b, choices=choices,
                    full_ab_bytes=full_bytes, budget_bytes=cap, selected_bytes=0,
                    budget_utilization=0.0, captured_gain=0.0, oracle_gain=0.0,
                    captured_gain_ratio=0.0, reliability=reliability,
                    option_counts={mode: sum(value == mode for value in choices.values()) for mode in ("none", "a", "b", "ab")},
                )
            elif codec in {"lossaware_rift", "lookahead_rift", "balanced_lookahead_rift"}:
                grad_w = kwargs.pop("grad_W", {}) or {}
                age_limit = int(kwargs.pop("loss_aware_age_limit", 999))
                before_a = {name: a_after[name] - delta_a[name] for name in delta_a}
                before_b = {name: b_after[name] - delta_b[name] for name in delta_b}
                rate = kwargs.get("delivery_rate_bps")
                hazard = float(kwargs.get("duration_hazard_rate", 0.0))
                arrival = {}
                for name in sorted(set(delta_a).intersection(delta_b)):
                    arrival[name] = {"none": 1.0}
                    for mode, payload in {
                        "a": int(delta_a[name].numel() * delta_a[name].element_size()),
                        "b": int(delta_b[name].numel() * delta_b[name].element_size()),
                        "ab": int(delta_a[name].numel() * delta_a[name].element_size())
                        + int(delta_b[name].numel() * delta_b[name].element_size()),
                    }.items():
                        if rate is None or float(rate) <= 0.0 or hazard <= 0.0:
                            arrival[name][mode] = float(reliability)
                        else:
                            arrival[name][mode] = float(reliability) * math.exp(
                                -hazard * 8.0 * float(payload) / float(rate)
                            )
                selector = (
                    select_symmetric_lookahead_updates
                    if codec in {"lookahead_rift", "balanced_lookahead_rift"}
                    else select_loss_aware_updates
                )
                selector_kwargs = dict(
                    before_A=before_a,
                    before_B=before_b,
                    delta_A=delta_a,
                    delta_B=delta_b,
                    grad_W={
                        name: grad_w.get(name, torch.zeros_like(before_b[name] @ before_a[name], dtype=torch.float32))
                        for name in sorted(set(delta_a).intersection(delta_b))
                    },
                    arrival_probability=arrival,
                    curvature={name: 0.0 for name in arrival},
                    allocation_age={name: 0 for name in arrival},
                    budget_bytes=max(0, int(cap - header_reserve)),
                    step_size=1.0,
                    age_limit=age_limit,
                    quantum_bytes=int(kwargs.get("quantum_bytes", 4096)),
                )
                if codec in {"lookahead_rift", "balanced_lookahead_rift"}:
                    selector_kwargs.pop("age_limit", None)
                    selector_kwargs.pop("allocation_age", None)
                    selector_kwargs.update({
                        "future_weight": 1.0,
                        "balance_debt": kwargs.pop("factor_debt", {}),
                    })
                selection = selector(**selector_kwargs)
                selected_a, selected_b = selection.updates_A, selection.updates_B
            else:
                rng = self.rng.setdefault(int(client_id), np.random.default_rng(int(self.args.seed) * 1000003 + int(client_id) * 997))
                atoms = [(factor, name) for factor, tensors in (("A", delta_a), ("B", delta_b)) for name in sorted(tensors)]
                rng.shuffle(atoms)
                quantum = int(kwargs["quantum_bytes"])
                remaining = max(0, (cap - header_reserve) // quantum)
                selected_a, selected_b, choices = {}, {}, {name: "none" for name in sorted(delta_a)}
                for factor, name in atoms:
                    tensor = (delta_a if factor == "A" else delta_b)[name]
                    units = math.ceil(tensor.numel() * tensor.element_size() / quantum)
                    if units <= remaining:
                        (selected_a if factor == "A" else selected_b)[name] = tensor
                        remaining -= units
                for name in choices:
                    choices[name] = "ab" if name in selected_a and name in selected_b else "a" if name in selected_a else "b" if name in selected_b else "none"
                selection = RiftSelection(updates_A=selected_a, updates_B=selected_b, choices=choices,
                    full_ab_bytes=full_bytes, budget_bytes=cap, selected_bytes=0, budget_utilization=0,
                    captured_gain=0.0, oracle_gain=0.0, captured_gain_ratio=0.0, reliability=reliability,
                    option_counts={mode: sum(v == mode for v in choices.values()) for mode in ["none", "a", "b", "ab"]})
            blob = encode_raw(selected_a, selected_b)
            decoded_a, decoded_b, header = decode_message(blob, delta_a, delta_b)
            selection = replace(selection, updates_A=decoded_a, updates_B=decoded_b,
                                budget_bytes=cap, selected_bytes=len(blob), budget_utilization=len(blob) / max(1, cap))
        else:
            rng = self.rng.setdefault(int(client_id), np.random.default_rng(int(self.args.seed) * 1000003 + int(client_id) * 997))
            blob = encode_coordinates(delta_a, delta_b, codec, cap, rng)
            decoded_a, decoded_b, header = decode_message(blob, delta_a, delta_b)
            choices = {name: "ab" for name in sorted(set(decoded_a) | set(decoded_b))}
            selection = RiftSelection(updates_A=decoded_a, updates_B=decoded_b, choices=choices,
                full_ab_bytes=full_bytes, budget_bytes=cap, selected_bytes=len(blob),
                budget_utilization=len(blob) / max(1, cap), captured_gain=0.0, oracle_gain=0.0,
                captured_gain_ratio=0.0, reliability=reliability,
                option_counts={"none": 0, "a": 0, "b": 0, "ab": len(choices)})
        if len(blob) > cap:
            raise AssertionError("Message exceeds its true byte cap")
        message_dir = self.out / "messages"
        message_dir.mkdir(exist_ok=True)
        filename = f"r{int(round_num):03d}_c{int(client_id):03d}.bin"
        (message_dir / filename).write_bytes(blob)
        record = {"round": int(round_num), "client": int(client_id), "codec": codec,
                  "cap_bytes": cap, "wire_bytes": len(blob), "full_ab_bytes": full_bytes,
                  "message_sha256": hashlib.sha256(blob).hexdigest(), "message_file": filename,
                  "header": header, "choices": selection.choices,
                  "selection_seed": int(self.args.seed) * 1000003 + int(client_id) * 997,
                  "arrived": None}
        if edge_weights:
            record["edge_weight_min"] = float(min(edge_weights.values()))
            record["edge_weight_max"] = float(max(edge_weights.values()))
            record["edge_weight_mean"] = float(sum(edge_weights.values()) / len(edge_weights))
        self.rows[(int(round_num), int(client_id))] = record
        self.round_record["clients"][str(client_id)] = record
        if codec == "lookahead_rift":
            self.factor_debt[int(client_id)] = {
                "__global__": int(self.factor_debt.get(int(client_id), {}).get("__global__", 0)) + sum(
                    1 if mode == "a" else -1 if mode == "b" else 0
                    for mode in selection.choices.values()
                )
            }
        elif codec == "balanced_lookahead_rift":
            debt = self.factor_debt.setdefault(int(client_id), {})
            for name, mode in selection.choices.items():
                if mode == "ab":
                    debt[name] = 0
                elif mode == "a":
                    debt[name] = int(debt.get(name, 0)) + 1
                elif mode == "b":
                    debt[name] = int(debt.get(name, 0)) - 1
        return selection

    def record_arrival(self, round_num, cid, arrived, decoded_a, decoded_b, corrected_a, corrected_b, state):
        if self.args.wire_codec in {"rift", "random_rift", "edge_random_rift", "lossaware_rift", "lookahead_rift", "balanced_lookahead_rift", "counterfactual_rift"}:
            stats = state.acknowledge(cid, corrected_a, corrected_b,
                                      self.rows[(int(round_num), int(cid))]["choices"], arrived=arrived)
        else:
            stats = acknowledge_coordinates(state, cid, corrected_a, corrected_b, decoded_a, decoded_b, arrived)
        self.rows[(int(round_num), int(cid))]["arrived"] = bool(arrived)
        return stats

    def finish_round(self):
        with (self.out / "control_trace.jsonl").open("a") as stream:
            stream.write(json.dumps(self.round_record, sort_keys=True) + "\n")
        with (self.out / "wire_metrics.jsonl").open("a") as stream:
            for row in self.round_record["clients"].values():
                stream.write(json.dumps(row, sort_keys=True) + "\n")
