"""Prove the 'unused_train' proxy reserve leaves the client split untouched.

Compares two constructions of the same seeded CIFAR-100 partition:
  A. proxy_reserve_size = 0                      (published d034b main campaign)
  B. proxy_reserve_size = 100, mode unused_train (new Fed-HeLLo FIM proxy)

Required to hold:
  1. every client's train/val/test index list is bit-identical between A and B;
  2. the per-client train subsets actually consumed at data_fraction=0.2 are
     bit-identical between A and B;
  3. the reserved proxy indices are disjoint from every consumed train index;
  4. the reserved proxy indices are disjoint from every val/test index.
"""
from __future__ import annotations

import sys

import numpy as np

from dataset import CIFAR100Dataset

SEED = 42
FRACTION = 0.2
PROXY = 100
DATA_DIR = "/data/fedsonata_journal/data"


def build(reserve: int, mode: str):
    ds = CIFAR100Dataset(
        data_dir=DATA_DIR,
        alpha=0.1,
        num_clients=20,
        seed=SEED,
        splits=[0.8, 0.1, 0.1],
        data_partition="dirichlet",
        proxy_reserve_size=reserve,
        proxy_reserve_mode=mode,
    )
    used = {}
    for cid in range(ds.num_clients):
        # Same call main.py makes; consumes the global numpy RNG in order.
        loader = ds.get_client_dataloader(
            cid, batch_size=16, shuffle=True, split="train", fraction=FRACTION
        )
        base = ds.client_datasets[cid]["train"]
        base_idx = [int(i) for i in base.indices]
        # loader.dataset is Subset(base, positions) -> map positions to global ids
        used[cid] = set(base_idx[int(p)] for p in loader.dataset.indices)
    if reserve > 0 and mode == "unused_train":
        ds.reserve_unused_train_proxy()
    return ds, used


def subset_indices(ds, cid, split):
    sub = ds.client_datasets[cid][split]
    return [int(i) for i in sub.indices]


def main() -> int:
    print("== building A: reserve=0 (published path) ==")
    ds_a, used_a = build(0, "pre_partition")
    print("== building B: reserve=100 mode=unused_train ==")
    ds_b, used_b = build(PROXY, "unused_train")

    failures = []

    for cid in range(ds_a.num_clients):
        for split in ("train", "val", "test"):
            a = subset_indices(ds_a, cid, split)
            b = subset_indices(ds_b, cid, split)
            if a != b:
                failures.append(f"client {cid} split '{split}' differs")

    for cid in range(ds_a.num_clients):
        if used_a[cid] != used_b[cid]:
            failures.append(
                f"client {cid} consumed train subset differs "
                f"(|A|={len(used_a[cid])}, |B|={len(used_b[cid])}, "
                f"symdiff={len(used_a[cid] ^ used_b[cid])})"
            )

    proxy = set(int(i) for i in (ds_b.proxy_indices or []))
    if len(proxy) != PROXY:
        failures.append(f"proxy has {len(proxy)} indices, expected {PROXY}")

    consumed = set().union(*used_b.values()) if used_b else set()
    overlap_train = proxy & consumed
    if overlap_train:
        failures.append(
            f"proxy overlaps consumed train data in {len(overlap_train)} indices"
        )

    held_out = set()
    for cid in range(ds_b.num_clients):
        for split in ("val", "test"):
            held_out.update(subset_indices(ds_b, cid, split))
    overlap_eval = proxy & held_out
    if overlap_eval:
        failures.append(
            f"proxy overlaps val/test data in {len(overlap_eval)} indices"
        )

    allocated = set()
    for cid in range(ds_b.num_clients):
        allocated.update(subset_indices(ds_b, cid, "train"))
    outside = proxy - allocated
    if outside:
        failures.append(f"{len(outside)} proxy indices are outside every train split")

    print("\n--- checks ---")
    print(f"clients                  : {ds_a.num_clients}")
    print(f"consumed train (A)       : {len(set().union(*used_a.values()))}")
    print(f"consumed train (B)       : {len(consumed)}")
    print(f"allocated train pool     : {len(allocated)}")
    print(f"unused train pool        : {len(allocated - consumed)}")
    print(f"proxy size               : {len(proxy)}")
    print(f"proxy ∩ consumed train   : {len(overlap_train)}")
    print(f"proxy ∩ val/test         : {len(overlap_eval)}")

    if failures:
        print("\nFAIL:")
        for line in failures:
            print("  -", line)
        return 1
    print("\nPASS: partition, val/test and consumed-train subsets are bit-identical; "
          "proxy is disjoint from everything used for training/validation/reporting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
