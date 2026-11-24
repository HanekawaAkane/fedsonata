#!/usr/bin/env python
"""FS23 campaign status monitor (read-only).

Usage: python tests/monitor_status.py
Prints per-run state (DONE/FAILED/running/round progress) and pairing checks:
trace hash + data split hash must match within each (scenario, seed) block.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")


def main() -> int:
    if not os.path.isdir(RESULTS):
        print("no results dir")
        return 1
    dirs = sorted(d for d in os.listdir(RESULTS) if d.startswith("fs23_20261117_") and not d.endswith("_smoke"))
    blocks = {}
    n_done = n_fail = n_run = n_other = 0
    for d in dirs:
        p = os.path.join(RESULTS, d)
        if os.path.exists(os.path.join(p, "DONE")):
            state = "DONE"
            n_done += 1
        elif os.path.exists(os.path.join(p, "FAILED")):
            state = "FAILED"
            n_fail += 1
        elif os.path.exists(os.path.join(p, "stdout.log")):
            state = "RUNNING"
            n_run += 1
        else:
            state = "EMPTY/NEW"
            n_other += 1
        # parse tag: fs23_20261117_<scen>_<arm>_s<seed>
        body = d.split("fs23_20261117_", 1)[1]
        scen = body.split("_", 1)[0]
        rest = body.split("_", 1)[1]
        seed = rest.rsplit("_s", 1)[1]
        arm = rest.rsplit("_s", 1)[0]
        rounds = ""
        cfg_info = {}
        cfg_p = os.path.join(p, "config.json")
        if os.path.exists(cfg_p):
            try:
                cfg = json.load(open(cfg_p, encoding="utf-8"))
                cfg_info = {
                    "trace": cfg.get("trace_hash", "")[:12],
                    "split": cfg.get("client_data_split_hash", "")[:12],
                }
                blocks.setdefault((scen, seed), {})[d] = cfg_info
            except Exception:
                pass
        log_p = os.path.join(p, "run.log")
        if state == "RUNNING" and os.path.exists(log_p):
            try:
                with open(log_p, encoding="utf-8") as fh:
                    lines = fh.readlines()
                last = [l for l in lines if "summary" in l]
                rounds = last[-1].split("round ")[1].split(" ")[0] if last else "?"
                rounds = f" round~{rounds}"
            except Exception:
                pass
        print(f"{state:10s} {d}{rounds}  trace={cfg_info.get('trace','-'):12s} split={cfg_info.get('split','-'):12s}")

    print(f"\nTOTAL done={n_done} failed={n_fail} running={n_run} other={n_other} / {len(dirs)}")

    # Pairing: within (scenario, seed), trace/split hashes must be identical.
    bad = 0
    for (scen, seed), runs in sorted(blocks.items()):
        traces = {v["trace"] for v in runs.values()}
        splits = {v["split"] for v in runs.values()}
        if len(traces) > 1 or len(splits) > 1:
            bad += 1
            print(f"PAIRING MISMATCH scen={scen} seed={seed}: traces={traces} splits={splits}")
    print("pairing OK" if bad == 0 else f"{bad} pairing blocks MISMATCH")
    return 1 if (n_fail or bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
