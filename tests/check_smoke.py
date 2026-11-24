#!/usr/bin/env python
"""FS23 smoke validation.

Checks after the smoke matrix:
  S1  every smoke run has DONE and no FAILED.
  S2  frozen pre-radio plan: round-1 block allocation, factor mask and wire
      byte counts identical across the three radio arms (same seed/input);
      later arrivals may diverge (that is the point of the intervention).
  S3  consumer views actually mask: AGE_OFF visible age == 0; NO_RELIABILITY /
      DELIVERY_CONTROL_OPEN visible reliability == 1.0 while raw updates live.
  S4  IDEAL smoke: no dropout / duration / deadline-miss failures.
  S5  each run's config records the intended fs23_intervention mapping.

Usage: python tests/check_smoke.py   (from the fs23 experiment root)
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    smoke_dirs = sorted(
        d for d in os.listdir(RESULTS) if d.endswith("_smoke")
    ) if os.path.isdir(RESULTS) else []
    print(f"smoke dirs: {smoke_dirs}")
    check("S1 at least 6 smoke runs present", len(smoke_dirs) >= 6, str(smoke_dirs))

    by_tag = {}
    for d in smoke_dirs:
        p = os.path.join(RESULTS, d)
        done = os.path.exists(os.path.join(p, "DONE"))
        failed = os.path.exists(os.path.join(p, "FAILED"))
        check(f"S1 {d}: DONE present, no FAILED", done and not failed)
        cfg = json.load(open(os.path.join(p, "config.json"), encoding="utf-8"))
        fs23 = cfg.get("fs23_intervention") or {}
        by_tag[d] = {"path": p, "cfg": cfg, "fs23": fs23}

    # S5: intervention mapping sanity.
    expect = {
        "full": ("cora", "convex_joint", "dual", "observed"),
        "radio_equal": ("equal_fixed", "equal_fixed", "dual", "observed"),
        "radio_bapft": ("bapft_style", "bapft_ba", "dual", "observed"),
        "age_off": ("cora", "convex_joint", "zero", "observed"),
        "no_reliability": ("cora", "convex_joint", "dual", "constant_initial"),
        "delivery_control_open": ("cora", "convex_joint", "allocation_only", "constant_initial"),
    }
    for tag, (radio, dispatch, age, rel) in expect.items():
        match = [d for d in smoke_dirs if f"_{tag}_" in d or d.endswith(f"_{tag}_smoke")]
        # dir pattern: fs23_20261117_<scenario>_<arm>_s<seed>_smoke
        match = []
        for d in smoke_dirs:
            parts = d.split("_")
            # arm token spans between scenario and s<seed>
            arm = d.split("fs23_20261117_", 1)[1]
            arm = arm.split(f"_s", 1)[0]
            arm = arm.split("_", 1)[1] if "_" in arm else arm
            if arm == tag:
                match.append(d)
        if not match:
            # try direct containment as fallback
            match = [d for d in smoke_dirs if tag in d]
        for d in match:
            f = by_tag[d]["fs23"]
            ok = (
                f.get("radio_policy_cli") == radio
                and f.get("radio_policy_dispatch") == dispatch
                and f.get("age_view") == age
                and f.get("reliability_view") == rel
            )
            check(f"S5 {d} intervention mapping", ok, json.dumps(f))

    # S2: frozen pre-radio plan across radio arms (round 1).
    def round1_wire(path):
        cand = os.path.join(path, "control_trace.jsonl")
        if not os.path.exists(cand):
            return None
        rows = load_jsonl(cand)
        for row in rows:
            if int(row.get("round", -1)) == 1:
                return row
        return None

    radio_runs = {}
    for d in smoke_dirs:
        arm = d.split("fs23_20261117_", 1)[1].split(f"_s", 1)[0]
        arm = arm.split("_", 1)[1] if "_" in arm else arm
        if arm in ("full", "radio_equal", "radio_bapft"):
            radio_runs[arm] = by_tag[d]["path"]
    if len(radio_runs) == 3:
        rows = {k: round1_wire(v) for k, v in radio_runs.items()}
        ok_alloc = (
            rows["full"] is not None
            and rows["full"].get("allocation") == rows["radio_equal"].get("allocation")
            == rows["radio_bapft"].get("allocation")
        )
        check("S2 round-1 block allocation identical across radio arms", ok_alloc)

        def caps(row):
            return {
                cid: (c.get("cap_bytes"), c.get("full_ab_bytes"))
                for cid, c in (row.get("clients") or {}).items()
            }
        ok_bytes = (
            rows["full"] is not None
            and caps(rows["full"]) == caps(rows["radio_equal"]) == caps(rows["radio_bapft"])
        )
        check("S2 round-1 wire byte caps identical across radio arms", ok_bytes)

        def masks(row):
            return {
                cid: c.get("choices") or c.get("option_counts")
                for cid, c in (row.get("clients") or {}).items()
            }
        ok_mask = (
            rows["full"] is not None
            and masks(rows["full"]) == masks(rows["radio_equal"]) == masks(rows["radio_bapft"])
        )
        check("S2 round-1 factor masks identical across radio arms", ok_mask)

        def msgs(row):
            return {
                cid: c.get("message_sha256")
                for cid, c in (row.get("clients") or {}).items()
            }
        ok_msg = (
            rows["full"] is not None
            and msgs(rows["full"]) == msgs(rows["radio_equal"]) == msgs(rows["radio_bapft"])
        )
        check("S2 round-1 message bytes identical across radio arms", ok_msg)
    else:
        check("S2 three radio smoke runs present", False, str(list(radio_runs)))

    # S3: view masking visible in STATE_VIEWS.
    for d in smoke_dirs:
        arm = d.split("fs23_20261117_", 1)[1].split(f"_s", 1)[0]
        arm = arm.split("_", 1)[1] if "_" in arm else arm
        sv = os.path.join(by_tag[d]["path"], "STATE_VIEWS.jsonl")
        if not os.path.exists(sv):
            check(f"S3 {d} STATE_VIEWS exists", False)
            continue
        rows = load_jsonl(sv)
        last = rows[-1]
        if arm == "age_off":
            ok = all(r["visible_effective"] == 0.0 for r in last["age_rows"])
            check("S3 AGE_OFF visible age all zero", ok)
        if arm in ("no_reliability", "delivery_control_open"):
            ok = all(v == 1.0 for v in last["reliability_visible"].values())
            raw_moves = any(abs(v - 1.0) > 1e-12 for v in last["reliability_raw"].values())
            check(f"S3 {arm} visible reliability pinned to 1.0", ok)
            check(f"S3 {arm} raw reliability still updates", raw_moves)
        if arm == "delivery_control_open" or arm == "alloc_only":
            ok = all(r["visible_effective"] == float(r["raw_alloc"]) for r in last["age_rows"])
            check(f"S3 {arm} visible age == raw alloc only", ok)
        if arm == "full":
            ok = all(
                r["visible_effective"] == float(r["raw_alloc"] + r["raw_net"])
                for r in last["age_rows"]
            )
            check("S3 FULL visible age == alloc+net", ok)

    # S4: IDEAL smoke has zero failures.
    for d in smoke_dirs:
        if "ideal" not in d:
            continue
        sv = os.path.join(by_tag[d]["path"], "STATE_VIEWS.jsonl")
        rows = load_jsonl(sv)
        fails = {}
        for r in rows:
            for k, v in (r.get("failure_counts") or {}).items():
                if k != "none":
                    fails[k] = fails.get(k, 0) + v
        check("S4 IDEAL smoke zero non-none failures", fails == {}, str(fails))
        raw_reliability = rows[-1]["reliability_raw"]
        ok = all(abs(v - 1.0) < 1e-12 for v in raw_reliability.values())
        check("S4 IDEAL reliability stays 1.0", ok)

    print(f"\n{len(FAILURES)} failure(s): {FAILURES}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
