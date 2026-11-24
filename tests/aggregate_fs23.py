#!/usr/bin/env python
"""FS23 results aggregator (read-only).

Reads every completed fs23 run under results/, extracts primary endpoints
and resource costs, and emits:
  - results/fs23_aggregate/report.json
  - results/fs23_aggregate/report.csv
  - a printed summary with paired per-seed deltas (PROTOCOL_CN statistics:
    paired difference -> mean, sample SD, 95% t interval, df=2 for 3 seeds).

Usage: python tests/aggregate_fs23.py [results_dir]
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results")
OUT = os.path.join(RESULTS, "fs23_aggregate")

T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_tag(d):
    body = d.split("fs23_20261117_", 1)[1]
    scen = body.split("_", 1)[0]
    rest = body.split("_", 1)[1]
    seed = rest.rsplit("_s", 1)[1]
    arm = rest.rsplit("_s", 1)[0]
    return scen, arm, seed


def collect_run(path):
    cfg = json.load(open(os.path.join(path, "config.json"), encoding="utf-8"))
    fin = json.load(open(os.path.join(path, "final_metrics.json"), encoding="utf-8"))
    stop = json.load(open(os.path.join(path, "STOPPING.json"), encoding="utf-8"))
    rounds = load_jsonl(os.path.join(path, "metrics.jsonl"))
    rounds = [r for r in rounds if r.get("row_type") == "round_summary"]
    rounds.sort(key=lambda r: r["round"])

    cum = {
        "attempted_ul_bytes": 0.0, "dl_bytes": 0.0, "compute_energy_j": 0.0,
        "tx_energy_j": 0.0, "mec_time_s": 0.0, "on_time_sum": 0.0,
        "deadline_miss": 0, "base_drop": 0, "duration_drop": 0,
        "lpc_events": 0, "solver_ms": 0.0, "fallback": 0,
    }
    per_round = []
    for r in rounds:
        cum["attempted_ul_bytes"] += float(r.get("upload_bytes", 0.0))
        cum["dl_bytes"] += float(r.get("download_bytes", 0.0))
        cum["compute_energy_j"] += float(r.get("compute_energy_j", 0.0))
        cum["tx_energy_j"] += float(r.get("tx_energy_j", 0.0))
        cum["mec_time_s"] += float(r.get("mec_latency_s", 0.0))
        cum["on_time_sum"] += float(r.get("on_time_rate", 0.0))
        cum["deadline_miss"] += int(r.get("deadline_miss_count", 0))
        cum["base_drop"] += int(r.get("base_dropout_clients", 0))
        cum["duration_drop"] += int(r.get("duration_dropout_clients", 0))
        cum["lpc_events"] += int(r.get("lpc_events", 0))
        cum["solver_ms"] += float(r.get("solver_ms", 0.0)) + float(r.get("cora_predictor_solver_ms", 0.0))
        per_round.append({
            "round": r["round"], "acc_val": r.get("accuracy_personalized"),
            "on_time": r.get("on_time_rate"), "deadline_miss": r.get("deadline_miss_count"),
        })
    n = max(1, len(rounds))

    # recovery/validation target info
    val_curve = [r.get("accuracy_personalized") for r in rounds]
    targets = {}
    for t in (85.0, 88.0):
        hit = next((i + 1 for i, v in enumerate(val_curve) if v is not None and v * 100.0 >= t), None)
        targets[f"val_ge_{int(t)}"] = hit

    # target hit costs: cumulative bytes/energy/time up to hit round
    target_costs = {}
    for key, hit in targets.items():
        if hit:
            sub = rounds[:hit]
            target_costs[key] = {
                "round": hit,
                "ul_bytes": sum(float(r.get("upload_bytes", 0.0)) for r in sub),
                "energy_j": sum(float(r.get("compute_energy_j", 0.0)) + float(r.get("tx_energy_j", 0.0)) for r in sub),
                "mec_time_s": sum(float(r.get("mec_latency_s", 0.0)) for r in sub),
            }
        else:
            target_costs[key] = {"round": None, "note": "not reached within 20 rounds"}

    return {
        "run_id": os.path.basename(path),
        "seed": cfg.get("seed"),
        "trace_hash": cfg.get("trace_hash"),
        "data_split_hash": cfg.get("client_data_split_hash"),
        "fs23": cfg.get("fs23_intervention"),
        "accuracy_final_round": fin.get("accuracy_final_round"),
        "accuracy_validation_selected": fin.get("accuracy_validation_selected"),
        "accuracy_legacy": fin.get("accuracy"),
        "selected_checkpoint_round": fin.get("selected_checkpoint_round"),
        "best_validation_round": stop.get("best_validation_round"),
        "restored_best_validation": stop.get("restored_best_validation"),
        "stopped_early": stop.get("stopped_early"),
        "completed_rounds": fin.get("completed_rounds"),
        "cum": cum,
        "mean_on_time": cum["on_time_sum"] / n,
        "targets": targets,
        "target_costs": target_costs,
        "per_round": per_round,
    }


def paired_stats(values):
    """values: list of per-seed differences -> mean, sd, 95% t CI."""
    n = len(values)
    if n == 0:
        return None
    m = sum(values) / n
    if n == 1:
        return {"n": 1, "mean": m, "sd": None, "ci95": None, "values": values}
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
    tc = T_CRIT_95.get(n - 1, 2.0)
    half = tc * sd / math.sqrt(n)
    return {"n": n, "mean": m, "sd": sd, "ci95": [m - half, m + half], "values": values}


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    runs = {}
    for d in sorted(os.listdir(RESULTS)):
        p = os.path.join(RESULTS, d)
        if not (d.startswith("fs23_20261117_") and not d.endswith("_smoke") and "smoke" not in d):
            continue
        if not os.path.exists(os.path.join(p, "DONE")):
            continue
        scen, arm, seed = parse_tag(d)
        runs[(scen, arm, seed)] = collect_run(p)

    print(f"collected {len(runs)} runs")
    report = {"runs": {}, "paired": {}}
    for key, r in sorted(runs.items()):
        report["runs"]["|".join(key)] = r

    # ---- per-arm aggregate table ---- #
    arms = sorted({k[1] for k in runs})
    seeds = sorted({k[2] for k in runs}, key=lambda s: int(s))
    scen_base = [("base", a, s) for a in arms for s in seeds]
    print("\n== Per-arm endpoints (base scenario) ==")
    header = f"{'arm':24s} " + " ".join(f"{'s'+s:>10s}" for s in seeds) + f" {'mean':>10s} {'sd':>8s}"
    print(header)
    arm_means = {}
    for field in ("accuracy_final_round", "accuracy_validation_selected"):
        print(f"-- {field} (%)")
        for arm in arms:
            vals = []
            for s in seeds:
                r = runs.get(("base", arm, s))
                v = r[field] * 100.0 if r and r.get(field) is not None else None
                vals.append(v)
            ok = [v for v in vals if v is not None]
            mean = sum(ok) / len(ok) if ok else float("nan")
            sd = math.sqrt(sum((v - mean) ** 2 for v in ok) / (len(ok) - 1)) if len(ok) > 1 else float("nan")
            arm_means[(field, arm)] = (vals, mean, sd)
            cells = " ".join(f"{v:10.3f}" if v is not None else f"{'-':>10s}" for v in vals)
            print(f"{arm:24s} {cells} {mean:10.3f} {sd:8.3f}")

    print("\n== Per-arm cumulative resources (base, mean over seeds) ==")
    res_fields = [
        ("attempted_ul_bytes", 1e6, "UL_MB"),
        ("dl_bytes", 1e6, "DL_MB"),
        ("compute_energy_j", 1.0, "comp_E_J"),
        ("tx_energy_j", 1.0, "tx_E_J"),
        ("mec_time_s", 1.0, "MEC_s"),
        ("deadline_miss", 1.0, "dl_miss"),
        ("lpc_events", 1.0, "lpc"),
    ]
    for arm in arms:
        parts = []
        for f, scale, label in res_fields:
            vals = []
            for s in seeds:
                r = runs.get(("base", arm, s))
                if r:
                    vals.append(r["cum"][f] / scale)
            mean = sum(vals) / len(vals) if vals else float("nan")
            parts.append(f"{label}={mean:.2f}")
        r0 = runs.get(("base", arm, seeds[0]))
        ot = sum(runs[("base", arm, s)]["mean_on_time"] for s in seeds if ("base", arm, s) in runs)
        ot /= max(1, len(seeds))
        print(f"{arm:24s} on_time={ot:.3f} " + " ".join(parts))

    # ---- pre-specified paired comparisons ---- #
    def diffs(field, scen, arm_a, arm_b, scale=1.0):
        out = []
        for s in seeds:
            ra = runs.get((scen, arm_a, s))
            rb = runs.get((scen, arm_b, s))
            if ra and rb and ra.get(field) is not None and rb.get(field) is not None:
                out.append((ra[field] - rb[field]) * scale)
        return out

    comps = [
        ("A: FULL - RADIO_EQUAL (holdout pp)", ("base", "full", "radio_equal", "accuracy_final_round", 100.0)),
        ("A: RADIO_BAPFT - FULL (holdout pp, secondary)", ("base", "radio_bapft", "full", "accuracy_final_round", 100.0)),
        ("B: FULL - ALLOC_ONLY (holdout pp)", ("base", "full", "alloc_only", "accuracy_final_round", 100.0)),
        ("B: FULL - DELIVERY_ONLY (holdout pp)", ("base", "full", "delivery_only", "accuracy_final_round", 100.0)),
        ("B: FULL - AGE_OFF (holdout pp)", ("base", "full", "age_off", "accuracy_final_round", 100.0)),
        ("C: FULL - DELIVERY_CONTROL_OPEN (holdout pp)", ("base", "full", "delivery_control_open", "accuracy_final_round", 100.0)),
        ("C: FULL - NO_RELIABILITY (holdout pp)", ("base", "full", "no_reliability", "accuracy_final_round", 100.0)),
        ("A: FULL - RADIO_EQUAL (UL MB)", ("base", "full", "radio_equal", "UL_MB", 1.0)),
    ]
    print("\n== Pre-specified paired comparisons (per-seed diff, 95% t CI, df=2) ==")
    for name, (scen, a, b, field, scale) in comps:
        if field == "UL_MB":
            vals = []
            for s in seeds:
                ra = runs.get((scen, a, s))
                rb = runs.get((scen, b, s))
                if ra and rb:
                    vals.append((ra["cum"]["attempted_ul_bytes"] - rb["cum"]["attempted_ul_bytes"]) / 1e6)
        else:
            vals = diffs(field, scen, a, b, scale)
        st = paired_stats(vals)
        report["paired"][name] = st
        if st:
            ci = f"[{st['ci95'][0]:.3f}, {st['ci95'][1]:.3f}]" if st.get("ci95") else "n/a"
            print(f"{name:52s} mean={st['mean']:+.3f} sd={st['sd']:.3f} ci95={ci} values={[round(v,3) for v in st['values']]}")

    # ---- interaction quantity I = Y11 - Y10 - Y01 + Y00 (staleness 2x2) ---- #
    print("\n== B interaction I = FULL - ALLOC_ONLY - DELIVERY_ONLY + AGE_OFF (holdout pp) ==")
    inter = []
    for s in seeds:
        y11 = runs.get(("base", "full", s))
        y10 = runs.get(("base", "alloc_only", s))
        y01 = runs.get(("base", "delivery_only", s))
        y00 = runs.get(("base", "age_off", s))
        if y11 and y10 and y01 and y00:
            inter.append(
                (y11["accuracy_final_round"] - y10["accuracy_final_round"]
                 - y01["accuracy_final_round"] + y00["accuracy_final_round"]) * 100.0
            )
    st = paired_stats(inter)
    report["paired"]["B_interaction"] = st
    if st:
        ci = f"[{st['ci95'][0]:.3f}, {st['ci95'][1]:.3f}]" if st.get("ci95") else "n/a"
        print(f"{'I':52s} mean={st['mean']:+.3f} sd={st['sd']:.3f} ci95={ci} values={[round(v,3) for v in st['values']]}")

    # ---- CSV flat export ---- #
    csv_path = os.path.join(OUT, "report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "arm", "seed", "acc_final_round_pct", "acc_val_selected_pct",
                    "selected_round", "best_val_round", "UL_MB", "DL_MB", "comp_E_J", "tx_E_J",
                    "MEC_s", "on_time", "deadline_miss", "base_drop", "duration_drop",
                    "lpc_events", "trace_hash", "split_hash"])
        for (scen, arm, seed), r in sorted(runs.items()):
            w.writerow([
                scen, arm, seed,
                round(r["accuracy_final_round"] * 100, 4) if r.get("accuracy_final_round") is not None else "",
                round(r["accuracy_validation_selected"] * 100, 4) if r.get("accuracy_validation_selected") is not None else "",
                r.get("selected_checkpoint_round"), r.get("best_validation_round"),
                round(r["cum"]["attempted_ul_bytes"] / 1e6, 2), round(r["cum"]["dl_bytes"] / 1e6, 2),
                round(r["cum"]["compute_energy_j"], 2), round(r["cum"]["tx_energy_j"], 2),
                round(r["cum"]["mec_time_s"], 2), round(r["mean_on_time"], 4),
                r["cum"]["deadline_miss"], r["cum"]["base_drop"], r["cum"]["duration_drop"],
                r["cum"]["lpc_events"], r.get("trace_hash", "")[:12], r.get("data_split_hash", "")[:12],
            ])

    with open(os.path.join(OUT, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"\nwritten: {csv_path}")
    print(f"written: {os.path.join(OUT, 'report.json')}")

    # ---- completion checks ---- #
    print("\n== Completion checks ==")
    bad = 0
    for (scen, arm, seed), r in sorted(runs.items()):
        issues = []
        if r.get("completed_rounds") != 20:
            issues.append(f"rounds={r.get('completed_rounds')}")
        if r.get("stopped_early"):
            issues.append("stopped_early")
        for pr in r["per_round"]:
            if pr.get("acc_val") is not None and (math.isnan(pr["acc_val"]) or math.isinf(pr["acc_val"])):
                issues.append(f"NaN/Inf at round {pr['round']}")
        if issues:
            bad += 1
            print(f"ISSUE {r['run_id']}: {issues}")
    print(f"runs with issues: {bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
