#!/usr/bin/env python
"""FS23 intervention-switch patcher for the archived Nov6 projection-LPC main.

Applies surgical text patches to the hash-verified snapshot of
lossaware_wire_matched_20261016/code so that:

  1. --radio_policy {auto,cora,equal_fixed,bapft_style} dispatches ONLY the
     final actual-payload radio-plan block (CORA / Equal-Fixed / BAPFT-style),
     keeping mec_method=proposed_joint_rift_ef_cora (predictor, block
     allocation, RIFT factor selection, LPC, LEF untouched and counted in all
     arms).
  2. --age_view {dual,allocation_only,delivery_only,zero,merged} masks the
     consumer-visible effective staleness while raw dual counters keep true
     updates (via patched mec/staleness.py).
  3. --reliability_view {observed,constant_initial} masks what reliability
     consumers read (P1, byte cap, predictor, corrector) while the raw EWMA
     keeps true updates.
  4. --radio_outer_age_weight decouples the radio outer age coefficient from
     the effective-age composition weight (--age_delivery_weight, defaulting
     to legacy --mec_omega_net when unset).
  5. --network_scenario {base,ideal,shift} drives pre-declared GE / hazard
     schedules (patched mec/traces.py); BASE stays byte-identical.

Usage: python patch_fs23_switches.py <code_dir>
Every anchor is asserted to match exactly once. The original files are
backed up as *.orig_fs23 next to the patched copies.
"""

from __future__ import annotations

import os
import shutil
import sys


def patch(path: str, replacements, inserts=None) -> None:
    """Apply (old, new) replacements and (anchor, inserted_after) inserts."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    original = text
    for old, new in replacements:
        n = text.count(old)
        assert n == 1, f"anchor appears {n}x (expected 1) in {path}:\n{old[:200]}"
        text = text.replace(old, new)
    for anchor, inserted in inserts or []:
        n = text.count(anchor)
        assert n == 1, f"insert anchor appears {n}x (expected 1) in {path}:\n{anchor[:200]}"
        text = text.replace(anchor, anchor + inserted)
    if text == original:
        raise AssertionError(f"no change applied to {path}")
    backup = path + ".orig_fs23"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"patched {path}")


def main() -> int:
    code = sys.argv[1] if len(sys.argv) > 1 else "."
    main_py = os.path.join(code, "main.py")
    staleness_py = os.path.join(code, "mec", "staleness.py")
    traces_py = os.path.join(code, "mec", "traces.py")

    # ------------------------------------------------------------------ #
    # mec/staleness.py and mec/traces.py: full replacements (already
    # prepared, byte-checked against the unpatched originals before swap).
    # ------------------------------------------------------------------ #
    here = os.path.dirname(os.path.abspath(__file__))
    for src, dst in (
        (os.path.join(here, "patched_files", "mec_staleness.py"), staleness_py),
        (os.path.join(here, "patched_files", "mec_traces.py"), traces_py),
    ):
        backup = dst + ".orig_fs23"
        if not os.path.exists(backup):
            shutil.copy2(dst, backup)
        shutil.copyfile(src, dst)
        print(f"replaced {dst}")

    # ------------------------------------------------------------------ #
    # main.py patches
    # ------------------------------------------------------------------ #
    repl = []

    # --- 0. import ReliabilityView ------------------------------------ #
    repl.append((
        "from mec.scheduler import update_success_ewma",
        "from mec.scheduler import update_success_ewma\n"
        "from mec.staleness import ReliabilityView",
    ))

    # --- 1. new CLI args ---------------------------------------------- #
    repl.append((
        '    parser.add_argument("--mec_reliability_beta", type=float, default=0.8,\n'
        '                        help="到达可靠性 EWMA 系数 beta，越大越平滑 (默认: 0.8)")',
        '    parser.add_argument("--mec_reliability_beta", type=float, default=0.8,\n'
        '                        help="到达可靠性 EWMA 系数 beta，越大越平滑 (默认: 0.8)")\n'
        '    # ---- FS23 intervention switches (2026-11-17). These are independent\n'
        '    # ---- overrides; legacy aliases are rejected by argparse choices.\n'
        '    parser.add_argument("--radio_policy", type=str, default="auto",\n'
        '                        choices=["auto", "cora", "equal_fixed", "bapft_style"],\n'
        '                        help="最终 actual-payload 无线计划的 corrector 分派覆盖；"\n'
        '                             "auto=沿用 mec_method 派生策略 (默认: auto)")\n'
        '    parser.add_argument("--age_view", type=str, default="dual",\n'
        '                        choices=["dual", "allocation_only", "delivery_only",\n'
        '                                 "zero", "merged"],\n'
        '                        help="消费者可见有效陈旧度视图；raw 计数器始终按真实事件更新 (默认: dual)")\n'
        '    parser.add_argument("--reliability_view", type=str, default="observed",\n'
        '                        choices=["observed", "constant_initial"],\n'
        '                        help="消费者可见可靠性视图；raw EWMA 始终真实更新 (默认: observed)")\n'
        '    parser.add_argument("--age_delivery_weight", type=float, default=None,\n'
        '                        help="有效陈旧度合成中 delivery 项权重；未设置时沿用 --mec_omega_net")\n'
        '    parser.add_argument("--radio_outer_age_weight", type=float, default=None,\n'
        '                        help="无线规划外层年龄系数；未设置时沿用 --mec_omega_net")\n'
        '    parser.add_argument("--network_scenario", type=str, default="base",\n'
        '                        choices=["base", "ideal", "shift"],\n'
        '                        help="预设网络条件：base=主协议；ideal=无中断校验；"\n'
        '                             "shift=轮11-15 GE/hazard 压力段 (默认: base)")',
    ))

    # --- 2. radio_policy derivation + decoupled weights + helpers ------ #
    repl.append((
        "    else:\n"
        "        mec_network_policy = \"equal_fixed\"\n"
        "\n"
        "    method_l = str(args.method).lower()",
        "    else:\n"
        "        mec_network_policy = \"equal_fixed\"\n"
        "\n"
        "    # FS23: independent radio corrector dispatch (final payload plan only).\n"
        "    _radio_policy_map = {\n"
        "        \"cora\": \"convex_joint\",\n"
        "        \"equal_fixed\": \"equal_fixed\",\n"
        "        \"bapft_style\": \"bapft_ba\",\n"
        "    }\n"
        "    if str(args.radio_policy) == \"auto\":\n"
        "        radio_policy = mec_network_policy\n"
        "    else:\n"
        "        radio_policy = _radio_policy_map[str(args.radio_policy)]\n"
        "    # FS23: decouple the effective-age composition weight from the radio\n"
        "    # outer age coefficient (PROTOCOL_CN: both fixed at 1 in the protocol).\n"
        "    age_delivery_weight = (\n"
        "        float(args.mec_omega_net) if args.age_delivery_weight is None\n"
        "        else float(args.age_delivery_weight)\n"
        "    )\n"
        "    radio_outer_age_weight = (\n"
        "        float(args.mec_omega_net) if args.radio_outer_age_weight is None\n"
        "        else float(args.radio_outer_age_weight)\n"
        "    )\n"
        "\n"
        "    method_l = str(args.method).lower()",
    ))

    # --- 3. scenario setup, EffectiveStaleness view, hazard helper ----- #
    repl.append((
        "            mec_profiles.dropout.ge_good_to_bad = p_ge\n"
        "            mec_profiles.dropout.ge_bad_to_good = q_ge\n"
        "        mec_trace = TraceGenerator(\n"
        "            args.seed, args.num_clients, mec_profiles.channel, mec_profiles.dropout\n"
        "        )\n"
        "        mec_staleness = EffectiveStaleness(omega_net=args.mec_omega_net)",
        "            mec_profiles.dropout.ge_good_to_bad = p_ge\n"
        "            mec_profiles.dropout.ge_bad_to_good = q_ge\n"
        "        # FS23 network scenarios: pre-declared GE / hazard schedules.\n"
        "        # BASE keeps the exact archived construction (byte-identical).\n"
        "        fs23_ge_schedule = None\n"
        "        fs23_hazard_schedule = None\n"
        "        fs23_scenario = str(getattr(args, \"network_scenario\", \"base\"))\n"
        "        fs23_base_hazard = float(args.mec_duration_hazard_rate)\n"
        "        if fs23_scenario == \"ideal\":\n"
        "            # PROTOCOL_CN IDEAL: availability=1, hazard=0, same channel.\n"
        "            from mec.profiles import DropoutProfile as _DP\n"
        "            mec_profiles.dropout = _DP(model=\"none\", dropout_rate=0.0)\n"
        "            fs23_base_hazard = 0.0\n"
        "        elif fs23_scenario == \"shift\":\n"
        "            # PROTOCOL_CN SHIFT: rounds 1-10 main; 11-15 stress; 16-20 back.\n"
        "            # Explicit (p, q) per phase; GE state continuous (no re-draw).\n"
        "            _p1, _q1 = float(args.mec_ge_good_to_bad), (\n"
        "                float(args.mec_ge_good_to_bad) * (1.0 - float(args.mec_dropout_rate))\n"
        "                / float(args.mec_dropout_rate)\n"
        "                if float(args.mec_dropout_rate) > 0 else 1.0\n"
        "            )\n"
        "            fs23_ge_schedule = (\n"
        "                (1, 10, _p1, _q1),\n"
        "                (11, 15, 0.2, 0.3),\n"
        "                (16, int(args.num_rounds), _p1, _q1),\n"
        "            )\n"
        "            fs23_hazard_schedule = (\n"
        "                (1, 10, fs23_base_hazard),\n"
        "                (11, 15, 0.2),\n"
        "                (16, int(args.num_rounds), fs23_base_hazard),\n"
        "            )\n"
        "        mec_trace = TraceGenerator(\n"
        "            args.seed, args.num_clients, mec_profiles.channel, mec_profiles.dropout,\n"
        "            ge_schedule=fs23_ge_schedule,\n"
        "            hazard_schedule=fs23_hazard_schedule,\n"
        "            default_hazard=fs23_base_hazard,\n"
        "        )\n"
        "        mec_staleness = EffectiveStaleness(\n"
        "            omega_net=age_delivery_weight,\n"
        "            age_view=str(getattr(args, \"age_view\", \"dual\")),\n"
        "        )\n"
        "        mec_rel_view = ReliabilityView(\n"
        "            raw=mec_est_success,\n"
        "            view=str(getattr(args, \"reliability_view\", \"observed\")),\n"
        "            r_init=1.0,\n"
        "        )\n"
        "\n"
        "        def mec_hazard_for_round(round_num: int) -> float:\n"
        "            \"\"\"Duration hazard for this round under the active scenario.\"\"\"\n"
        "            if fs23_hazard_schedule is not None:\n"
        "                return float(mec_trace.hazard_rate(int(round_num)))\n"
        "            return fs23_base_hazard",
    ))

    # --- 4. trace hash with schedules ---------------------------------- #
    repl.append((
        "        mec_trace_hash = compute_trace_hash(\n"
        "            args.seed, args.num_clients, mec_profiles.channel,\n"
        "            mec_profiles.dropout, args.num_rounds,\n"
        "            duration_hazard_rate=float(args.mec_duration_hazard_rate),\n"
        "        )",
        "        mec_trace_hash = compute_trace_hash(\n"
        "            args.seed, args.num_clients, mec_profiles.channel,\n"
        "            mec_profiles.dropout, args.num_rounds,\n"
        "            duration_hazard_rate=fs23_base_hazard,\n"
        "            ge_schedule=fs23_ge_schedule,\n"
        "            hazard_schedule=fs23_hazard_schedule,\n"
        "        )",
    ))

    # --- 5. config dict: record intervention spec ---------------------- #
    repl.append((
        '                    "trace_hash": mec_trace_hash,\n'
        '                    "method": args.method,',
        '                    "trace_hash": mec_trace_hash,\n'
        '                    "fs23_intervention": {\n'
        '                        "radio_policy_cli": str(getattr(args, "radio_policy", "auto")),\n'
        '                        "radio_policy_dispatch": radio_policy,\n'
        '                        "age_view": str(getattr(args, "age_view", "dual")),\n'
        '                        "reliability_view": str(getattr(args, "reliability_view", "observed")),\n'
        '                        "age_delivery_weight": age_delivery_weight,\n'
        '                        "radio_outer_age_weight": radio_outer_age_weight,\n'
        '                        "network_scenario": fs23_scenario,\n'
        '                        "ge_schedule": (list(fs23_ge_schedule) if fs23_ge_schedule else None),\n'
        '                        "hazard_schedule": (list(fs23_hazard_schedule) if fs23_hazard_schedule else None),\n'
        '                    },\n'
        '                    "method": args.method,',
    ))

    # --- 6. consumer: P1 reliability ----------------------------------- #
    repl.append((
        "                    mec_alloc[cid] = reliability_aware_selection(\n"
        "                        cid, mec_layer_names, budget, st_state, mec_est_success,",
        "                    mec_alloc[cid] = reliability_aware_selection(\n"
        "                        cid, mec_layer_names, budget, st_state, mec_rel_view,",
    ))

    # --- 7. consumer: predictor reliability + outer weight + hazard ---- #
    repl.append((
        "                channel_gain=pre_channel,\n"
        "                reliability=mec_est_success,\n"
        "                budget_ratio=effective_rift_budget_ratio,",
        "                channel_gain=pre_channel,\n"
        "                reliability=mec_rel_view,\n"
        "                budget_ratio=effective_rift_budget_ratio,",
    ))
    repl.append((
        "                omega_net=float(args.mec_omega_net),\n"
        "                prev_staleness=pre_stale,\n"
        "                duration_hazard_rate=float(args.mec_duration_hazard_rate),",
        "                omega_net=radio_outer_age_weight,\n"
        "                prev_staleness=pre_stale,\n"
        "                duration_hazard_rate=mec_hazard_for_round(round_num),",
    ))

    # --- 8. consumer: wire byte-cap reliability ------------------------ #
    repl.append((
        "                    reliability=float(mec_est_success[client_idx]),",
        "                    reliability=float(mec_rel_view[client_idx]),",
    ))

    # --- 9. consumer: RIFT duration hazard ----------------------------- #
    repl.append((
        "                    duration_hazard_rate=(\n"
        "                        float(args.mec_duration_hazard_rate)\n"
        "                        if rift_duration_value_enabled else 0.0\n"
        "                    ),",
        "                    duration_hazard_rate=(\n"
        "                        float(mec_hazard_for_round(round_num))\n"
        "                        if rift_duration_value_enabled else 0.0\n"
        "                    ),",
    ))

    # --- 10. dispatch: radio_policy only in the final plan block -------- #
    repl.append((
        "            if args.mec_method != \"proposed_equal_bandwidth\":\n"
        "                if mec_network_policy == \"convex_joint\":",
        "            if args.mec_method != \"proposed_equal_bandwidth\":\n"
        "                if radio_policy == \"convex_joint\":",
    ))
    repl.append((
        "                elif mec_network_policy == \"proposed_joint\" and args.mec_method != \"proposed_fixed_deadline\":\n"
        "                    plan = solve_round(\n"
        "                        clients_k, s_up, t_cmp, h,\n"
        "                        mec_profiles.total_bandwidth_hz, mec_profiles.power_profile,\n"
        "                        n0, args.mec_d_min, args.mec_d_max, args.mec_lambda_e,\n"
        "                        args.mec_omega_net, prev_stale, mec_est_success,",
        "                elif radio_policy == \"proposed_joint\" and args.mec_method != \"proposed_fixed_deadline\":\n"
        "                    plan = solve_round(\n"
        "                        clients_k, s_up, t_cmp, h,\n"
        "                        mec_profiles.total_bandwidth_hz, mec_profiles.power_profile,\n"
        "                        n0, args.mec_d_min, args.mec_d_max, args.mec_lambda_e,\n"
        "                        radio_outer_age_weight, prev_stale, mec_rel_view,",
    ))
    repl.append((
        "                elif mec_network_policy == \"bapft_ba\":",
        "                elif radio_policy == \"bapft_ba\":",
    ))

    # --- 11. consumer: CORA corrector (convex) ------------------------- #
    repl.append((
        "                        omega_net=float(args.mec_omega_net),\n"
        "                        prev_staleness=prev_stale,\n"
        "                        est_success=mec_est_success,",
        "                        omega_net=radio_outer_age_weight,\n"
        "                        prev_staleness=prev_stale,\n"
        "                        est_success=mec_rel_view,",
    ))

    # --- 12. arrival-gate hazards -------------------------------------- #
    repl.append((
        "                duration_probability = (\n"
        "                    1.0 - math.exp(-float(args.mec_duration_hazard_rate) * t_up)",
        "                duration_probability = (\n"
        "                    1.0 - math.exp(-mec_hazard_for_round(round_num) * t_up)",
    ))
    repl.append((
        "                duration_event = mec_trace.duration_failure(\n"
        "                    round_num,\n"
        "                    cid,\n"
        "                    t_up,\n"
        "                    float(args.mec_duration_hazard_rate),\n"
        "                )",
        "                duration_event = mec_trace.duration_failure(\n"
        "                    round_num,\n"
        "                    cid,\n"
        "                    t_up,\n"
        "                    mec_hazard_for_round(round_num),\n"
        "                )",
    ))
    repl.append((
        "                c[\"rift_delivery_probability_corrector\"] = float(\n"
        "                    math.exp(-float(args.mec_duration_hazard_rate) * t_up)",
        "                c[\"rift_delivery_probability_corrector\"] = float(\n"
        "                    math.exp(-mec_hazard_for_round(round_num) * t_up)",
    ))
    repl.append((
        "                    attempted_t_up = min(\n"
        "                        t_up,\n"
        "                        mec_trace.duration_failure_time(\n"
        "                            round_num, cid, float(args.mec_duration_hazard_rate)\n"
        "                        ),\n"
        "                    )",
        "                    attempted_t_up = min(\n"
        "                        t_up,\n"
        "                        mec_trace.duration_failure_time(\n"
        "                            round_num, cid, mec_hazard_for_round(round_num)\n"
        "                        ),\n"
        "                    )",
    ))

    # --- 13. per-round raw/visible state log --------------------------- #
    repl.append((
        "                res.failure_type = (\n"
        "                    \"dropout\" if base_dropped[cid]\n"
        "                    else (\"duration_dropout\" if duration_dropped\n"
        "                          else (\"deadline_miss\" if not arrived else \"none\"))\n"
        "                )\n"
        "                res.dropped = dropped[cid]",
        "                res.failure_type = (\n"
        "                    \"dropout\" if base_dropped[cid]\n"
        "                    else (\"duration_dropout\" if duration_dropped\n"
        "                          else (\"deadline_miss\" if not arrived else \"none\"))\n"
        "                )\n"
        "                res.dropped = dropped[cid]\n"
        "                c[\"failure_type_final\"] = str(res.failure_type)\n"
        "                c[\"arrived_final\"] = bool(arrived)",
    ))

    patch(main_py, repl)

    # insert STATE_VIEWS writer before finish_round (single-use insert)
    patch(
        main_py,
        [(
            "            if wire_controller is not None:\n"
            "                wire_controller.finish_round()",
            "            # FS23: per-round raw vs consumer-visible state record.\n"
            "            try:\n"
            "                _sv_rows = []\n"
            "                for (_c_id, _b_name) in sorted(mec_staleness.alloc.keys()):\n"
            "                    if mec_layer_names and _b_name not in mec_layer_names:\n"
            "                        continue\n"
            "                    _sv_rows.append({\n"
            "                        \"client\": int(_c_id),\n"
            "                        \"block\": _b_name,\n"
            "                        \"raw_alloc\": int(mec_staleness.alloc.get((_c_id, _b_name), 0)),\n"
            "                        \"raw_net\": int(mec_staleness.net.get((_c_id, _b_name), 0)),\n"
            "                        \"raw_effective\": float(mec_staleness.raw_effective(_c_id, _b_name)),\n"
            "                        \"visible_effective\": float(mec_staleness.effective(_c_id, _b_name)),\n"
            "                    })\n"
            "                _sv_fail = {}\n"
            "                for _c in mec_round_info:\n"
            "                    _ft = str(_c.get(\"failure_type_final\", \"unknown\"))\n"
            "                    _sv_fail[_ft] = _sv_fail.get(_ft, 0) + 1\n"
            "                with open(os.path.join(mec_run_output_dir, \"STATE_VIEWS.jsonl\"), \"a\",\n"
            "                          encoding=\"utf-8\") as _sv_fh:\n"
            "                    _sv_fh.write(json.dumps({\n"
            "                        \"round\": int(round_num),\n"
            "                        \"reliability_raw\": {\n"
            "                            str(k): float(v) for k, v in mec_est_success.items()\n"
            "                        },\n"
            "                        \"reliability_visible\": {\n"
            "                            str(k): float(mec_rel_view[k]) for k in mec_est_success\n"
            "                        },\n"
            "                        \"age_rows\": _sv_rows,\n"
            "                        \"failure_counts\": _sv_fail,\n"
            "                    }) + \"\\n\")\n"
            "            except Exception as _sv_exc:\n"
            "                mec_log(f\"STATE_VIEWS write failed: {_sv_exc}\")\n"
            "            if wire_controller is not None:\n"
            "                wire_controller.finish_round()",
        )],
    )

    # compile check
    import py_compile
    py_compile.compile(main_py, doraise=True)
    py_compile.compile(staleness_py, doraise=True)
    py_compile.compile(traces_py, doraise=True)
    print("compile OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
