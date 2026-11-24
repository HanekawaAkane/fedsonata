#!/usr/bin/env python
"""S3 idea patches: I1 PREDICT-LPC / I2 VS-LPC / I3 DP-RIFT (minimal versions).

Adds three independent switches (defaults keep baseline bit-identical):
  --lpc_trigger {age,predictive,value_budget}
      age          : legacy round(age)==b_train_every (default)
      predictive   : I1 -- score = last-delta-norm x alloc-freq-EMA > threshold
      value_budget : I2 -- top-K by the same score per round (K=--lpc_round_budget)
  --lpc_score_threshold FLOAT (default 0.0)
  --lpc_round_budget INT (default 0 = no cap)
  --cap_policy {linear,expected_delivery}
      linear             : legacy cap=full*ratio*max(floor, reliability) (default)
      expected_delivery  : I3 -- reliability * exp(-hazard * full_bytes*8/rate)

Usage: python patch_ideas_s3.py <code_dir>
"""

from __future__ import annotations

import os
import shutil
import sys


def patch(path, replacements):
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    for old, new in replacements:
        n = text.count(old)
        assert n == 1, f"anchor {n}x in {path}:\n{old[:180]}"
        text = text.replace(old, new)
    backup = path + ".bak_ideas_s3"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    import py_compile
    py_compile.compile(path, doraise=True)
    print("patched", path)


def main() -> int:
    code = sys.argv[1] if len(sys.argv) > 1 else "."
    main_py = os.path.join(code, "main.py")
    wire_py = os.path.join(code, "wire_codec.py")

    # ---------------- main.py ---------------- #
    repl = []

    # 1. parser
    repl.append((
        '    parser.add_argument("--network_scenario", type=str, default="base",\n'
        '                        choices=["base", "ideal", "shift"],\n'
        '                        help="预设网络条件：base=主协议；ideal=无中断校验；"\n'
        '                             "shift=轮11-15 GE/hazard 压力段 (默认: base)")',
        '    parser.add_argument("--network_scenario", type=str, default="base",\n'
        '                        choices=["base", "ideal", "shift"],\n'
        '                        help="预设网络条件：base=主协议；ideal=无中断校验；"\n'
        '                             "shift=轮11-15 GE/hazard 压力段 (默认: base)")\n'
        '    # ---- S3 idea switches (2026-11-18). Defaults = baseline behaviour.\n'
        '    parser.add_argument("--lpc_trigger", type=str, default="age",\n'
        '                        choices=["age", "predictive", "value_budget"],\n'
        '                        help="LPC 触发语义：age=基线等值触发；predictive=I1 期望边际收益；"\n'
        '                             "value_budget=I2 预算内按分数取前 K (默认: age)")\n'
        '    parser.add_argument("--lpc_score_threshold", type=float, default=0.0,\n'
        '                        help="I1 分数阈值（delta 范数 × 分配频率）(默认: 0)")\n'
        '    parser.add_argument("--lpc_round_budget", type=int, default=0,\n'
        '                        help="I2 每客户端每轮校正预算 K；0 表示不限 (默认: 0)")\n'
        '    parser.add_argument("--cap_policy", type=str, default="linear",\n'
        '                        choices=["linear", "expected_delivery"],\n'
        '                        help="RIFT byte cap 策略：linear=基线；expected_delivery=I3 期望交付定价 (默认: linear)")',
    ))

    # 2. state dicts init near EffectiveStaleness creation
    repl.append((
        "        mec_staleness = EffectiveStaleness(\n"
        "            omega_net=age_delivery_weight,\n"
        "            age_view=str(getattr(args, \"age_view\", \"dual\")),\n"
        "        )",
        "        mec_staleness = EffectiveStaleness(\n"
        "            omega_net=age_delivery_weight,\n"
        "            age_view=str(getattr(args, \"age_view\", \"dual\")),\n"
        "        )\n"
        "        fs23_last_delta_norm = {}\n"
        "        fs23_alloc_freq = {}",
    ))

    # 3. LPC mask dispatcher
    repl.append((
        "                mec_lpc_mask[cid] = lpc_trigger_mask(\n"
        "                    mec_staleness, cid, allocation_map.get(cid, set()), b_train_every\n"
        "                ) if lpc_applicable else set()",
        "                if lpc_applicable:\n"
        "                    _trig = str(getattr(args, \"lpc_trigger\", \"age\"))\n"
        "                    _alloc = allocation_map.get(cid, set())\n"
        "                    for _n in mec_layer_names:\n"
        "                        _prev = float(fs23_alloc_freq.get((cid, _n), 0.0))\n"
        "                        fs23_alloc_freq[(cid, _n)] = (\n"
        "                            0.8 * _prev + 0.2 * (1.0 if _n in _alloc else 0.0)\n"
        "                        )\n"
        "                    if _trig == \"age\":\n"
        "                        mec_lpc_mask[cid] = lpc_trigger_mask(\n"
        "                            mec_staleness, cid, _alloc, b_train_every\n"
        "                        )\n"
        "                    else:\n"
        "                        _scored = []\n"
        "                        for _n in _alloc:\n"
        "                            _age = mec_staleness.effective(cid, _n)\n"
        "                            if int(round(_age)) < 1:\n"
        "                                continue\n"
        "                            _dn = float(fs23_last_delta_norm.get((cid, _n), 0.0))\n"
        "                            _rf = float(fs23_alloc_freq.get((cid, _n), 0.0))\n"
        "                            _scored.append((_dn * _rf, _n))\n"
        "                        _scored.sort(reverse=True)\n"
        "                        if _trig == \"value_budget\":\n"
        "                            _k = int(getattr(args, \"lpc_round_budget\", 0))\n"
        "                            mec_lpc_mask[cid] = (\n"
        "                                {n for _, n in _scored[:max(0, _k)]}\n"
        "                                if _k > 0\n"
        "                                else {n for s, n in _scored if s > 0.0}\n"
        "                            )\n"
        "                        else:\n"
        "                            _thr = float(getattr(args, \"lpc_score_threshold\", 0.0))\n"
        "                            mec_lpc_mask[cid] = {\n"
        "                                n for s, n in _scored if s > _thr\n"
        "                            }\n"
        "                else:\n"
        "                    mec_lpc_mask[cid] = set()",
    ))

    # 4. record delta norms after wire selection
    repl.append((
        "                rift_selection_ms = (time.perf_counter() - rift_t0) * 1000.0\n"
        "                delta_A = rift_selection.updates_A\n"
        "                delta_B = rift_selection.updates_B",
        "                rift_selection_ms = (time.perf_counter() - rift_t0) * 1000.0\n"
        "                delta_A = rift_selection.updates_A\n"
        "                delta_B = rift_selection.updates_B\n"
        "                for _n, _t in (selection_delta_A or {}).items():\n"
        "                    fs23_last_delta_norm[(client_idx, _n)] = float(\n"
        "                        _t.detach().float().norm()\n"
        "                    )\n"
        "                for _n, _t in (selection_delta_B or {}).items():\n"
        "                    fs23_last_delta_norm[(client_idx, _n)] = max(\n"
        "                        float(fs23_last_delta_norm.get((client_idx, _n), 0.0)),\n"
        "                        float(_t.detach().float().norm()),\n"
        "                    )",
    ))

    # 5. pass cap hazard unconditionally to wire select
    repl.append((
        "                    delivery_rate_bps=rift_delivery_rate_bps,\n"
        "                    duration_hazard_rate=(\n"
        "                        float(mec_hazard_for_round(round_num))\n"
        "                        if rift_duration_value_enabled else 0.0\n"
        "                    ),",
        "                    delivery_rate_bps=rift_delivery_rate_bps,\n"
        "                    cap_hazard_rate=float(mec_hazard_for_round(round_num)),\n"
        "                    duration_hazard_rate=(\n"
        "                        float(mec_hazard_for_round(round_num))\n"
        "                        if rift_duration_value_enabled else 0.0\n"
        "                    ),",
    ))

    patch(main_py, repl)

    # ---------------- wire_codec.py ---------------- #
    patch(wire_py, [(
        "        full_bytes = sum(t.numel() * t.element_size() for maps in (delta_a, delta_b) for t in maps.values())\n"
        "        cap = int(math.floor(full_bytes * budget_ratio * max(reliability_floor, reliability)))",
        "        full_bytes = sum(t.numel() * t.element_size() for maps in (delta_a, delta_b) for t in maps.values())\n"
        "        # S3 I3 (DP-RIFT): expected-delivery pricing of the byte cap.\n"
        "        _cap_signal = float(reliability)\n"
        "        if str(getattr(self.args, \"cap_policy\", \"linear\")) == \"expected_delivery\":\n"
        "            _rate = float(kwargs.get(\"delivery_rate_bps\") or 0.0)\n"
        "            _hz = float(kwargs.get(\"cap_hazard_rate\") or 0.0)\n"
        "            if _rate > 0.0 and math.isfinite(_rate) and _hz >= 0.0:\n"
        "                _t_est = (full_bytes * 8.0) / _rate\n"
        "                _cap_signal = float(reliability) * math.exp(-_hz * _t_est)\n"
        "        cap = int(math.floor(full_bytes * budget_ratio * max(reliability_floor, _cap_signal)))",
    ), (
        "            reduced_ratio = max(1e-8, (cap - header_reserve) / (full_bytes * max(reliability_floor, reliability)))",
        "            reduced_ratio = max(1e-8, (cap - header_reserve) / (full_bytes * max(reliability_floor, _cap_signal)))",
    )])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
