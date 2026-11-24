#!/usr/bin/env python
"""S3 engineering fix round 1 for I1/I2 (REFINE).

First screening run exposed an engineering defect: --lpc_score_threshold=0 and
--lpc_round_budget=0 made predictive/value_budget trigger EVERY scored block
(lpc events 884 vs baseline 38). Fix: make the budget GLOBAL per round.

  value_budget: rank ALL (client,block) candidates by score, keep top-K globally
                (K=--lpc_round_budget). K<=0 -> no candidates (fail-safe, NOT all).
  predictive  : score > threshold as before; if --lpc_round_budget>0 the same
                global top-K prune applies afterwards.

Usage: python patch_ideas_s3_fix1.py <code_dir>
"""

from __future__ import annotations

import os
import shutil
import sys


def main() -> int:
    code = sys.argv[1] if len(sys.argv) > 1 else "."
    main_py = os.path.join(code, "main.py")
    with open(main_py, "r", encoding="utf-8") as fh:
        text = fh.read()

    old_loop = '''                if lpc_applicable:
                    _trig = str(getattr(args, "lpc_trigger", "age"))
                    _alloc = allocation_map.get(cid, set())
                    for _n in mec_layer_names:
                        _prev = float(fs23_alloc_freq.get((cid, _n), 0.0))
                        fs23_alloc_freq[(cid, _n)] = (
                            0.8 * _prev + 0.2 * (1.0 if _n in _alloc else 0.0)
                        )
                    if _trig == "age":
                        mec_lpc_mask[cid] = lpc_trigger_mask(
                            mec_staleness, cid, _alloc, b_train_every
                        )
                    else:
                        _scored = []
                        for _n in _alloc:
                            _age = mec_staleness.effective(cid, _n)
                            if int(round(_age)) < 1:
                                continue
                            _dn = float(fs23_last_delta_norm.get((cid, _n), 0.0))
                            _rf = float(fs23_alloc_freq.get((cid, _n), 0.0))
                            _scored.append((_dn * _rf, _n))
                        _scored.sort(reverse=True)
                        if _trig == "value_budget":
                            _k = int(getattr(args, "lpc_round_budget", 0))
                            mec_lpc_mask[cid] = (
                                {n for _, n in _scored[:max(0, _k)]}
                                if _k > 0
                                else {n for s, n in _scored if s > 0.0}
                            )
                        else:
                            _thr = float(getattr(args, "lpc_score_threshold", 0.0))
                            mec_lpc_mask[cid] = {
                                n for s, n in _scored if s > _thr
                            }
                else:
                    mec_lpc_mask[cid] = set()'''
    new_loop = '''                if lpc_applicable:
                    _trig = str(getattr(args, "lpc_trigger", "age"))
                    _alloc = allocation_map.get(cid, set())
                    for _n in mec_layer_names:
                        _prev = float(fs23_alloc_freq.get((cid, _n), 0.0))
                        fs23_alloc_freq[(cid, _n)] = (
                            0.8 * _prev + 0.2 * (1.0 if _n in _alloc else 0.0)
                        )
                    if _trig == "age":
                        mec_lpc_mask[cid] = lpc_trigger_mask(
                            mec_staleness, cid, _alloc, b_train_every
                        )
                    else:
                        _scored = []
                        for _n in _alloc:
                            _age = mec_staleness.effective(cid, _n)
                            if int(round(_age)) < 1:
                                continue
                            _dn = float(fs23_last_delta_norm.get((cid, _n), 0.0))
                            _rf = float(fs23_alloc_freq.get((cid, _n), 0.0))
                            if _dn * _rf > 0.0:
                                _scored.append((_dn * _rf, cid, _n))
                        if _trig == "value_budget":
                            mec_lpc_mask[cid] = {n for _, _, n in _scored}
                        else:
                            _thr = float(getattr(args, "lpc_score_threshold", 0.0))
                            mec_lpc_mask[cid] = {
                                n for s, _, n in _scored if s > _thr
                            }
                        fs23_score_pool.extend(_scored)
                else:
                    mec_lpc_mask[cid] = set()'''
    n = text.count(old_loop)
    assert n == 1, f"loop anchor {n}x"
    text = text.replace(old_loop, new_loop)

    # score pool init + global budget prune
    old_head = '''        if mec_enabled:
            for cid in range(args.num_clients):
                # LPC is a FedHARP-family mechanism.'''
    new_head = '''        if mec_enabled:
            fs23_score_pool = []
            for cid in range(args.num_clients):
                # LPC is a FedHARP-family mechanism.'''
    n = text.count(old_head)
    assert n == 1, f"head anchor {n}x"
    text = text.replace(old_head, new_head)

    old_tail = '''                else:
                    mec_lpc_mask[cid] = set()
        
        # 步骤 3: 客户端接收全局 A 并执行本地训练'''
    new_tail = '''                else:
                    mec_lpc_mask[cid] = set()
            # S3 fix1: GLOBAL per-round correction budget over all candidates.
            _k = int(getattr(args, "lpc_round_budget", 0))
            if _k > 0 and fs23_score_pool:
                fs23_score_pool.sort(reverse=True)
                _keep = {(c, n) for _, c, n in fs23_score_pool[:_k]}
                for _cid in range(args.num_clients):
                    mec_lpc_mask[_cid] = {
                        _n for _n in mec_lpc_mask.get(_cid, set())
                        if (_cid, _n) in _keep
                    }
            elif (
                str(getattr(args, "lpc_trigger", "age")) == "value_budget"
                and _k <= 0
            ):
                # fail-safe: value_budget without K selects nothing (the first
                # screening run showed "no budget == trigger all" is an
                # explosion, not a default).
                for _cid in range(args.num_clients):
                    mec_lpc_mask[_cid] = set()

        # 步骤 3: 客户端接收全局 A 并执行本地训练'''
    n = text.count(old_tail)
    assert n == 1, f"tail anchor {n}x"
    text = text.replace(old_tail, new_tail)

    backup = main_py + ".bak_ideas_fix1"
    if not os.path.exists(backup):
        shutil.copy2(main_py, backup)
    with open(main_py, "w", encoding="utf-8") as fh:
        fh.write(text)
    import py_compile
    py_compile.compile(main_py, doraise=True)
    print("patched", main_py)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
