#!/usr/bin/env python
"""FS23 endpoint patcher: separate final-round vs validation-selected holdout.

PROTOCOL_CN: "第 20 轮是主终点，另保存按验证集选取的 checkpoint；选取结果与
最终轮结果分列，holdout 不参与选择。旧 restore_best_validation 输出不能直接
充当第 20 轮结果。"  The stock code evaluates holdout ONCE on the restored
best-validation model. This patch:

  1. captures a full ``final_round_snapshot`` before any restore,
  2. runs the (unchanged) personalised holdout evaluation once per phase:
     ``final_round`` then ``validation_selected``,
  3. records ``accuracy_final_round`` / ``accuracy_validation_selected`` /
     ``fs23_phase_holdout`` in final_metrics.json while keeping the legacy
     ``accuracy`` field meaning (validation-selected when restore is on),
  4. persists both endpoint checkpoints to disk.

Usage: python patch_fs23_endpoints.py <code_dir>
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

    # ------------------------------------------------------------------ #
    # 1. Replace the restore block + eval header with snapshot capture and
    #    a two-phase evaluation loop.  The eval body is re-indented +4.
    # ------------------------------------------------------------------ #
    old_header = '''    completed_rounds = int(round_num)
    restored_best_validation = False
    if (
        validation_enabled
        and bool(args.restore_best_validation)
        and validation_snapshot is not None
    ):
        server.set_global_A_matrices(validation_snapshot["global_A"])
        if validation_snapshot["global_B"] is not None:
            server.set_global_B_matrices(validation_snapshot["global_B"])
        client_personal_A_states = validation_snapshot["personal_A"]
        client_personal_B_states = validation_snapshot["personal_B"]
        client_personal_classifier_states = validation_snapshot["personal_classifier"]
        global_classifier_state = validation_snapshot["classifier"]
        _set_classifier_state(global_model, global_classifier_state)
        restored_best_validation = True

    # One reporting-only evaluation after model selection.  No value produced
    # here can alter the checkpoint or stopping decision.
    final_test_accuracy = 0.0
    final_test_loss = 0.0
    final_test_samples = 0
    final_test_dataset = "official_test" if (
        args.evaluation_scope == "server_global_test"
    ) else "personalized_client_holdout"
    final_global_A = server.get_global_A_matrices()
    final_global_B = server.get_global_B_matrices() if upload_b_method else None
'''
    new_header = '''    completed_rounds = int(round_num)
    restored_best_validation = False
    # FS23: capture the FINAL-ROUND state before any restore so that the
    # round-20 endpoint and the validation-selected endpoint are reported
    # separately (PROTOCOL_CN: legacy restore_best_validation output must NOT
    # stand in for the round-20 result).
    final_round_snapshot = {
        "global_A": {
            k: v.detach().clone()
            for k, v in server.get_global_A_matrices().items()
        },
        "global_B": (
            {
                k: v.detach().clone()
                for k, v in server.get_global_B_matrices().items()
            }
            if upload_b_method else None
        ),
        "classifier": _get_classifier_state(global_model),
        "personal_A": {
            cid: {k: v.detach().clone() for k, v in state.items()}
            for cid, state in client_personal_A_states.items()
        },
        "personal_B": {
            cid: {k: v.detach().clone() for k, v in state.items()}
            for cid, state in client_personal_B_states.items()
        },
        "personal_classifier": {
            cid: {k: v.detach().clone() for k, v in state.items()}
            for cid, state in client_personal_classifier_states.items()
        },
    }
    _fs23_eval_phases = [("final_round", final_round_snapshot)]
    if (
        validation_enabled
        and bool(args.restore_best_validation)
        and validation_snapshot is not None
    ):
        _fs23_eval_phases.append(("validation_selected", validation_snapshot))
        restored_best_validation = True

    # FS23: reporting-only holdout evaluation, once per phase (final-round
    # model first, then the restored validation-selected model).  No value
    # produced here can alter the checkpoint or stopping decision.
    _fs23_phase_metrics = {}
    for _fs23_phase_name, _fs23_snap in _fs23_eval_phases:
        server.set_global_A_matrices(_fs23_snap["global_A"])
        if _fs23_snap.get("global_B") is not None:
            server.set_global_B_matrices(_fs23_snap["global_B"])
        client_personal_A_states = _fs23_snap["personal_A"]
        client_personal_B_states = _fs23_snap["personal_B"]
        client_personal_classifier_states = _fs23_snap["personal_classifier"]
        global_classifier_state = _fs23_snap["classifier"]
        _set_classifier_state(global_model, global_classifier_state)
        final_test_accuracy = 0.0
        final_test_loss = 0.0
        final_test_samples = 0
        final_test_dataset = "official_test" if (
            args.evaluation_scope == "server_global_test"
        ) else "personalized_client_holdout"
        final_global_A = _fs23_snap["global_A"]
        final_global_B = _fs23_snap.get("global_B")
'''
    n = text.count(old_header)
    assert n == 1, f"restore/eval header appears {n}x"
    text = text.replace(old_header, new_header)

    # ------------------------------------------------------------------ #
    # 2. Re-indent the eval body +4 and close the phase loop after it.
    # ------------------------------------------------------------------ #
    body_start = '''    if args.evaluation_scope == "server_global_test":
        for name, layer in global_lora_layers.items():
            if name in final_global_A:
                layer.set_A(final_global_A[name])
            if final_global_B is not None and name in final_global_B:
                layer.set_B(final_global_B[name])
        _set_classifier_state(global_model, global_classifier_state)
'''
    body_end = '''        if final_test_samples > 0:
            final_test_accuracy = 100.0 * final_correct / final_test_samples
            final_test_loss = final_loss_sum / final_test_samples
'''
    i0 = text.find(body_start)
    assert i0 != -1, "eval body start not found"
    assert text.count(body_start) == 1, "eval body start not unique"
    i1 = text.find(body_end, i0)
    assert i1 != -1, "eval body end not found"
    assert text.count(body_end) == 1, "eval body end not unique"
    i1_end = i1 + len(body_end)
    body = text[i0:i1_end]
    indented = "\n".join(
        ("    " + ln) if ln.strip() else ln for ln in body.split("\n")
    )
    closer = '''        _fs23_phase_metrics[_fs23_phase_name] = {
            "accuracy": float(final_test_accuracy) / 100.0,
            "loss": float(final_test_loss),
            "num_samples": int(final_test_samples),
            "test_dataset": str(final_test_dataset),
        }
    # FS23: legacy `accuracy` keeps its meaning -- the restored
    # validation-selected holdout when restore is on, else the final round.
    _fs23_default_phase = (
        "validation_selected"
        if "validation_selected" in _fs23_phase_metrics
        else "final_round"
    )
    final_test_accuracy = (
        _fs23_phase_metrics[_fs23_default_phase]["accuracy"] * 100.0
    )
    final_test_loss = _fs23_phase_metrics[_fs23_default_phase]["loss"]
    final_test_samples = _fs23_phase_metrics[_fs23_default_phase]["num_samples"]
    for _fs23_phase_name, _fs23_pm in _fs23_phase_metrics.items():
        print(
            f"[FS23] holdout[{_fs23_phase_name}] "
            f"accuracy={_fs23_pm['accuracy'] * 100.0:.2f}% "
            f"loss={_fs23_pm['loss']:.4f} samples={_fs23_pm['num_samples']}"
        )
'''
    text = text[:i0] + indented + closer + text[i1_end:]

    # ------------------------------------------------------------------ #
    # 3. final_metrics_record: explicit endpoint fields.
    # ------------------------------------------------------------------ #
    old_record = '''        "completed_rounds": completed_rounds,
        "accuracy": float(final_test_accuracy) / 100.0,
        "loss": float(final_test_loss),
        "num_samples": int(final_test_samples),
    }
'''
    new_record = '''        "completed_rounds": completed_rounds,
        "accuracy": float(final_test_accuracy) / 100.0,
        "loss": float(final_test_loss),
        "num_samples": int(final_test_samples),
        "fs23_phase_holdout": _fs23_phase_metrics,
        "accuracy_final_round": (
            _fs23_phase_metrics.get("final_round", {}) or {}
        ).get("accuracy"),
        "accuracy_validation_selected": (
            _fs23_phase_metrics.get("validation_selected", {}) or {}
        ).get("accuracy"),
    }
'''
    n = text.count(old_record)
    assert n == 1, f"final_metrics_record block appears {n}x"
    text = text.replace(old_record, new_record)

    # ------------------------------------------------------------------ #
    # 4. Persist both endpoint checkpoints.
    # ------------------------------------------------------------------ #
    old_tail = '''        if stopped_early:
            with open(
                os.path.join(mec_run_output_dir, "EARLY_STOPPED"), "w", encoding="utf-8"
            ) as fh:
                fh.write(
                    f"completed_rounds={completed_rounds} best_round={best_validation_round}\\n"
                )
'''
    new_tail = old_tail + '''        # FS23: persist both endpoint checkpoints (PROTOCOL_CN delivery checks).
        try:
            _ckpt_dir = (
                args.checkpoint_dir if args.save_checkpoints else mec_run_output_dir
            )
            os.makedirs(_ckpt_dir, exist_ok=True)
            _torch = __import__("torch")
            _torch.save(
                final_round_snapshot,
                os.path.join(_ckpt_dir, "checkpoint_final_round.pt"),
            )
            if validation_snapshot is not None:
                _torch.save(
                    validation_snapshot,
                    os.path.join(_ckpt_dir, "checkpoint_validation_selected.pt"),
                )
            mec_log(
                "FS23 endpoint checkpoints saved: "
                f"final_round + validation_selected -> {_ckpt_dir}"
            )
        except Exception as _ckpt_exc:
            mec_log(f"FS23 endpoint checkpoint save failed: {_ckpt_exc}")
'''
    n = text.count(old_tail)
    assert n == 1, f"EARLY_STOPPED tail appears {n}x"
    text = text.replace(old_tail, new_tail)

    backup = main_py + ".bak_endpoints_fs23"
    if not os.path.exists(backup):
        shutil.copy2(main_py, backup)
    with open(main_py, "w", encoding="utf-8") as fh:
        fh.write(text)
    import py_compile
    py_compile.compile(main_py, doraise=True)
    print("patched", main_py, "compile OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
