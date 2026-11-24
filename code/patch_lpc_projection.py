"""Patch FedSONATA main.py: LPC mechanism gains a projection mode.

Adds:
  * --lpc_mode {train,projection} and --projection_ridge CLI flags
  * pre-sync snapshot of the retained personal factor pair
  * LPC branch: closed-form FedHARP Local Projection Corrector transport
    (projection mode) instead of the legacy task-loss B-only / A-only pass
  * lpc_projection_event logging + end-of-run summary
  * run_id / config.json provenance for the new mode

Usage: python patch_lpc_projection.py <path-to-main.py>
Idempotent: refuses to patch a file that already contains the marker.
"""

from __future__ import annotations

import os
import shutil
import sys

MARKER = "lpc_projection_enabled"

ARGPARSE_ANCHOR = (
    '    parser.add_argument("--b_num_epochs", type=int, default=1,\n'
    '                        help="B-only 阶段本地训练的 Epoch 数 (默认: 1)")\n'
)
ARGPARSE_NEW = ARGPARSE_ANCHOR + (
    '    parser.add_argument("--lpc_mode", type=str, default="train",\n'
    '                        choices=["train", "projection"],\n'
    '                        help="LPC 机制：train=遗留 B-only/A-only 本地修正；"\n'
    '                             "projection=FedHARP 闭式 Local Projection Corrector 输运 "\n'
    '                             "(默认: train)")\n'
    '    parser.add_argument("--projection_ridge", type=float, default=1e-6,\n'
    '                        help="LPC projection 的 Tikhonov ridge (默认: 1e-6)")\n'
)

IMPORT_ANCHOR = (
    "from mec.early_stopping import ValidationEarlyStopper\n"
    "from mec.fedhello import empirical_fim_scores, should_refresh_fim\n"
)
IMPORT_NEW = IMPORT_ANCHOR + (
    "from factor_projection import project_factor_dicts, project_factor_dicts_A\n"
)

INIT_ANCHOR = (
    '    b_train_every = int(args.b_train_every) if (is_fedharp_family and method_l != "fedharp_l") else 0\n'
)
INIT_NEW = INIT_ANCHOR + (
    "    lpc_projection_enabled = (\n"
    '        is_fedharp_family\n'
    '        and b_train_every > 0\n'
    '        and str(getattr(args, "lpc_mode", "train")).lower() == "projection"\n'
    "    )\n"
    "    lpc_projection_events = 0\n"
)

CONFIG_ANCHOR = (
    '                    "strict_determinism": bool(args.strict_determinism),\n'
    '                    "fedra_scope": "a" if fedra_a_only else "native",\n'
)
CONFIG_NEW = (
    '                    "strict_determinism": bool(args.strict_determinism),\n'
    '                    "lpc_mode": str(getattr(args, "lpc_mode", "train")),\n'
    '                    "projection_ridge": (\n'
    '                        float(args.projection_ridge)\n'
    '                        if str(getattr(args, "lpc_mode", "train")).lower() == "projection"\n'
    '                        else None\n'
    '                    ),\n'
    '                    "fedra_scope": "a" if fedra_a_only else "native",\n'
)

RUNID_ANCHOR = (
    "        run_id = (\n"
    "            f\"mec_{args.mec_method or 'custom'}_\"\n"
    "            f\"{args.dataset}_{args.num_clients}cl_{args.mec_channel}_\"\n"
    "            f\"dr{args.mec_dropout_rate}_bw{args.mec_total_bw_mhz}_s{args.seed}\"\n"
    "        )\n"
)
RUNID_NEW = (
    "        lpc_mode_tag = (\n"
    '            "_lpcproj"\n'
    '            if str(getattr(args, "lpc_mode", "train")).lower() == "projection"\n'
    '            else ""\n'
    "        )\n"
    "        run_id = (\n"
    "            f\"mec_{args.mec_method or 'custom'}_\"\n"
    "            f\"{args.dataset}_{args.num_clients}cl_{args.mec_channel}_\"\n"
    "            f\"dr{args.mec_dropout_rate}_bw{args.mec_total_bw_mhz}_s{args.seed}\"\n"
    "            f\"{lpc_mode_tag}\"\n"
    "        )\n"
)

SNAPSHOT_ANCHOR = (
    "            client_global_A = global_A\n"
    "            client_global_B = global_B\n"
    "            fedharp_b_sharing = is_fedharp_family and fedharp_matrix_mode == \"b\"\n"
)
SNAPSHOT_NEW = (
    "            # Snapshot the retained factor state before synchronization for\n"
    "            # projection-mode LPC (FedHARP Local Projection Corrector). The\n"
    "            # explicit per-client snapshots are used because the live module\n"
    "            # may hold a restored evaluation state.\n"
    "            lpc_projection_old_A = None\n"
    "            lpc_projection_old_B = None\n"
    "            if lpc_projection_enabled:\n"
    "                lpc_projection_old_A = {\n"
    "                    name: tensor.clone()\n"
    "                    for name, tensor in client_personal_A_states.get(\n"
    "                        client_idx, client.get_all_A_matrices()\n"
    "                    ).items()\n"
    "                }\n"
    "                lpc_projection_old_B = {\n"
    "                    name: tensor.clone()\n"
    "                    for name, tensor in client_personal_B_states.get(\n"
    "                        client_idx, client.get_all_B_matrices()\n"
    "                    ).items()\n"
    "                }\n"
) + SNAPSHOT_ANCHOR

LPC_ANCHOR = (
    "                if len(stale_layers) > 0:\n"
    "                    if fedharp_matrix_mode == \"b\":\n"
    "                        # Symmetric LPC realigns stale local A against the\n"
    "                        # newly downloaded global B. DSA remains based on the\n"
    "                        # B-gradient measured during standard local training.\n"
    "                        client.train_A_only(stale_layers, num_epochs=args.b_num_epochs)\n"
    "                    else:\n"
    "                        new_b_sensitivity = client.train_B_only(\n"
    "                            stale_layers,\n"
    "                            num_epochs=args.b_num_epochs,\n"
    "                        )\n"
    "                        server.update_client_sensitivities(client_idx, new_b_sensitivity)\n"
)
LPC_NEW = (
    "                if len(stale_layers) > 0:\n"
    "                    if lpc_projection_enabled:\n"
    "                        # Closed-form FedHARP Local Projection Corrector:\n"
    "                        # transport the retained factor pair across the new\n"
    "                        # basis instead of running a task-loss correction pass.\n"
    "                        if fedharp_matrix_mode == \"b\":\n"
    "                            projected_factors, lpc_diag = project_factor_dicts_A(\n"
    "                                lpc_projection_old_A,\n"
    "                                lpc_projection_old_B,\n"
    "                                client.get_all_A_matrices(),\n"
    "                                client.get_all_B_matrices(),\n"
    "                                stale_layers,\n"
    "                                ridge=float(args.projection_ridge),\n"
    "                            )\n"
    "                            client.set_A_matrices(projected_factors)\n"
    "                        else:\n"
    "                            projected_factors, lpc_diag = project_factor_dicts(\n"
    "                                lpc_projection_old_A,\n"
    "                                lpc_projection_old_B,\n"
    "                                client.get_all_A_matrices(),\n"
    "                                client.get_all_B_matrices(),\n"
    "                                stale_layers,\n"
    "                                ridge=float(args.projection_ridge),\n"
    "                            )\n"
    "                            client.set_B_matrices(projected_factors)\n"
    "                        lpc_projection_events += len(lpc_diag)\n"
    "                        for _lpc_layer, _lpc_d in sorted(lpc_diag.items()):\n"
    "                            print(\n"
    "                                f\"round={round_num}, type=lpc_projection_event, \"\n"
    "                                f\"client={client_idx}, layer={_lpc_layer}, \"\n"
    "                                f\"mode={fedharp_matrix_mode}, \"\n"
    "                                f\"ridge={float(args.projection_ridge):.8g}, \"\n"
    "                                f\"residual_before={float(_lpc_d['residual_before']):.8g}, \"\n"
    "                                f\"residual_after={float(_lpc_d['residual_after']):.8g}, \"\n"
    "                                f\"relative_reduction={float(_lpc_d['relative_reduction']):.8g}\"\n"
    "                            )\n"
    "                    elif fedharp_matrix_mode == \"b\":\n"
    "                        # Symmetric LPC realigns stale local A against the\n"
    "                        # newly downloaded global B. DSA remains based on the\n"
    "                        # B-gradient measured during standard local training.\n"
    "                        client.train_A_only(stale_layers, num_epochs=args.b_num_epochs)\n"
    "                    else:\n"
    "                        new_b_sensitivity = client.train_B_only(\n"
    "                            stale_layers,\n"
    "                            num_epochs=args.b_num_epochs,\n"
    "                        )\n"
    "                        server.update_client_sensitivities(client_idx, new_b_sensitivity)\n"
)

SUMMARY_ANCHOR = (
    '    print(f"[总结] 历史最佳选择指标: {best_accuracy:.2f}%")\n'
)
SUMMARY_NEW = SUMMARY_ANCHOR + (
    '    if str(getattr(args, "lpc_mode", "train")).lower() == "projection":\n'
    '        print(f"[总结] LPC projection events={lpc_projection_events}")\n'
)

REPLACEMENTS = [
    ("argparse", ARGPARSE_ANCHOR, ARGPARSE_NEW),
    ("import", IMPORT_ANCHOR, IMPORT_NEW),
    ("init", INIT_ANCHOR, INIT_NEW),
    ("config", CONFIG_ANCHOR, CONFIG_NEW),
    ("run_id", RUNID_ANCHOR, RUNID_NEW),
    ("snapshot", SNAPSHOT_ANCHOR, SNAPSHOT_NEW),
    ("lpc_branch", LPC_ANCHOR, LPC_NEW),
    ("summary", SUMMARY_ANCHOR, SUMMARY_NEW),
]


def apply(path: str) -> None:
    with open(path, "rb") as fh:
        raw = fh.read()
    newline = "\r\n" if b"\r\n" in raw else "\n"
    text = raw.decode("utf-8").replace("\r\n", "\n")

    if MARKER in text:
        print(f"SKIP already patched: {path}")
        return

    backup = path + ".bak_lpcproj_20261028"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f"backup -> {backup}")

    for name, old, new in REPLACEMENTS:
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"FAIL [{name}]: anchor count={count} (want 1) in {path}")
        text = text.replace(old, new, 1)
        print(f"patched [{name}]")

    out = text.replace("\n", newline) if newline == "\r\n" else text
    with open(path, "wb") as fh:
        fh.write(out.encode("utf-8"))
    print(f"OK {path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python patch_lpc_projection.py <main.py>")
    apply(sys.argv[1])
