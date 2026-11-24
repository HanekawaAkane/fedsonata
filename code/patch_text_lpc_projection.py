r"""Patch the gsm8k/pubmedqa embedded template: LPC gains a projection mode.

Inserts four replace_once_checked calls after the "gsm8k-lpc-before-ordinary-training"
entry of the patch chain in main.py:

  gsm8k-lpc-projection-args   -- add --lpc_mode/--projection_ridge to the template CLI
  gsm8k-lpc-projection-aux    -- track client_A_states alongside client_B_states
  gsm8k-lpc-projection-branch -- closed-form FedHARP transport replaces the B-only pass
  gsm8k-lpc-projection-asave  -- snapshot A after ordinary training

Usage: python patch_text_lpc_projection.py <path-to-main.py>
"""

from __future__ import annotations

import os
import shutil
import sys

MARKER = "gsm8k-lpc-projection-branch"

ANCHOR = (
    '            "gsm8k-lpc-before-ordinary-training",\n'
    '        )\n'
)

INSERT = r'''        qwen = replace_once_checked(
            qwen,
            '    p.add_argument("--save_final_state", action=argparse.BooleanOptionalAction, default=True)\n',
            '    p.add_argument("--save_final_state", action=argparse.BooleanOptionalAction, default=True)\n'
            '    p.add_argument("--lpc_mode", type=str, default="train", choices=["train", "projection"])\n'
            '    p.add_argument("--projection_ridge", type=float, default=1e-6)\n',
            "gsm8k-lpc-projection-args",
        )
        qwen = replace_once_checked(
            qwen,
            '    client_B_states: Dict[int, Dict[str, torch.Tensor]] = {cid: _get_all_B(lora_layers) for cid in range(num_clients)}\n',
            '    client_B_states: Dict[int, Dict[str, torch.Tensor]] = {cid: _get_all_B(lora_layers) for cid in range(num_clients)}\n'
            '    client_A_states: Dict[int, Dict[str, torch.Tensor]] = {cid: _get_all_A(lora_layers) for cid in range(num_clients)}\n',
            "gsm8k-lpc-projection-aux",
        )
        qwen = replace_once_checked(
            qwen,
            '                    _, _, _, _, lpc_loss = _train_one_client_round(\n'
            '                        model=model, lora_layers=lora_layers, dataloader=loader,\n'
            '                        allocated_layers=set(lpc_layers), method="ffa-lora",\n'
            '                        lr=float(args.client_lr), num_epochs=int(args.lpc_num_epochs),\n'
            '                        grad_accum_steps=int(args.grad_accum_steps),\n'
            '                        max_steps=(int(args.max_client_steps) if getattr(args, "max_client_steps", None) is not None else None),\n'
            '                        device=device, client_B_state=None, client_id=cid_int,\n'
            '                        round_num=round_num, logger=logger, log_interval=int(args.log_interval),\n'
            '                    )\n'
            '                    client_B_states[cid_int] = _get_all_B(lora_layers)\n'
            '                    lpc_event_count += 1\n'
            '                    logger.info(\n'
            '                        f"round={round_num} client={cid_int} lpc_done "\n'
            '                        f"layers={len(lpc_layers)} loss={float(lpc_loss):.6f} events={lpc_event_count}"\n'
            '                    )\n',
            '                    if str(getattr(args, "lpc_mode", "train")).lower() == "projection":\n'
            '                        from factor_projection import project_B_to_new_A as _lpc_project_B\n'
            '                        _lpc_old_A = client_A_states.get(cid_int)\n'
            '                        if _lpc_old_A is None:\n'
            '                            _lpc_old_A = {n: t.detach().clone() for n, t in _get_all_A(lora_layers).items()}\n'
            '                        _lpc_old_B = client_B_states[cid_int]\n'
            '                        _lpc_new_B = _get_all_B(lora_layers)\n'
            '                        for _ln in sorted(set(lpc_layers)):\n'
            '                            if _ln not in _lpc_old_A or _ln not in _lpc_old_B or _ln not in global_A:\n'
            '                                continue\n'
            '                            _Bstar, _diag = _lpc_project_B(\n'
            '                                _lpc_old_A[_ln], _lpc_old_B[_ln], global_A[_ln],\n'
            '                                ridge=float(getattr(args, "projection_ridge", 1e-6)),\n'
            '                            )\n'
            '                            _lpc_new_B[_ln] = _Bstar\n'
            '                            logger.info(\n'
            '                                f"round={round_num} client={cid_int} type=lpc_projection_event "\n'
            '                                f"layer={_ln} residual_before={float(_diag[\'residual_before\']):.8g} "\n'
            '                                f"residual_after={float(_diag[\'residual_after\']):.8g} "\n'
            '                                f"relative_reduction={float(_diag[\'relative_reduction\']):.8g}"\n'
            '                            )\n'
            '                        _set_all_B(lora_layers, _lpc_new_B)\n'
            '                        lpc_loss = 0.0\n'
            '                    else:\n'
            '                        _, _, _, _, lpc_loss = _train_one_client_round(\n'
            '                            model=model, lora_layers=lora_layers, dataloader=loader,\n'
            '                            allocated_layers=set(lpc_layers), method="ffa-lora",\n'
            '                            lr=float(args.client_lr), num_epochs=int(args.lpc_num_epochs),\n'
            '                            grad_accum_steps=int(args.grad_accum_steps),\n'
            '                            max_steps=(int(args.max_client_steps) if getattr(args, "max_client_steps", None) is not None else None),\n'
            '                            device=device, client_B_state=None, client_id=cid_int,\n'
            '                            round_num=round_num, logger=logger, log_interval=int(args.log_interval),\n'
            '                        )\n'
            '                    client_B_states[cid_int] = _get_all_B(lora_layers)\n'
            '                    client_A_states[cid_int] = _get_all_A(lora_layers)\n'
            '                    lpc_event_count += 1\n'
            '                    logger.info(\n'
            '                        f"round={round_num} client={cid_int} lpc_done "\n'
            '                        f"layers={len(lpc_layers)} loss={float(lpc_loss):.6f} events={lpc_event_count}"\n'
            '                    )\n',
            "gsm8k-lpc-projection-branch",
        )
        qwen = replace_once_checked(
            qwen,
            '            if new_B_state is not None:\n'
            '                client_B_states[cid_int] = new_B_state\n',
            '            if new_B_state is not None:\n'
            '                client_B_states[cid_int] = new_B_state\n'
            '                client_A_states[cid_int] = _get_all_A(lora_layers)\n',
            "gsm8k-lpc-projection-asave",
        )
'''


def apply(path: str) -> None:
    with open(path, "rb") as fh:
        raw = fh.read()
    newline = "\r\n" if b"\r\n" in raw else "\n"
    text = raw.decode("utf-8").replace("\r\n", "\n")

    if MARKER in text:
        print(f"SKIP already patched: {path}")
        return

    backup = path + ".bak_textlpc_20261029"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f"backup -> {backup}")

    count = text.count(ANCHOR)
    if count != 1:
        raise SystemExit(f"FAIL: anchor count={count} (want 1) in {path}")
    text = text.replace(ANCHOR, ANCHOR + INSERT, 1)
    print("patched [gsm8k-lpc-projection-*]")

    out = text.replace("\n", newline) if newline == "\r\n" else text
    with open(path, "wb") as fh:
        fh.write(out.encode("utf-8"))
    print(f"OK {path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python patch_text_lpc_projection.py <main.py>")
    apply(sys.argv[1])
