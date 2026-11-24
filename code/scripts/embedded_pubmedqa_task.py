"""PubMedQA federated task — wrapper over the GSM8K causal-LM harness.

This is an embedded task source (see ``_EMBEDDED_TASK_SOURCES`` in ``main.py``).
It reuses the whole Gemma-4 federated LoRA loop of the ``gsm8k`` task by
monkeypatching three module-level helpers of that module at load time:

  * ``_read_gsm8k_rows``         -> PubMedQA parquet reader
  * ``_build_train_prompt``      -> PubMedQA yes/no/maybe prompt
  * ``_evaluate_gsm8k_accuracy`` -> PubMedQA accuracy (+ macro-F1 report)

Everything else (Dirichlet client split, per-client LoRA training, aggregation,
MEC scheduling, logging) is inherited unchanged, so PubMedQA results are directly
comparable with the GSM8K(D048) / GLUE runs.

Data: ``--data_path`` (train parquet) / ``--test_data_path`` (test parquet);
use ``--train_variant en --test_variant en`` so each row yields exactly one
sample (the reader emits no zh field).
"""

from __future__ import annotations

import re
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn

_HOST = sys.modules.get("__main__")
_GSM8K = _HOST._get_embedded_task_module("gsm8k")

_PQA_LABELS = ("yes", "no", "maybe")
_PQA_CONFUSION_TOTAL = {}


def _pqa_take_confusion():
    """D063: return the summed confusion across eval calls, then reset."""
    global _PQA_CONFUSION_TOTAL
    snapshot = {g: dict(per) for g, per in _PQA_CONFUSION_TOTAL.items()}
    _PQA_CONFUSION_TOTAL = {}
    return snapshot


def _ctx_text(context) -> str:
    if context is None:
        return ""
    get = getattr(context, "get", None)
    parts = get("contexts") if callable(get) else None
    if parts is None:
        return ""
    try:
        return " ".join(str(p) for p in parts).strip()
    except TypeError:
        return str(parts).strip()


def _read_pubmedqa_rows(path: str) -> List[Dict[str, object]]:
    df = pd.read_parquet(str(path))
    rows: List[Dict[str, object]] = []
    for _, r in df.iterrows():
        abstract = _ctx_text(r.get("context") if hasattr(r, "get") else None)
        gold = str(r["final_decision"]).strip().lower()
        rows.append(
            {
                "question": f"Abstract:\n{abstract}\n\nQuestion: {r['question']}",
                "answer": gold,
            }
        )
    if not rows:
        raise ValueError(f"未从 PubMedQA 读取到样本: {path}")
    return rows


def _parse_pqa(text: str) -> str:
    m = re.search(r"\b(yes|no|maybe)\b", str(text).lower())
    return m.group(1) if m else "unknown"


def _evaluate_pubmedqa_accuracy(
    *,
    model: nn.Module,
    tokenizer,
    samples: List[object],
    chat_format: str,
    max_length: int,
    max_new_tokens: int,
    device: torch.device,
) -> Tuple[float, int, int]:
    total = int(len(samples))
    if total == 0:
        return 0.0, 0, 0
    model.eval()
    correct = 0
    # D063: include an unknown column so unparseable predictions count
    # toward the recall side of every class instead of being dropped.
    confusion = {g: {p: 0 for p in _PQA_LABELS + ("unknown",)} for g in _PQA_LABELS}

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        eos_id = getattr(tokenizer, "eod_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_id if eos_id is not None else 0
    eos_token_id = eos_id
    if str(chat_format).lower() == "chatml":
        im_end = getattr(tokenizer, "im_end_id", None)
        if im_end is not None:
            eos_token_id = int(im_end)

    eff = int(max_new_tokens)
    if int(max_length) > 1:
        eff = min(eff, max(1, int(max_length) - 16))
    prompt_max = int(max(1, int(max_length) - eff))

    with torch.no_grad():
        for s in samples:
            pids = getattr(s, "prompt_token_ids", None)
            if pids is not None:
                pids = list(pids)[-prompt_max:]
            else:
                pids = tokenizer(
                    str(s.prompt), add_special_tokens=True,
                    truncation=True, max_length=prompt_max,
                )["input_ids"]
            input_ids = torch.tensor([pids], dtype=torch.long, device=device)
            attn = torch.ones_like(input_ids, dtype=torch.long, device=device)
            gen = model.generate(
                input_ids=input_ids, attention_mask=attn, do_sample=False,
                num_beams=1, max_new_tokens=int(eff), pad_token_id=int(pad_id),
                eos_token_id=int(eos_token_id) if eos_token_id is not None else None,
                use_cache=False,
            )
            pred_text = tokenizer.decode(gen[0].tolist()[len(pids):], skip_special_tokens=True)
            pred = _parse_pqa(pred_text)
            gold = str(getattr(s, "answer", "")).strip().lower()
            if pred == gold:
                correct += 1
            if gold in confusion:
                # D063: unknown predictions are not silently dropped; they
                # count as errors for accuracy and for the recall side of F1.
                confusion[gold][pred if pred in _PQA_LABELS else "unknown"] += 1

    acc = float(correct) / float(max(1, total))
    # D063: accumulate confusion across per-client eval calls so the
    # caller can compute macro-F1 from SUMMED confusion counts.
    for _g, _per in confusion.items():
        _row = _PQA_CONFUSION_TOTAL.setdefault(_g, {})
        for _p, _c in _per.items():
            _row[_p] = int(_row.get(_p, 0)) + int(_c)
    f1s = []
    for lab in _PQA_LABELS:
        tp = confusion[lab][lab]
        fp = sum(confusion[g][lab] for g in _PQA_LABELS if g != lab)
        fn = sum(confusion[lab][p] for p in confusion[lab].keys() if p != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    macro = sum(f1s) / len(_PQA_LABELS)
    print(
        f"[pubmedqa] n={total} acc={acc:.4f} macro_f1={macro:.4f} confusion={confusion}",
        flush=True,
    )
    return acc, int(correct), int(total)


def _build_pubmedqa_prompt(*, question: str, lang: str) -> str:
    return (
        "You are a biomedical research assistant. Use ONLY the given PubMed "
        "abstract to answer. Respond with exactly one word: yes, no, or maybe.\n\n"
        f"{str(question)}\n\nAnswer (yes/no/maybe):"
    )


# ---- install the PubMedQA behaviour into the GSM8K module globals ----
# D063: expose the summed-confusion accessor to the causal-LM harness
# (consumed at each round_summary; see scripts/d063_text_patches.py).
_GSM8K._pqa_take_confusion = _pqa_take_confusion
_GSM8K._read_gsm8k_rows = _read_pubmedqa_rows
_GSM8K._build_train_prompt = _build_pubmedqa_prompt
_GSM8K._evaluate_gsm8k_accuracy = _evaluate_pubmedqa_accuracy


def main(argv=None):
    return _GSM8K.main(argv)
