# FedSONATA — Federated LoRA Experiment Suite

Experiment code for **FedSONATA**: federated LoRA fine-tuning of a ViT-B/16 backbone over a
simulated wireless MEC network (20 clients, Dirichlet(0.1) non-IID CIFAR-100/10), with a
per-round three-stage controller:

| stage | controller | this repo's switch |
|---|---|---|
| **P1** | block allocation (reliability + dual staleness) | `--age_view`, `--reliability_view` |
| **P2** | factor payload selection (RIFT / Edge-factor) | `--wire_codec`, `--cap_policy` |
| **P3** | radio plan (bandwidth / power / deadline, CORA) | `--radio_policy` |

Plus: projection-LPC correction (`--lpc_mode projection`), LEF error feedback, and
pre-declared network scenarios (`--network_scenario base|ideal|shift`).

All switches default to the baseline behaviour; an unmodified run reproduces the archived
main arm **bit-for-bit** (see *Provenance*).

---

## Provenance (why this code tree is trustworthy)

This `code/` directory is a snapshot of the archived Nov-6 projection-LPC main arm
(`lossaware_wire_matched_20261016/code`), verified against the recorded `WIRE_PROTOCOL.json`
before release:

- 37/37 source files matched their recorded SHA-256 (`main.py` = `34b5fa87…`) at capture time.
  Two files were then edited **only in their path defaults** so the release does not carry
  site-specific absolute paths — `code/main.py` (`--model_path` default) and
  `code/check_partition_identity.py` (`DATA_DIR`). Every other byte is unchanged; their
  recorded hashes therefore refer to the private archive, not to this tree. See
  `docs/SANITIZATION.md`.
- BASE trace hash (seed 42) = `afbe686aef47ad7cbc42f4d9adf132ba0d92d707058e85368799464802e5df46`.
- Data-split hash = `6464a074…` / `98d9cf9b…` / `6e24ce98…` for seeds 42 / 2025 / 2026.
- Full arm validation-selected holdout = 90.976 / 91.254 / 91.096 % (91.11 ± 0.14), matching
  the archived record to the last digit.

The intervention switches are applied by anchored text patches in `patcher/`; every patch
asserts its anchor is unique and re-compiles the file. Originals are backed up as `*.orig_fs23`.

## Layout

```
code/      archived training source (main.py + mec/ modules)
patcher/   anchored patch scripts (switches, endpoints, S3 ideas) + patched file copies
scripts/   launchers: run_arm.sh (full protocol), run_subset.sh (screening), run_matrix.sh (campaign)
tests/     gate tests, smoke checks, aggregation/monitoring tools
docs/      experiment protocol, source audit, design manifest, results tables
```

## Protocol in one paragraph

20 rounds, 20 clients/round, LoRA r=8 α=16, lr 3e-4 (LoRA) / 1e-3 (classifier), batch 16,
1 local epoch, Dirichlet α=0.1, data fraction 0.2, 6-3-1 device heterogeneity, allocation
ratio 0.5. Wireless: correlated fading (ρ=0.9, mean gain 1e-10), Gilbert–Elliott dropout
(rate 0.2, p=0.1, q=0.4), duration hazard 0.1/s, 5 MHz, 23 dBm, deadline ∈ [0.5, 10] s.
Primary endpoint: round-20 personalized client-holdout accuracy, reported **together with**
modeled tx energy / UL bytes / MEC time; 3 paired seeds (42/2025/2026); per-seed differences
with 95 % t-intervals (df = 2).

## Quick start

`scripts/run_arm.sh <arm> <seed> <gpu> [rounds] [scenario]` with arms
`FULL | RADIO_EQUAL | RADIO_BAPFT | AGE_OFF | ALLOC_ONLY | DELIVERY_ONLY | NO_RELIABILITY | DELIVERY_CONTROL_OPEN`.

```bash
# full protocol, one seed
bash scripts/run_arm.sh FULL 42 0 20 base

# screening protocol (5 rounds x 5% data)
bash scripts/run_subset.sh baseline cifar100 42 0

# gate tests (no training)
python tests/test_fs23_gates.py
```

The launchers assume a site-specific environment: a Python 3.11 venv, CIFAR data at
`/data/fedsonata_journal/data`, local ViT weights at
`.../vit`. Edit the `ROOT`/`CODE`/`PY` variables at the top of each script.

## Results snapshot (CIFAR-100, 3 seeds, round-20 holdout)

| arm | acc % | UL MB | tx_E J | MEC s | LPC events |
|---|---|---|---|---|---|
| FULL (CORA) | 91.32 ± 0.36 | 686 | **340** | 30.7 | 2874 |
| RADIO_EQUAL | 91.32 ± 0.37 | 715 | 1150 | 32.6 | 2513 |
| RADIO_BAPFT | 91.28 ± 0.30 | 661 | 2815 | 38.4 | 3057 |
| AGE_OFF | 91.22 ± 0.39 | 686 | 340 | 30.7 | **0** |
| NO_RELIABILITY | 91.35 ± 0.33 | 802 | 383 | 32.7 | 2706 |

All paired accuracy differences have 95 % CIs crossing zero; the signal is in cost and
correction volume. IDEAL scenario: FULL ≡ DELIVERY_CONTROL_OPEN round-by-round (negative
control passes). SHIFT scenario: the byte gap between the two arms grows with reliability
decay (−19 → −36 → −61 MB across phases). Full tables in `docs/report.csv`, `docs/report.json`.

## Safety notes

- The gate tests (`tests/test_fs23_gates.py`) must pass before any launch; they cover the
  merged-vs-dual staleness equivalence, LPC trigger semantics, zero-hazard branch, and the
  parser rejecting legacy aliases.
- `--lpc_trigger value_budget` without `--lpc_round_budget` selects nothing by design
  (fail-safe after an engineering incident where "no budget" meant "trigger everything").
- `--mode` semantics are documented in `docs/PROTOCOL_CN.md` (Chinese) — read it before
  changing any switch.

## License / status

Research code, unpublished. No license granted yet; treat as all-rights-reserved until a
license is added.
