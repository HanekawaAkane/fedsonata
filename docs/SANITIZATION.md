# Release sanitisation log

Before publication, site-specific identifiers were removed from the code tree. Nothing in the
science changed: only absolute filesystem paths and local usernames were replaced. No logic,
constant, seed, hyper-parameter, threshold or measurement was touched.

## Substitutions

| pattern (before) | replacement (after) |
|---|---|
| `/media/<site>/users/<user>/fedsonata_journal` | `/data/fedsonata_journal` |
| `/media/<site>/data3/<user>/model/gemma-4-12B-it/` | `./base_models/gemma-4-12B-it/` |

## Files touched (11)

| file | occurrences |
|---|---|
| `README.md` | 1 |
| `code/main.py` | 1 (`--model_path` default) |
| `code/check_partition_identity.py` | 1 (`DATA_DIR`) |
| `docs/campaign_status.json` | 2 |
| `docs/implementation_audit.md` | 1 |
| `scripts/run_arm.sh` | 4 |
| `scripts/run_matrix.sh` | 1 |
| `scripts/run_subset.sh` | 4 |
| `tests/analyze_shift_ideal.py` | 1 |
| `tests/check_reachability.py` | 1 |
| `tests/subset_agg.py` | 1 |

## Verification performed after editing

- `grep -rniE "yanji|h3c|/media/|10\.246\.|@qq\.com"` over the release tree → no matches.
- No e-mail address, token, key, hostname or SSH target is present in any file.
- `results/`, `*.pt`, `__pycache__/`, `*.log` and `*.bak*` are excluded by `.gitignore`.
- The two edited `.py` files still parse (`python -m py_compile`), and their public interfaces
  (`argparse` options, module-level constants) are unchanged.

## Effect on provenance hashes

The archive's `WIRE_PROTOCOL.json` records SHA-256 digests for the captured files, and those
digests remain valid **for the private archive**, not for this sanitised tree: `main.py` and
`check_partition_identity.py` differ in one path literal each. All other 35 files are byte
identical to the archived snapshot. Behavioural anchors that do not depend on path literals
(BASE trace hash for seed 42, data-split hashes) are unaffected and are reproducible from this
tree once `FEDSONATA`-style paths are pointed at local data and weights.
