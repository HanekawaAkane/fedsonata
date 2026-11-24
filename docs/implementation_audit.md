# Experiment implementation audit (2026-11-17)

Scope: read-only inspection of local FedSONATA source snapshots and archived Nov6 run evidence. No code changes, remote reads or training launches were performed.

## 1. Which source tree to use

The recorded unified Edge-factor main arm is `runs/2026-11-05-fedsonata-audit/run_edge_factor_main.sh` (also archived as `artifacts/edge_factor_main_20261106/run_arm.sh`). It launches remote `/data/fedsonata_journal/lossaware_wire_matched_20261016/code/main.py`, with `--mec_method proposed_joint_rift_ef_cora --wire_codec edge_random_rift --lpc_mode projection`, 20 rounds, seeds 42/2025/2026, and **no** `--wire_replay`.

Local root `main.py` is a different Oct29 copy and does not provide the same wire-codec entry point. The closest local snapshot is `artifacts/factor_policy_redesign_20261016/wire_matched_source/` (abbreviated **WMS** below). Its `main.py` SHA-256 is `1c8fa492cb6e5f70448fe07ed537b9001926bd8e4736c55ce580ea28490bff83`, whereas Nov6 `WIRE_PROTOCOL.json` records `34b5fa87620dd3bff6d0309e56854d8d8301d5882a4532f4f93327ed106d730c`. Thus WMS main is **not** an exact runnable reconstruction of the projection-LPC main arm.

The following WMS files DO match Nov6 recorded SHA-256 values byte-for-byte: `wire_codec.py`, `mec/edge_random_factor_policy.py`, `mec/staleness.py`, `mec/scheduler.py`, `mec/optimal_resource_solver.py`, `mec/system_model.py`, `mec/traces.py`. These are reliable evidence for the corresponding modules, but the entry point still needs a fresh source capture before experiments.

Archived `artifacts/edge_factor_main_20261106/results/main_s{42,2025,2026}/` each contain DONE, configs, final metrics and wire metrics. Their holdout accuracies are 90.9762%, 91.2540%, 91.0960% (91.11 +/- 0.14%). Do not silently substitute these into a PDF whose main row is 90.99%; its different provenance is a separate manuscript issue.

## 2. Missing plus is a manuscript error, not the audited sampler

Both root `mec/edge_random_factor_policy.py:41-45` and the matching WMS module use

```python
raw[name] = floor + 0.5 * (
    exp(-position / tau) + exp(-(1.0 - position) / tau)
)
normalizer = sum(raw.values()) / len(raw)
```

WMS `wire_codec.py:355-365` invokes `sample_edge_weighted_atoms_v2` from that module for `edge_random_rift`. The byte cap is computed at `wire_codec.py:335-342`; a full-message header reserve is applied at `:345-361`. This module SHA-256 equals the Nov6 protocol's recorded `7765a209508999a42c11ca8fd278d849147f78078a41d4f1e6a1c90afd272945`.

## 3. Pure-radio intervention

WMS `main.py:806-821` selects network policy from `mec_method`; `:1209-1221` independently whitelist RIFT/LEF activation. Merely replacing the method with `proposed_equal_bandwidth`, `proposed_no_power_optimization` or a BAPFT suffix can change multiple components or disable factor selection/LEF. In particular `proposed_no_network_staleness` and `proposed_no_power_optimization` occur only in the parser (`:257-258`): they are not implemented as named mechanisms.

An existing `proposed_joint_rift_ef_equal_fixed` variant preserves proposed block allocation, RIFT and LEF while selecting Equal-Fixed, but a clean BAPFT counterpart is absent. The safest new implementation keeps `mec_method=proposed_joint_rift_ef_cora`, `wire_codec=edge_random_rift`, projection-LPC, and introduces an explicit independent `radio_policy` enum. Dispatch only the final resource-plan block (`WMS main.py:2441-2456`) to CORA / Equal-Fixed / BAPFT-style. Equal-Fixed initialization is already at `:2422-2425`; BAPFT-style is at `:2502-2521`. Keep correct shared bandwidth, power cap, deadline interval, and accounting identical.

Two distinct questions require two distinct experiments:

- **Systems-only radio effect:** record a common payload/compute-time/client trace, replay it into each radio solver using identical channel, base-dropout, and duration-hazard uniforms. Report latency, compute/transmit energy, on-time rate and feasibility; do not claim end-to-end accuracy from this offline replay.
- **End-to-end radio effect:** hold the algorithm/configuration fixed and change only `radio_policy`; let the induced arrivals and later learning states diverge naturally. Do not force equal later payloads and then describe the result as the natural closed-loop effect.

MEC traces support common random numbers: `mec/traces.py:103-124` provides a fixed uniform per (round, client), while failure events depend on the arm's upload duration; trace hashing includes these uniforms (`:150-172`). Match uniforms, not realized failure booleans.

## 4. Staleness consumers and non-isolated CLI flags

Matched WMS `mec/staleness.py:36-40` defines `effective = alloc + omega_net * net`. Updates reset both counters on selected+arrived (`:57-60`), increment allocation age when unselected (`:61-62`), and delivery age when selected+failed (`:63-65`). Only observed candidate clients are updated by `WMS main.py:2626-2629`; unsampled clients retain state.

Consumers in WMS:

- Block scheduler: `main.py:1688` wraps the state, `:1747-1750` selects blocks; `mec/scheduler.py:127-144` reads effective age in utility and coverage.
- LPC trigger: `main.py:1766-1768`, `mec/staleness.py:103-107`. The audited implementation triggers on `round(age) == b_train_every`, not a general `>=` threshold. Preserve the chosen implementation for new comparisons and test overshoot explicitly; do not quietly alter it mid-experiment.
- RIFT selector inputs: `main.py:2011-2014`, `:2099-2100`. The Edge sampler does not consume these values; they are unused kwargs in its branch.
- CORA predictor and corrector: effective age maps `main.py:1819-1825` / `:2415-2418`; both also receive `args.mec_omega_net` (`:1844`, `:2449`). Therefore `--mec_omega_net 0` changes both the effective-age composition and radio utility weighting; it is **not** an isolated allocation-only age intervention.
- Legacy server allocation-age dampening is separate: `server.py:776-796`, not the MEC dual-counter store. Keep this aggregation component identical across the staleness comparison unless intentionally studied separately.

Minimal implementation: preserve the true `(alloc, net)` counters and add a read-only consumer view `age_signal in {dual, allocation_only, delivery_only, zero, merged}`. Give radio's outer staleness coefficient a separate parameter held fixed. Log true counters and consumed signals separately. Keep all reliability, residual updates, correction mode, budgets and radio settings fixed.

**Identifiability warning:** with the reported `omega_net=1`, dual age `alloc+net` is exactly equivalent to a single merged counter incremented on either unselected or selected+failed and reset on selected+arrived, for the same observed clients. A merged-vs-dual comparison at weight 1 should be an equivalence/negative-control check, not an expected gain. Allocation-only vs full can establish the usefulness of delivery history; it cannot by itself establish that storing two counters beats an equivalently updated single counter.

## 5. Reliability / delivery feedback loop

True end-to-end arrival updates are applied at `WMS main.py:2626-2635`: effective staleness and reliability EWMA update after the honest arrival gate. Reliability affects scheduler utility (`:1748`; `mec/scheduler.py:125`), the byte cap (`wire_codec.py:338`), and radio scores (`main.py:2451`; `mec/optimal_resource_solver.py:513-526`).

For an explicitly defined transport-to-control ablation, leave real arrivals, arrival-gated aggregation (`main.py:2619-2624`) and ACK residual updates (`:2606-2616`) honest in every arm. Create a separate controller-visible reliability view (`observed` vs predeclared constant), and an allocation-only vs dual age view. This permits a 2x2 design for reliability-history feedback and delivery-age feedback while holding learning-side error feedback (LEF) fixed. Calling this a test of **all** ACK feedback would be inaccurate because LEF remains arrival-gated by design.

Existing `--wire_replay` is unsuitable for this end-to-end question: `wire_codec.py:319-326` overwrites block allocation, reliability, server age and both MEC ages from a donor trace. It forces several feedback consumers simultaneously. Use it only in a clearly labeled fixed-control diagnostic, not as a no-feedback algorithm or a causal radio-only ablation.

## 6. Predictor dependency

WMS main computes CORA reference rates (`main.py:1796-1853`) and passes them to the selector (`:1993-2008`, `:2107-2110`). But `wire_codec.py:355-365` forwards to Edge only deltas, byte cap, header reserve, quantum, seed/client/round; no reference rate, utility or staleness is read. Predictor still performs a feasibility check and an upper-bound assertion (`main.py:2427-2438`), so removing it may change abort behavior for infeasible cases, but its rates do not determine Edge actions in this audited path. Treat predictor-off as a structural equivalence/overhead test for feasible Edge runs, not an assumed accuracy-improvement experiment. RIFT's value-based branch does consume the rate; the distinction must be reflected in the manuscript.

## 7. Prelaunch gates and meaningful tests

1. Capture the actual remote source revision corresponding to the recorded Nov6 projection-LPC launcher and a full hash manifest. Port isolated overrides there, not into a guessed stale local entry point.
2. Verify the unchanged Full arm reproduces protocol, split hash, trace hash, payload format, projection/LEF modes and 20-round checkpoint selection semantics. Use new output paths and a fresh Full arm for paired inference.
3. Parser-to-consumer test: changing each new enum reaches exactly its intended dispatch; unsupported legacy aliases must fail instead of silently falling back.
4. Synthetic state trajectory: unselected / failed / success / unsampled events yield correct true and controller-visible counters; merged-vs-dual at omega 1 is bit-identical. Include LPC threshold overshoot and noninteger-age cases.
5. Frozen tensor/state test: switching radio policy cannot alter pre-radio block allocation, selected factor mask, message bytes, or LPC/LEF work; afterward arrival differences may propagate normally.
6. Resource constraints: sum bandwidth <= total budget, per-client power within cap, deadline within bounds, arrival/attempted-duration/energy recomputed identically. Count infeasible rounds explicitly; do not silently drop failed cases.
7. Common-random-number test: across paired arms seeds/splits/channel/base-dropout/hazard uniforms match; realized duration failures are allowed to differ.
8. No-feedback test: true counters/arrival logs continue updating, but controller reads only declared views; ACK residuals and aggregation remain honest.
9. Predictor-off feasible test: Edge masks, wire byte counts and final corrector plans match while predictor call time disappears; classify infeasible differences separately.
10. Report paired per-seed deltas and the actual number of independent seeds; do not treat clients/rounds as independent replicates. Save DONE, FAILED, configs, code hashes, selected/final-round metrics, per-round controller inputs/outputs and individual raw outcomes.

No new experimental result is asserted by this audit.
