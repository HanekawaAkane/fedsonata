#!/usr/bin/env bash
# FS23 campaign dispatcher (2 streams per GPU).
# Usage: run_matrix.sh <stage> [gpu_csv]
#   stage=main   : 8 arms x 3 seeds (base)               -> 24 runs
#   stage=shift  : (FULL, DELIVERY_CONTROL_OPEN) x 3 x (ideal, shift) -> 12 runs
#   gpu_csv      : e.g. "0,2,4"
#
# The 24-run list is enumerated seed-major (s42 block, s2025 block, s2026
# block) and dealt round-robin to the streams, so each stream gets a mix and
# paired seed blocks finish close together. run_arm.sh skips DONE and refuses
# non-empty incomplete dirs -> re-invoking is safe.
set -u

STAGE=${1:?stage}
GPUS=${2:-0,2,4}
ROOT=/data/fedsonata_journal/fs23_experiments_20261117
SCRIPTS=$ROOT/scripts
mkdir -p "$ROOT"

ARMS=(FULL RADIO_EQUAL RADIO_BAPFT AGE_OFF ALLOC_ONLY DELIVERY_ONLY NO_RELIABILITY DELIVERY_CONTROL_OPEN)
SEEDS=(42 2025 2026)

TASKS=()
if [[ "$STAGE" == "main" ]]; then
  for seed in "${SEEDS[@]}"; do
    for arm in "${ARMS[@]}"; do
      TASKS+=("$arm:$seed:base")
    done
  done
elif [[ "$STAGE" == "shift" ]]; then
  for sc in ideal shift; do
    for seed in "${SEEDS[@]}"; do
      for arm in FULL DELIVERY_CONTROL_OPEN; do
        TASKS+=("$arm:$seed:$sc")
      done
    done
  done
else
  echo "unknown stage $STAGE" >&2; exit 2
fi

IFS=',' read -ra GPU_ARR <<< "$GPUS"
N=${#GPU_ARR[@]}
# 2 streams per GPU.
STREAMS=()
for g in "${GPU_ARR[@]}"; do STREAMS+=("$g"); done
for g in "${GPU_ARR[@]}"; do STREAMS+=("$g"); done
S=${#STREAMS[@]}
echo "stage=$STAGE tasks=${#TASKS[@]} streams=$S gpus=${STREAMS[*]}"

pids=()
for ((i = 0; i < S; i++)); do
  (
    idx=$i
    while [[ $idx -lt ${#TASKS[@]} ]]; do
      IFS=':' read -r arm seed sc <<< "${TASKS[$idx]}"
      echo "[dispatch] idx=$idx $arm s$seed $sc gpu=${STREAMS[$i]}"
      bash "$SCRIPTS/run_arm.sh" "$arm" "$seed" "${STREAMS[$i]}" 20 "$sc" \
        && echo "[ok] $arm s$seed $sc" || echo "[fail] $arm s$seed $sc rc=$?"
      idx=$((idx + S))
    done
  ) > "$ROOT/logs_${STAGE}_stream${i}.log" 2>&1 &
  pids+=($!)
  echo "stream $i gpu=${STREAMS[$i]} pid=${pids[-1]} log=logs_${STAGE}_stream${i}.log"
done
for p in "${pids[@]}"; do wait "$p"; done
echo "stage $STAGE complete"
