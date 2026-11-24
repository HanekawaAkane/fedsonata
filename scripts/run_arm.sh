#!/usr/bin/env bash
# FS23 unified launcher: same training protocol as
# runs/2026-11-05-fedsonata-audit/run_edge_factor_main.sh (Nov6 archive,
# source-hash verified), plus the independent intervention switches.
#
# Usage: run_arm.sh <arm> <seed> <gpu> [rounds] [scenario]
#   arm:      FULL | RADIO_EQUAL | RADIO_BAPFT | AGE_OFF | ALLOC_ONLY |
#             DELIVERY_ONLY | NO_RELIABILITY | DELIVERY_CONTROL_OPEN
#   scenario: base (default) | ideal | shift
#   rounds:   20 (default)
# Env: MIN_ROUNDS / PATIENCE / DATA_FRACTION override smoke knobs.
set -u

ARM=${1:?arm}
SEED=${2:?seed}
GPU=${3:?gpu}
ROUNDS=${4:-20}
SCENARIO=${5:-base}

ROOT=/data/fedsonata_journal/fs23_experiments_20261117
CODE=$ROOT/code
PY=/data/fedsonata_journal/.venv/bin/python

MIN_ROUNDS=${MIN_ROUNDS:-$ROUNDS}
PATIENCE=${PATIENCE:-5}
DATA_FRACTION=${DATA_FRACTION:-0.2}

# ---- intervention mapping (manifest.jsonl intervention_spec) ---- #
RADIO=cora
AGE_VIEW=dual
REL_VIEW=observed
case "$ARM" in
  FULL)                 RADIO=cora;        AGE_VIEW=dual;             REL_VIEW=observed ;;
  RADIO_EQUAL)          RADIO=equal_fixed; AGE_VIEW=dual;             REL_VIEW=observed ;;
  RADIO_BAPFT)          RADIO=bapft_style; AGE_VIEW=dual;             REL_VIEW=observed ;;
  AGE_OFF)              RADIO=cora;        AGE_VIEW=zero;             REL_VIEW=observed ;;
  ALLOC_ONLY)           RADIO=cora;        AGE_VIEW=allocation_only;  REL_VIEW=observed ;;
  DELIVERY_ONLY)        RADIO=cora;        AGE_VIEW=delivery_only;    REL_VIEW=observed ;;
  NO_RELIABILITY)       RADIO=cora;        AGE_VIEW=dual;             REL_VIEW=constant_initial ;;
  DELIVERY_CONTROL_OPEN) RADIO=cora;       AGE_VIEW=allocation_only;  REL_VIEW=constant_initial ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
case "$SCENARIO" in
  base|ideal|shift) ;;
  *) echo "unknown scenario $SCENARIO" >&2; exit 2 ;;
esac

SCEN_TAG=$SCENARIO
SMOKE_TAG=""
if [[ "${SMOKE:-0}" == "1" ]]; then SMOKE_TAG="_smoke"; fi
OUT="$ROOT/results/fs23_20261117_${SCEN_TAG}_${ARM,,}_s${SEED}${SMOKE_TAG}"

cd "$CODE" || exit 1
if [[ -f "$OUT/DONE" ]]; then echo "SKIP DONE $OUT"; exit 0; fi
if [[ -d "$OUT" ]] && find "$OUT" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "REFUSE existing incomplete $OUT" >&2; exit 4
fi
mkdir -p "$OUT"

echo "START arm=$ARM seed=$SEED gpu=$GPU rounds=$ROUNDS scenario=$SCENARIO radio=$RADIO age=$AGE_VIEW rel=$REL_VIEW out=$OUT"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u main.py \
  --dataset cifar100 --data_dir /data/fedsonata_journal/data \
  --model_name vit_base_patch16_224 --pretrained --pretrained_source local \
  --pretrained_path /data/fedsonata_journal/vit \
  --method fedharp --mec_method proposed_joint_rift_ef_cora \
  --fedharp_matrix_mode ab \
  --num_clients 20 --clients_per_round 20 --num_rounds "$ROUNDS" \
  --data_partition dirichlet --alpha 0.1 --client_splits 0.8,0.1,0.1 \
  --data_fraction "$DATA_FRACTION" \
  --heterogeneity_type 6-3-1 --allocation_ratio 0.5 \
  --lora_rank 8 --lora_alpha 16 --lora_dropout 0 \
  --client_lr 0.01 --lora_lr 0.0003 --classifier_lr 0.001 \
  --batch_size 16 --test_batch_size 100 --num_epochs 1 --b_train_every 4 \
  --lpc_mode projection --projection_ridge 1e-6 \
  --eval_every 1 --validation_patience "$PATIENCE" --validation_min_delta 0.1 \
  --min_rounds_before_stop "$MIN_ROUNDS" \
  --restore_best_validation --classifier_policy client_local \
  --evaluation_scope personalized_client_holdout \
  --mec_channel correlated --mec_channel_correlation 0.9 --mec_channel_gain_mean 1e-10 \
  --mec_dropout_model gilbert_elliott --mec_dropout_rate 0.2 --mec_ge_good_to_bad 0.1 \
  --mec_duration_hazard_rate 0.1 --mec_total_bw_mhz 5 --mec_power_max_dbm 23 \
  --mec_d_min 0.5 --mec_d_max 10 \
  --mec_omega_net 1 --mec_reliability_beta 0.8 \
  --rift_budget_ratio 0.35 --rift_ef_decay 0.5 \
  --rift_reliability_floor 0.25 \
  --mec_lambda_e 0.1 --mec_lambda_u 1 \
  --mec_deadline_grid_size 24 \
  --strict_determinism --seed "$SEED" --device cuda \
  --save_checkpoints --checkpoint_dir "$OUT" --mec_output_dir "$OUT" \
  --mec_protocol_version "fs23-20261117-${SCEN_TAG}-${ARM,,}" \
  --mec_code_version "fs23-switches-v1-20261117" \
  --wire_codec edge_random_rift \
  --radio_policy "$RADIO" \
  --age_view "$AGE_VIEW" \
  --reliability_view "$REL_VIEW" \
  --age_delivery_weight 1.0 \
  --radio_outer_age_weight 1.0 \
  --network_scenario "$SCENARIO" \
  > "$OUT/stdout.log" 2>&1

status=$?
if [[ $status -ne 0 ]]; then echo "exit_code=$status" > "$OUT/FAILED"; exit "$status"; fi
if [[ ! -f "$OUT/DONE" ]]; then echo "no DONE marker" > "$OUT/FAILED"; exit 5; fi
echo "DONE arm=$ARM seed=$SEED scenario=$SCENARIO"
