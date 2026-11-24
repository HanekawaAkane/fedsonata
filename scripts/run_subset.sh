#!/usr/bin/env bash
# FS23-ScientistTwo subset runner: 5 rounds x fraction 0.05 screening protocol.
# Usage: run_subset.sh <variant> <dataset> <seed> <gpu>
#   variant: baseline | I1 | I2 | I3   (idea variants need their flags landed)
#   dataset: cifar100 | cifar10
# Output: $ROOT/results/subset_${variant}_${dataset}_s${seed}
set -u

VAR=${1:?variant}
DS=${2:?dataset}
SEED=${3:?seed}
GPU=${4:?gpu}

ROOT=/data/fedsonata_journal/fs23_experiments_20261117
CODE=$ROOT/code
PY=/data/fedsonata_journal/.venv/bin/python

ROUNDS=5
FRAC=0.05
EXTRA=""
case "$VAR" in
  baseline) ;;
  I1) EXTRA="--lpc_trigger predictive" ;;
  I2) EXTRA="--lpc_trigger value_budget" ;;
  I2r) EXTRA="--lpc_trigger value_budget --lpc_round_budget 5" ;;
  I2r2) EXTRA="--lpc_trigger value_budget --lpc_round_budget 2" ;;
  I3) EXTRA="--cap_policy expected_delivery" ;;
  *) echo "unknown variant $VAR" >&2; exit 2 ;;
esac

OUT="$ROOT/results/subset_${VAR}_${DS}_s${SEED}"
cd "$CODE" || exit 1
if [[ -f "$OUT/DONE" ]]; then echo "SKIP DONE $OUT"; exit 0; fi
if [[ -d "$OUT" ]] && find "$OUT" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "REFUSE existing incomplete $OUT" >&2; exit 4
fi
mkdir -p "$OUT"
echo "START subset var=$VAR ds=$DS seed=$SEED gpu=$GPU -> $OUT"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u main.py \
  --dataset "$DS" --data_dir /data/fedsonata_journal/data \
  --model_name vit_base_patch16_224 --pretrained --pretrained_source local \
  --pretrained_path /data/fedsonata_journal/vit \
  --method fedharp --mec_method proposed_joint_rift_ef_cora \
  --fedharp_matrix_mode ab \
  --num_clients 20 --clients_per_round 20 --num_rounds "$ROUNDS" \
  --data_partition dirichlet --alpha 0.1 --client_splits 0.8,0.1,0.1 \
  --data_fraction "$FRAC" \
  --heterogeneity_type 6-3-1 --allocation_ratio 0.5 \
  --lora_rank 8 --lora_alpha 16 --lora_dropout 0 \
  --client_lr 0.01 --lora_lr 0.0003 --classifier_lr 0.001 \
  --batch_size 16 --test_batch_size 100 --num_epochs 1 --b_train_every 4 \
  --lpc_mode projection --projection_ridge 1e-6 \
  --eval_every 1 --validation_patience 5 --validation_min_delta 0.1 \
  --min_rounds_before_stop "$ROUNDS" \
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
  --mec_protocol_version "fs23-subset-${VAR}-${DS}" \
  --mec_code_version "fs23-switches-v1-20261117" \
  --wire_codec edge_random_rift \
  --radio_policy cora --age_view dual --reliability_view observed \
  --age_delivery_weight 1.0 --radio_outer_age_weight 1.0 \
  --network_scenario base \
  $EXTRA \
  > "$OUT/stdout.log" 2>&1

status=$?
if [[ $status -ne 0 ]]; then echo "exit_code=$status" > "$OUT/FAILED"; exit "$status"; fi
if [[ ! -f "$OUT/DONE" ]]; then echo "no DONE marker" > "$OUT/FAILED"; exit 5; fi
echo "DONE subset var=$VAR ds=$DS seed=$SEED"
