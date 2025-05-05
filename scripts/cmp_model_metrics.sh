#!/usr/bin/env bash
# scripts/compare_model_metrics.sh

# ──────────────────────────────────────────────────────────────────────────────
# Go to project root
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
# ──────────────────────────────────────────────────────────────────────────────

# How many samples to evaluate & which split
NUM_SAMPLES=2500
SPLIT="test"

# Map model names → their checkpoint subdirs
declare -A MODELS=(
  [pegasus]="pegasus_subset/best_model"
  [bart]="bart/best_model"
  [led]="led/best_model"
)

for name in "${!MODELS[@]}"; do
  CKPT_DIR="outputs/${MODELS[$name]}"
  CONFIG="configs/${name}.yaml"

  echo "=== Evaluating $name (subset=$NUM_SAMPLES on split=$SPLIT) ==="
  echo "  config     : $CONFIG"
  echo "  checkpoint : $CKPT_DIR"
  echo

  python -m src.main eval \
    --config      "$CONFIG" \
    --ckpt_dir    "$CKPT_DIR" \
    --split       "$SPLIT" \
    --num_samples "$NUM_SAMPLES"

  echo "→ Metrics for $name at: $CKPT_DIR/eval_metrics.csv"
  echo
done
