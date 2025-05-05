# #!/usr/bin/env bash
# # scripts/compare_model_metrics.sh

# # ──────────────────────────────────────────────────────────────────────────────
# # Go to project root
# PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# cd "$PROJECT_ROOT"
# # ──────────────────────────────────────────────────────────────────────────────

# NUM_SAMPLES=20
# SPLIT="test"

# # Map model name → checkpoint subdir (under outputs/)
# declare -A MODELS=(
# #   [pegasus]="pegasus/best_model"
#   [bart]="bart_subset/best_model"
#   [led]="led_subset/best_model"
# )

# # Collect metrics into a temp file
# TMP_SUMMARY="$(mktemp)"
# printf "MODEL,rouge1,rouge2,rougeL,bertscore_f1,avg_extractiveness,avg_density\n" > "$TMP_SUMMARY"

# for name in pegasus bart led; do
#   CKPT_DIR="outputs/${MODELS[$name]}"
#   CONFIG="configs/${name}.yaml"

#   echo "=== Evaluating $name ==="
#   echo "  config     : $CONFIG"
#   echo "  checkpoint : $CKPT_DIR"
#   echo

#   # build the eval command
#   CMD=( python -m src.main eval
#         --config      "$CONFIG"
#         --ckpt_dir    "$CKPT_DIR"
#         --split       "$SPLIT"
#       )
#   # only sample for bart & led
#   if [[ "$name" != "pegasus" ]]; then
#     CMD+=( --num_samples "$NUM_SAMPLES" )
#   fi

#   "${CMD[@]}"

#   # pull out the CSV line of metrics
#   METRICS_CSV="$CKPT_DIR/eval_metrics.csv"
#   if [[ -f "$METRICS_CSV" ]]; then
#     # assume second line has the values
#     VALS=$(sed -n '2p' "$METRICS_CSV")
#     printf "%s,%s\n" "$name" "$VALS" >> "$TMP_SUMMARY"
#   else
#     printf "%s,ERROR\n" "$name" >> "$TMP_SUMMARY"
#   fi

#   echo
# done

# echo "=== Combined Metrics ==="
# column -t -s, "$TMP_SUMMARY"

# # cleanup
# rm "$TMP_SUMMARY"


# #!/usr/bin/env bash
# # scripts/compare_model_metrics.sh

# # ──────────────────────────────────────────────────────────────────────────────
# # Go to project root
# PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# cd "$PROJECT_ROOT"
# # ──────────────────────────────────────────────────────────────────────────────

# NUM_SAMPLES=10
# SPLIT="test"

# # Only BART and LED now
# declare -A MODELS=(
#   [pegasus]="pegasus/best_model"
#   [bart]="bart/best_model"
#   [led]="led/best_model"
# )

# # Temp summary file
# TMP_SUMMARY="$(mktemp)"
# printf "MODEL,rouge1,rouge2,rougeL,bertscore_f1,avg_extractiveness,avg_density\n" \
#   > "$TMP_SUMMARY"

# for name in pegasus bart led; do
#   CKPT_DIR="outputs/${MODELS[$name]}"
#   CONFIG="configs/${name}.yaml"

#   echo "=== Evaluating $name on $NUM_SAMPLES samples of $SPLIT ==="
#   echo "  config     : $CONFIG"
#   echo "  checkpoint : $CKPT_DIR"
#   echo

#   python -m src.main eval \
#     --config      "$CONFIG" \
#     --ckpt_dir    "$CKPT_DIR" \
#     --split       "$SPLIT" \
#     --num_samples "$NUM_SAMPLES"

#   METRICS_CSV="$CKPT_DIR/eval_metrics.csv"
#   if [[ -f "$METRICS_CSV" ]]; then
#     VALS=$(sed -n '2p' "$METRICS_CSV")
#     printf "%s,%s\n" "$name" "$VALS" >> "$TMP_SUMMARY"
#   else
#     printf "%s,ERROR\n" "$name" >> "$TMP_SUMMARY"
#   fi

#   echo
# done

# echo "=== Combined Metrics (BART & LED) ==="
# column -t -s, "$TMP_SUMMARY"

# rm "$TMP_SUMMARY"

#!/usr/bin/env bash
# scripts/compare_model_metrics.sh

# ──────────────────────────────────────────────────────────────────────────────
# Go to project root
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
# ──────────────────────────────────────────────────────────────────────────────

NUM_SAMPLES=100   # change to whatever subset size you want
SPLIT="test"

# Map each model to its best_model checkpoint
declare -A MODELS=(
  [pegasus]="pegasus/best_model"
  [bart]="bart/best_model"
  [led]="led/best_model"
)

# Prepare a temporary CSV for summary
TMP_SUMMARY="$(mktemp)"
printf "MODEL,rouge1,rouge2,rougeL,bertscore_f1,avg_extractiveness,avg_density\n" > "$TMP_SUMMARY"

for name in pegasus bart led; do
  CKPT_DIR="outputs/${MODELS[$name]}"
  CONFIG="configs/${name}.yaml"

  echo "=== Evaluating $name on $NUM_SAMPLES samples of $SPLIT ==="
  echo "  config     : $CONFIG"
  echo "  checkpoint : $CKPT_DIR"
  echo

  python -m src.main eval \
    --config      "$CONFIG" \
    --ckpt_dir    "$CKPT_DIR" \
    --split       "$SPLIT" \
    --num_samples "$NUM_SAMPLES"

  METRICS_CSV="$CKPT_DIR/eval_metrics.csv"
  if [[ -f "$METRICS_CSV" ]]; then
    VALS=$(sed -n '2p' "$METRICS_CSV")
    printf "%s,%s\n" "$name" "$VALS" >> "$TMP_SUMMARY"
  else
    printf "%s,ERROR\n" "$name" >> "$TMP_SUMMARY"
  fi

  echo
done

echo "=== Combined Metrics (All Models) ==="
column -t -s, "$TMP_SUMMARY"

rm "$TMP_SUMMARY"
