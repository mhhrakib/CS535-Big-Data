# #!/usr/bin/env bash
# # scripts/compare_model_summaries.sh

# # Move to project root (where src/ lives)
# PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# cd "$PROJECT_ROOT"

# # Common settings
# DATASET="alexfabbri/multi_news"
# SPLIT="test"
# NUM_SAMPLES=3

# # Model → checkpoint subdir
# declare -A MODELS=(
#   [pegasus]="pegasus_subset/best_model"
#   [bart]="bart_subset/best_model"
#   [led]="led_subset/best_model"
# )

# # Model → max input/output lengths
# declare -A MAX_IN=(
#   [pegasus]=1024
#   [bart]=1024
#   [led]=16384
# )
# declare -A MAX_OUT=(
#   [pegasus]=256
#   [bart]=142
#   [led]=512
# )

# for name in "${!MODELS[@]}"; do
#   CKPT_DIR="outputs/${MODELS[$name]}"
#   IN_LEN=${MAX_IN[$name]}
#   OUT_LEN=${MAX_OUT[$name]}
#   OUTFILE="summaries_${name}.txt"

#   echo "=== Generating $NUM_SAMPLES summaries with $name ==="
#   echo "  checkpoint: $CKPT_DIR"
#   echo "  max_input_length: $IN_LEN, max_output_length: $OUT_LEN"
#   echo "  output file: $OUTFILE"
#   echo

#   python -m src.inference \
#     --model_path "$CKPT_DIR" \
#     --dataset "$DATASET" \
#     --split "$SPLIT" \
#     --num_samples "$NUM_SAMPLES" \
#     --output_file "$OUTFILE" \
#     --max_input_length "$IN_LEN" \
#     --max_output_length "$OUT_LEN"

#   echo
# done

#!/usr/bin/env bash
# scripts/compare_model_summaries.sh

# ──────────────────────────────────────────────────────────────────────────────
# Jump to the project root (where src/ and configs/ live)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
# ──────────────────────────────────────────────────────────────────────────────

# How many docs to sample and which split
NUM_SAMPLES=3
SPLIT="test"

# Map model names → their subset checkpoint directories
declare -A MODELS=(
  [pegasus]="pegasus_subset/best_model"
  [bart]="bart_subset/best_model"
  [led]="led_subset/best_model"
)

for name in "${!MODELS[@]}"; do
  CKPT_DIR="outputs/${MODELS[$name]}"
  CONFIG="configs/${name}.yaml"
  OUTFILE="summaries_${name}.txt"

  echo "=== Generating $NUM_SAMPLES summaries with $name ==="
  echo "  config : $CONFIG"
  echo "  checkpoint : $CKPT_DIR"
  echo "  split : $SPLIT"
  echo "  samples : $NUM_SAMPLES"
  echo "  output : $OUTFILE"
  echo

  python -m src.inference \
    --config      "$CONFIG" \
    --model_path  "$CKPT_DIR" \
    --split       "$SPLIT" \
    --num_samples "$NUM_SAMPLES" \
    --output_file "$OUTFILE"

  echo
done

