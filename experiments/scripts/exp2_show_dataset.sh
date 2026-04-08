#!/bin/bash
# exp2_show_dataset.sh — Compare with and without --show-dataset
# Runs the SAME kernel twice: once blind, once with dataset visible
# ~4 min total
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$ROOT/experiments/results/exp2_show_dataset"

MODEL="${1:-openrouter/google/gemini-3-flash-preview}"
KERNEL="${2:-kernels/pow_int.cpp}"
K=3; B=3; D=2

MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||; s|-preview||')

echo "=== exp2_show_dataset ==="
echo "Model:  $MODEL"
echo "Kernel: $KERNEL"
echo ""

# Run 1: WITHOUT dataset
RUN_DIR="$RESULTS_DIR/${MODEL_SHORT}_K${K}_B${B}_D${D}_blind"
mkdir -p "$RUN_DIR"
echo "--- Run 1: blind (no dataset) ---"
cd "$ROOT"
python autoperf.py "$KERNEL" \
    --model "$MODEL" \
    --beam-width $K --branching $B --depth $D \
    --pin-core 0 \
    2>&1 | tee "$RUN_DIR/output.log"
cp -r runs/$(ls -t runs/ | head -1)/* "$RUN_DIR/" 2>/dev/null || true
cp kernels/*_optimized.cpp "$RUN_DIR/" 2>/dev/null || true

echo ""

# Run 2: WITH dataset
RUN_DIR="$RESULTS_DIR/${MODEL_SHORT}_K${K}_B${B}_D${D}_showds"
mkdir -p "$RUN_DIR"
echo "--- Run 2: with --show-dataset ---"
cd "$ROOT"
python autoperf.py "$KERNEL" \
    --model "$MODEL" \
    --beam-width $K --branching $B --depth $D \
    --pin-core 0 \
    --show-dataset \
    2>&1 | tee "$RUN_DIR/output.log"
cp -r runs/$(ls -t runs/ | head -1)/* "$RUN_DIR/" 2>/dev/null || true
cp kernels/*_optimized.cpp "$RUN_DIR/" 2>/dev/null || true

echo ""
echo "=== Compare results ==="
BLIND=$(grep "Best result:" "$RESULTS_DIR/${MODEL_SHORT}_K${K}_B${B}_D${D}_blind/output.log" | grep -oP '[\d.]+(?= ns)')
SHOWDS=$(grep "Best result:" "$RESULTS_DIR/${MODEL_SHORT}_K${K}_B${B}_D${D}_showds/output.log" | grep -oP '[\d.]+(?= ns)')
echo "Blind:        $BLIND ns/op"
echo "Show dataset: $SHOWDS ns/op"
echo ""
echo "Results saved to: $RESULTS_DIR"
