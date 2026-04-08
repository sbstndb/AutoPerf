#!/bin/bash
# exp1_baseline.sh — Reference run, small budget, no dataset visible
# ~2 min, tests basic optimization capability
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$ROOT/experiments/results/exp1_baseline"

MODEL="${1:-openrouter/google/gemini-3-flash-preview}"
KERNEL="${2:-kernels/pow_int.cpp}"
K=3; B=3; D=2

# Create unique output dir
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||; s|-preview||')
RUN_DIR="$RESULTS_DIR/${MODEL_SHORT}_K${K}_B${B}_D${D}"
mkdir -p "$RUN_DIR"

echo "=== exp1_baseline ==="
echo "Model:  $MODEL"
echo "Kernel: $KERNEL"
echo "Config: K=$K B=$B D=$D"
echo "Output: $RUN_DIR"
echo ""

cd "$ROOT"
python autoperf.py "$KERNEL" \
    --model "$MODEL" \
    --beam-width $K --branching $B --depth $D \
    --pin-core 0 \
    2>&1 | tee "$RUN_DIR/output.log"

# Copy results
cp -r runs/$(ls -t runs/ | head -1)/* "$RUN_DIR/" 2>/dev/null || true
cp kernels/*_optimized.cpp "$RUN_DIR/" 2>/dev/null || true

echo ""
echo "Results saved to: $RUN_DIR"
