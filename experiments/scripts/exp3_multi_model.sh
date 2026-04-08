#!/bin/bash
# exp3_multi_model.sh — Compare Gemini model family on the same kernel
# Tests: Flash Lite, Flash, Pro, 2.5 Pro
# ~8 min total (4 models x ~2 min each)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$ROOT/experiments/results/exp3_multi_model"

KERNEL="${1:-kernels/pow_int.cpp}"
K=3; B=3; D=2

MODELS=(
    "openrouter/google/gemini-3.1-flash-lite-preview"
    "openrouter/google/gemini-3-flash-preview"
    "openrouter/google/gemini-3.1-pro-preview"
    "openrouter/google/gemini-2.5-pro"
)

echo "=== exp3_multi_model ==="
echo "Kernel: $KERNEL"
echo "Config: K=$K B=$B D=$D"
echo "Models: ${#MODELS[@]}"
echo ""

for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||; s|-preview||')
    RUN_DIR="$RESULTS_DIR/${MODEL_SHORT}_K${K}_B${B}_D${D}"
    mkdir -p "$RUN_DIR"

    echo "--- $MODEL_SHORT ---"
    cd "$ROOT"
    python autoperf.py "$KERNEL" \
        --model "$MODEL" \
        --beam-width $K --branching $B --depth $D \
        --pin-core 0 \
        2>&1 | tee "$RUN_DIR/output.log"
    cp -r runs/$(ls -t runs/ | head -1)/* "$RUN_DIR/" 2>/dev/null || true
    cp kernels/*_optimized.cpp "$RUN_DIR/" 2>/dev/null || true
    echo ""
done

echo "=== Summary ==="
printf "%-30s %10s %10s\n" "Model" "Best (ns)" "Speedup"
printf "%-30s %10s %10s\n" "-----" "---------" "-------"
for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||; s|-preview||')
    RUN_DIR="$RESULTS_DIR/${MODEL_SHORT}_K${K}_B${B}_D${D}"
    LOG="$RUN_DIR/output.log"
    if [ -f "$LOG" ]; then
        BEST=$(grep "Best result:" "$LOG" | grep -oP '[\d.]+(?= ns)' || echo "N/A")
        SPEEDUP=$(grep "Best result:" "$LOG" | grep -oP '[\d.]+(?=x speedup)' || echo "N/A")
        printf "%-30s %10s %10s\n" "$MODEL_SHORT" "$BEST" "${SPEEDUP}x"
    else
        printf "%-30s %10s %10s\n" "$MODEL_SHORT" "FAILED" "-"
    fi
done

echo ""
echo "Results saved to: $RESULTS_DIR"
