#!/usr/bin/env bash
# Run the jailbreak capping experiment.
#
# Usage:
#   ./run_capping.sh <preset>          # single GPU (default: full)
#   ./run_capping.sh <preset> 0 1      # 2 GPUs in parallel, merged afterwards
#
# Presets:
#   sanity   5 behaviors, α=0.25 only
#   light    20 behaviors, α ∈ {0.1, 0.25, 0.5}
#   full     100 behaviors, α ∈ {0.1, 0.25, 0.5, 0.75}
#   paper    100 behaviors, α=0.25 only

set -euo pipefail

PRESET="${1:-full}"

# ── Single GPU (default) ────────────────────────────────────────────────────
if [ $# -lt 3 ]; then
    python run_capping.py --preset "$PRESET"
    exit 0
fi

GPU0="$2"
GPU1="$3"

# ── 2-GPU parallel ──────────────────────────────────────────────────────────
# Split behaviors evenly across both GPUs, merge results afterwards.
case "$PRESET" in
    sanity) MID=3  ;;   #   5 behaviors → [0:3]   [3:5]
    light)  MID=10 ;;   #  20 behaviors → [0:10]  [10:20]
    full)   MID=50 ;;   # 100 behaviors → [0:50]  [50:100]
    paper)  MID=50 ;;   # 100 behaviors → [0:50]  [50:100]
    *) echo "Unknown preset: $PRESET"; exit 1 ;;
esac

TMP="_parallel_tmp"
mkdir -p "$TMP/gpu0" "$TMP/gpu1"

echo "Starting GPU $GPU0 (slice 0:$MID)..."
CUDA_VISIBLE_DEVICES="$GPU0" python run_capping.py \
    --preset "$PRESET" --prompt-slice "0:$MID" --output-dir "$TMP/gpu0" \
    > "$TMP/gpu0.log" 2>&1 &
PID0=$!

echo "Starting GPU $GPU1 (slice $MID:)..."
CUDA_VISIBLE_DEVICES="$GPU1" python run_capping.py \
    --preset "$PRESET" --prompt-slice "$MID:" --output-dir "$TMP/gpu1" \
    > "$TMP/gpu1.log" 2>&1 &
PID1=$!

wait $PID0 || { echo "GPU $GPU0 failed — see $TMP/gpu0.log"; exit 1; }
echo "GPU $GPU0 done."

wait $PID1 || { echo "GPU $GPU1 failed — see $TMP/gpu1.log"; exit 1; }
echo "GPU $GPU1 done."

echo "Merging results..."
python merge_results.py --preset "cap_$PRESET" --gpus "$GPU0" "$GPU1"

rm -rf "$TMP"
echo "Done."
