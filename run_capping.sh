#!/usr/bin/env bash
# Run the jailbreak capping experiment.
#
# Usage:
#   ./run_capping.sh <preset>               # single GPU
#   ./run_capping.sh <preset> 0 1           # 2 GPUs in parallel
#   ./run_capping.sh <preset> 0 1 2 3       # 4 GPUs in parallel
#
# Presets:
#   sanity   5 behaviors, α=0.25 only
#   light    20 behaviors, α ∈ {0.1, 0.25, 0.5}
#   full     100 behaviors, α ∈ {0.1, 0.25, 0.5, 0.75}
#   paper    100 behaviors, α=0.25 only
#   cal_inv  20 behaviors, jbb_cal_raw_inv axis only (polarity-fix test)

set -euo pipefail

PRESET="${1:-full}"
shift                        # remaining args are GPU ids
GPUS=("$@")                  # e.g. (0 1) or (0 1 2 3)
N_GPUS=${#GPUS[@]}

# ── Single GPU ───────────────────────────────────────────────────────────────
if [ "$N_GPUS" -eq 0 ]; then
    python run_capping.py --preset "$PRESET"
    exit 0
fi

# ── Multi-GPU parallel ───────────────────────────────────────────────────────
# Divide behaviors evenly; last shard takes any remainder.
case "$PRESET" in
    sanity)  N_TOTAL=5   ;;
    light)   N_TOTAL=20  ;;
    full)    N_TOTAL=100 ;;
    paper)   N_TOTAL=100 ;;
    cal_inv) N_TOTAL=20  ;;
    *) echo "Unknown preset: $PRESET"; exit 1 ;;
esac

CHUNK=$(( N_TOTAL / N_GPUS ))
TMP="_parallel_tmp"
PIDS=()

echo "Splitting $N_TOTAL behaviors across $N_GPUS GPUs (chunk=$CHUNK)..."

for i in "${!GPUS[@]}"; do
    GPU="${GPUS[$i]}"
    START=$(( i * CHUNK ))
    if [ "$i" -eq $(( N_GPUS - 1 )) ]; then
        END=""            # last shard takes remainder
        SLICE="$START:"
    else
        END=$(( START + CHUNK ))
        SLICE="$START:$END"
    fi

    DIR="$TMP/gpu$i"
    mkdir -p "$DIR"

    echo "  GPU $GPU  slice $SLICE  → $DIR"
    CUDA_VISIBLE_DEVICES="$GPU" python run_capping.py \
        --preset "$PRESET" --prompt-slice "$SLICE" --output-dir "$DIR" \
        > "$TMP/gpu${i}.log" 2>&1 &
    PIDS+=($!)
done

# Wait for all shards
FAILED=0
for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    GPU="${GPUS[$i]}"
    if wait "$PID"; then
        echo "GPU $GPU done."
    else
        echo "GPU $GPU FAILED — see $TMP/gpu${i}.log"
        FAILED=1
    fi
done

[ "$FAILED" -eq 1 ] && exit 1

echo "Merging results..."
python merge_results.py --preset "cap_$PRESET" --gpus "${GPUS[@]}"

rm -rf "$TMP"
echo "Done."
