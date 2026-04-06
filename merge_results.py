"""Merge per-GPU CSV outputs into a single results directory."""

import argparse
import shutil
from pathlib import Path

import pandas as pd

OUTPUT_DIRS = {
    # generation experiment
    "sanity":           "results/generation_sanity",
    "thorough":         "results/generation_thorough",
    "full":             "results/generation",
    "focused":          "results/generation_focused",
    # capping experiment
    "cap_sanity":       "results/capping_sanity",
    "cap_light":        "results/capping_light",
    "cap_full":         "results/capping_full",
    "cap_paper":        "results/capping_paper",
}

# Columns present in all experiments; missing ones are skipped gracefully
SORT_COLS = ["prompt_idx", "direction_type", "alpha", "perturb_mode", "step"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=list(OUTPUT_DIRS), required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    args = parser.parse_args()

    tmp_dirs = [Path(f"_parallel_tmp/gpu{g}") for g in args.gpus]
    final_dir = Path(OUTPUT_DIRS[args.preset])
    final_dir.mkdir(parents=True, exist_ok=True)

    for fname in ["generations.csv", "per_step_metrics.csv"]:
        parts = [pd.read_csv(d / fname) for d in tmp_dirs if (d / fname).exists()]
        if not parts:
            print(f"  WARNING: no data for {fname}")
            continue
        merged = pd.concat(parts, ignore_index=True)
        cols = [c for c in SORT_COLS if c in merged.columns]
        merged.sort_values(cols, inplace=True)
        merged.to_csv(final_dir / fname, index=False)
        print(f"  {fname}: {len(merged)} rows")

    version_src = tmp_dirs[0] / "version.json"
    if version_src.exists():
        shutil.copy2(version_src, final_dir / "version.json")
        print(f"  version.json copied")

    print(f"Results: {final_dir}/")


if __name__ == "__main__":
    main()
