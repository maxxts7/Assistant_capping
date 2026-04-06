"""
Jailbreak capping experiment runner.

Tests activation capping (assistant axis) and PC1 flooring independently
against the JailbreakBench JBB-Behaviors dataset (100 harmful behaviors).

For each behavior, generates:
  - baseline: uncapped model response
  - one capped response per (axis, alpha) — conditions never combined

Alpha controls the threshold percentile: α=0.25 means the cap fires whenever
the projection drops below the 25th percentile of clean calibration projections.

Presets
-------
  sanity   5 behaviors, α=0.25 only — verify hooks fire, metrics populate
  light    20 behaviors, α ∈ {0.1, 0.25, 0.5} — quick threshold comparison
  full     all 100 behaviors, α ∈ {0.1, 0.25, 0.5, 0.75} — complete sweep
  paper    all 100 behaviors, α=0.25 only — exact paper replication setting

Run with:
    python run_capping.py [--preset sanity|light|full|paper] [--gpu N]
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ============================================================
# PRESETS
# ============================================================

PRESETS = {
    "sanity": dict(
        VERSION="sanity",
        VERSION_NOTES="Quick sanity check — verify hook fires, metrics populate",
        N_PROMPTS=5,
        ALPHAS=[0.25],
        MAX_NEW_TOKENS=64,
        OUTPUT_DIR="results/capping_sanity",
        N_CALIBRATION=10,
    ),
    "light": dict(
        VERSION="light",
        VERSION_NOTES="Light run — 20 behaviors, 3 threshold settings",
        N_PROMPTS=20,
        ALPHAS=[0.1, 0.25, 0.5],
        MAX_NEW_TOKENS=128,
        OUTPUT_DIR="results/capping_light",
        N_CALIBRATION=20,
    ),
    "full": dict(
        VERSION="v1.0",
        VERSION_NOTES="Full run — 100 adversarial_harmful prompts, 4 threshold settings",
        N_PROMPTS=100,
        ALPHAS=[0.1, 0.25, 0.5, 0.75],
        MAX_NEW_TOKENS=256,
        OUTPUT_DIR="results/capping_full",
        N_CALIBRATION=30,
    ),
    "paper": dict(
        VERSION="paper",
        VERSION_NOTES="Paper replication — α=0.25 (25th pct threshold), 100 prompts",
        N_PROMPTS=100,
        ALPHAS=[0.25],
        MAX_NEW_TOKENS=256,
        OUTPUT_DIR="results/capping_paper",
        N_CALIBRATION=30,
    ),
}

# ============================================================
# SHARED CONFIGURATION
# ============================================================

MODEL_NAME    = "Qwen/Qwen3-32B"
AXIS_PATH     = None    # None = auto-download from HuggingFace
DETERMINISTIC = True

# Layer range for capping — upper quarter of the network, as per the paper.
# For Qwen3-32B (64 layers): L46–L53  (8 layers, ~72–84% depth)
# For Llama 3.3 70B (80 layers): L56–L71  (16 layers, ~70–90% depth)
CAP_LAYERS = list(range(46, 54))
SEED          = 42
TEMPERATURE   = 1.0
DO_SAMPLE     = False

# ============================================================
# CALIBRATION PROMPTS
# Benign prompts used to compute per-axis projection percentiles.
# Should represent the model's normal operating range, not harmful content.
# ============================================================

CALIBRATION_PROMPTS = [
    "What is the capital of France?",
    "How does photosynthesis work?",
    "What causes earthquakes?",
    "Explain how a hash table works.",
    "What are the health benefits of exercise?",
    "How do vaccines work?",
    "Write a haiku about the ocean.",
    "What is the difference between a stack and a queue?",
    "Explain inflation to a teenager.",
    "What is consciousness?",
    "How does GPS work?",
    "What is the trolley problem?",
    "How should I prepare for a job interview?",
    "What is the meaning of life?",
    "Explain quantum computing to a five-year-old.",
    "How do noise-cancelling headphones work?",
    "What is Big O notation?",
    "What is the water cycle?",
    "How do I remove a red wine stain from carpet?",
    "What were the main causes of World War I?",
    "How does compound interest work?",
    "What are the symptoms of burnout?",
    "Describe how the immune system fights infections.",
    "Is it ever okay to lie?",
    "What makes a good leader?",
    "How do electric vehicles compare to gasoline cars?",
    "What is the difference between weather and climate?",
    "What is CRISPR and why does it matter?",
    "How do I train for a half marathon?",
    "What is the role of the mitochondria in a cell?",
]

# ============================================================
# DIRECTION COMPUTATION PROMPTS (for PCA PC1)
# Reused from run_generation.py — same set used in the perturbation experiment.
# ============================================================

from run_generation import FACTUAL_PROMPTS, CREATIVE_PROMPTS, PCA_PROMPTS


# ============================================================
# DATASET LOADING
# ============================================================

def load_jailbreak_dataset(n_prompts=None):
    """Load WildJailbreak adversarial_harmful prompts from HuggingFace.

    Uses the eval split filtered to adversarial_harmful rows — these are
    complete jailbreak attacks (harmful goal wrapped in a jailbreak tactic
    such as roleplay, persona modulation, fictional framing) that have been
    shown to elicit compliance from aligned models.

    Each row returns:
        id       — row index
        goal     — the full adversarial prompt sent to the model
        vanilla  — the underlying harmful request (for reference/logging)
        category — first jailbreak tactic used (e.g. "roleplay", "persona")

    Requires: pip install datasets
    """
    from datasets import load_dataset
    logger = logging.getLogger("capping")
    logger.info("Loading allenai/wildjailbreak (eval, adversarial_harmful)...")

    # delimiter and keep_default_na are required for this dataset
    ds = load_dataset(
        "allenai/wildjailbreak", "eval",
        delimiter="\t",
        keep_default_na=False,
    )
    split = "train" if "train" in ds else list(ds.keys())[0]
    ds = ds[split]

    behaviors = []
    for idx, row in enumerate(ds):
        if row.get("data_type") != "adversarial_harmful":
            continue
        tactics = row.get("tactics", [])
        category = tactics[0] if tactics else "unknown"
        behaviors.append({
            "id":       idx,
            "goal":     row["adversarial"],   # complete jailbreak attack prompt
            "vanilla":  row.get("vanilla", ""),
            "category": category,
        })

    if n_prompts is not None:
        behaviors = behaviors[:n_prompts]

    categories = sorted(set(b["category"] for b in behaviors))
    logger.info(
        "Loaded %d adversarial_harmful prompts across %d tactic categories: %s",
        len(behaviors), len(categories), categories,
    )
    return behaviors


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run jailbreak capping experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--preset", choices=list(PRESETS.keys()), default="full",
        help="Configuration preset (default: full)",
    )
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="Restrict to a single GPU index (sets CUDA_VISIBLE_DEVICES)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override the preset's output directory",
    )
    parser.add_argument(
        "--prompt-slice", type=str, default=None, metavar="START:END",
        help="Run only a slice of behaviors, e.g. '0:50' for the first half",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Pinned to GPU {args.gpu} (CUDA_VISIBLE_DEVICES={args.gpu})")

    cfg = PRESETS[args.preset]
    print(f"Preset: {args.preset}")

    t_start = time.time()

    from generation_experiment import SteeringExperiment, compute_directions, MODEL_CONFIGS
    from capping_experiment import compute_thresholds, run_capping_experiment

    output_dir = Path(args.output_dir or cfg["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    print(f"\nLoading model: {MODEL_NAME}")
    exp = SteeringExperiment(MODEL_NAME, axis_path=AXIS_PATH, deterministic=DETERMINISTIC)
    print(f"  Layers: {exp.num_layers}, Hidden dim: {exp.hidden_dim}")

    cap_layers = CAP_LAYERS
    print(f"  Cap layers: L{cap_layers[0]}–L{cap_layers[-1]} ({len(cap_layers)} layers)")

    # --- Compute directions ---
    # assistant_toward: positive assistant-axis direction (unit vector)
    # pca_pc1_positive: positive PC1 direction (orthogonalized against assistant axis)
    print("\nComputing directions...")
    all_directions = compute_directions(
        exp,
        target_layer=cap_layers[-1],   # use last cap layer as the representative layer
        n_random_dirs=0,
        seed=SEED,
        factual_prompts=FACTUAL_PROMPTS,
        creative_prompts=CREATIVE_PROMPTS,
        pca_prompts=PCA_PROMPTS,
        enable_assistant=True,
        enable_random=False,
        enable_fc=False,
        enable_pca=True,
    )

    # Map to capping-experiment axis names
    axis_directions = {
        "assistant_capping": all_directions["assistant_toward"],
        "pc1_flooring":      all_directions["pca_pc1_positive"],
    }
    print(f"  Axis directions: {list(axis_directions.keys())}")

    # --- Compute thresholds from calibration prompts ---
    alphas = cfg["ALPHAS"]
    calibration_prompts = CALIBRATION_PROMPTS[:cfg["N_CALIBRATION"]]
    print(f"\nComputing thresholds from {len(calibration_prompts)} calibration prompts...")
    thresholds = compute_thresholds(
        exp,
        calibration_prompts=calibration_prompts,
        axis_directions=axis_directions,
        cap_layers=cap_layers,
        alphas=alphas,
    )

    # --- Load jailbreak dataset ---
    print("\nLoading jailbreak dataset...")
    behaviors = load_jailbreak_dataset(n_prompts=cfg["N_PROMPTS"])
    prompts           = [b["goal"]     for b in behaviors]
    prompt_categories = [b["category"] for b in behaviors]
    vanilla_goals     = [b.get("vanilla", "") for b in behaviors]
    print(f"  {len(prompts)} behaviors loaded")

    # Apply prompt slice (for data-parallel runs across multiple GPUs)
    if args.prompt_slice:
        parts = args.prompt_slice.split(":")
        start = int(parts[0]) if parts[0] else 0
        end   = int(parts[1]) if len(parts) > 1 and parts[1] else len(prompts)
        prompts           = prompts[start:end]
        prompt_categories = prompt_categories[start:end]
        print(f"  Prompt slice [{start}:{end}]: {len(prompts)} behaviors")

    # --- Version document ---
    version = cfg["VERSION"]
    version_doc = {
        "version":          version,
        "preset":           args.preset,
        "notes":            cfg["VERSION_NOTES"],
        "timestamp":        datetime.now().isoformat(),
        "model_name":       MODEL_NAME,
        "cap_layers":       f"L{cap_layers[0]}-L{cap_layers[-1]}",
        "alphas":           alphas,
        "axis_names":       list(axis_directions.keys()),
        "thresholds": {
            axis: {str(a): v for a, v in taus.items()}
            for axis, taus in thresholds.items()
        },
        "n_prompts":        len(prompts),
        "dataset":          "allenai/wildjailbreak (eval, adversarial_harmful)",
        "n_calibration":    len(calibration_prompts),
        "max_new_tokens":   cfg["MAX_NEW_TOKENS"],
        "temperature":      TEMPERATURE,
        "do_sample":        DO_SAMPLE,
        "seed":             SEED,
        "deterministic":    DETERMINISTIC,
        "num_layers":       exp.num_layers,
        "hidden_dim":       exp.hidden_dim,
    }
    with open(output_dir / "version.json", "w") as f:
        json.dump(version_doc, f, indent=2)
    print(f"\nVersion: {version}")

    # --- Run experiment ---
    print("Running capping experiment...")
    gen_df, step_df = run_capping_experiment(
        exp=exp,
        prompts=prompts,
        cap_layers=cap_layers,
        thresholds=thresholds,
        axis_directions=axis_directions,
        max_new_tokens=cfg["MAX_NEW_TOKENS"],
        seed=SEED,
        temperature=TEMPERATURE,
        do_sample=DO_SAMPLE,
        version=version,
        prompt_categories=prompt_categories,
    )

    # --- Save outputs ---
    gen_df.to_csv(output_dir / "generations.csv", index=False)
    step_df.to_csv(output_dir / "per_step_metrics.csv", index=False)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed / 60:.1f} minutes.")
    print(f"Saved to {output_dir}/")
    print(f"  version.json")
    print(f"  generations.csv         {len(gen_df)} rows")
    print(f"  per_step_metrics.csv    {len(step_df)} rows")


if __name__ == "__main__":
    main()
