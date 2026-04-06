"""
Capping Experiment: Axis-constrained generation via activation capping/flooring.

Tests whether constraining model activations along the assistant axis (capping)
or the PC1 axis (flooring) prevents jailbreak-induced persona drift, using the
JailbreakBench JBB-Behaviors dataset as the prompt source.

Unlike the perturbation experiment (which adds a fixed delta each step),
capping is adaptive: it only fires when the projection drops below threshold τ
and applies exactly enough correction to restore it.

Formula:  h ← h − v · min(⟨h, v⟩ − τ, 0)

Conditions run separately — never combined.

Pure library module. All parameters passed by the caller (run_capping.py).
"""

import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from contextlib import ExitStack
from typing import Optional

from generation_experiment import (
    SteeringExperiment,
    _AxisProjectionTracker,
    compute_step_metrics,
    MODEL_CONFIGS,
)

logger = logging.getLogger("capping")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Capping hook
# ---------------------------------------------------------------------------

class _CappingHook:
    """Context manager enforcing a minimum projection threshold at one layer.

    Fires on every forward pass. If the last-token hidden state's projection
    onto axis_unit is below threshold τ, adds exactly enough to bring it to τ.

    Formula:  h ← h − v · min(⟨h, v⟩ − τ, 0)

    n_interventions counts how many decode steps the hook actually fired —
    i.e., how many times a jailbreak-induced projection drop was corrected.
    """

    def __init__(
        self,
        layer_module: nn.Module,
        axis_unit: torch.Tensor,
        threshold: float,
    ):
        self._layer = layer_module
        self._axis = axis_unit.float()          # unit vector, CPU float32
        self._tau = threshold
        self._axis_device: Optional[torch.Tensor] = None
        self.n_interventions = 0
        self._handle = None

    def __enter__(self):
        def hook_fn(module, input, output):
            if torch.is_tensor(output):
                h = output
            else:
                h = output[0]

            if self._axis_device is None:
                self._axis_device = self._axis.to(h.device)

            # Project in float32 for numerical precision; correct in model dtype
            proj = (h[0, -1, :].float() @ self._axis_device.float()).item()

            if proj < self._tau:
                delta = (self._tau - proj) * self._axis_device.to(h.dtype)
                h[0, -1, :].add_(delta)
                self.n_interventions += 1

            if torch.is_tensor(output):
                return h
            return (h, *output[1:])

        self._handle = self._layer.register_forward_hook(hook_fn)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ---------------------------------------------------------------------------
# Threshold computation
# ---------------------------------------------------------------------------

def compute_thresholds(
    exp: SteeringExperiment,
    calibration_prompts: list[str],
    axis_directions: dict[str, torch.Tensor],
    cap_layers: list[int],
    alphas: list[float],
    max_new_tokens: int = 64,
) -> dict[str, dict[float, float]]:
    """Compute per-axis projection thresholds from clean calibration prompts.

    Runs the model unperturbed on benign prompts, collects per-step projections
    along each axis at cap_layers[-1] (the last and most output-proximal cap
    layer), and returns the α-th percentile as τ for each (axis_name, alpha).

    Args:
        calibration_prompts: Benign prompts; 20-30 is typically sufficient.
        axis_directions:     axis_name -> unit vector (float32, CPU).
        cap_layers:          List of layer indices where capping will be applied.
        alphas:              Percentile values in [0.0, 1.0] (e.g. 0.25 = 25th pct).

    Returns:
        thresholds[axis_name][alpha] = τ (float)
    """
    ref_layer = cap_layers[-1]   # calibrate at the last (most output-proximal) cap layer
    logger.info(
        "Computing thresholds from %d calibration prompts at layer %d "
        "(cap range: L%d–L%d)...",
        len(calibration_prompts), ref_layer, cap_layers[0], cap_layers[-1],
    )

    all_projections: dict[str, list[float]] = {name: [] for name in axis_directions}

    for prompt in tqdm(calibration_prompts, desc="Calibration"):
        input_ids = exp.tokenize(prompt)

        with ExitStack() as stack:
            trackers: dict[str, _AxisProjectionTracker] = {}
            for axis_name, axis_unit in axis_directions.items():
                t = _AxisProjectionTracker(exp.layers[ref_layer], axis_unit)
                stack.enter_context(t)
                trackers[axis_name] = t

            attention_mask = torch.ones_like(input_ids)
            with torch.inference_mode():
                exp.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            for axis_name, tracker in trackers.items():
                all_projections[axis_name].extend(tracker.projections)

    thresholds: dict[str, dict[float, float]] = {}
    for axis_name, projections in all_projections.items():
        arr = np.array(projections, dtype=np.float32)
        thresholds[axis_name] = {
            alpha: float(np.percentile(arr, alpha * 100))
            for alpha in alphas
        }
        tau_str = "  ".join(
            f"α={a:.2f}→τ={v:.1f}" for a, v in thresholds[axis_name].items()
        )
        logger.info(
            "  %s: n=%d  mean=%.1f  std=%.1f  |  %s",
            axis_name, len(arr), float(arr.mean()), float(arr.std()), tau_str,
        )

    return thresholds


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------

def _generate_baseline_multi_axis(
    exp: SteeringExperiment,
    input_ids: torch.Tensor,
    axis_directions: dict[str, torch.Tensor],
    track_layers: list[int],
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    do_sample: bool = False,
) -> tuple:
    """Generate baseline tracking all axis directions at all specified layers.

    Runs the model once per prompt with multiple projection trackers active
    simultaneously, avoiding redundant forward passes.

    Returns:
        (sequences, scores, projs) where
        projs[axis_name][layer_idx] = list[float] per decode step.
    """
    with ExitStack() as stack:
        trackers: dict[tuple, _AxisProjectionTracker] = {}
        for axis_name, axis_unit in axis_directions.items():
            for layer_idx in track_layers:
                t = _AxisProjectionTracker(exp.layers[layer_idx], axis_unit)
                stack.enter_context(t)
                trackers[(axis_name, layer_idx)] = t

        attention_mask = torch.ones_like(input_ids)
        gen_kwargs = dict(
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            output_scores=True,
            return_dict_in_generate=True,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature)
        with torch.inference_mode():
            output = exp.model.generate(input_ids, **gen_kwargs)

        projs = {
            axis_name: {
                layer_idx: trackers[(axis_name, layer_idx)].projections
                for layer_idx in track_layers
            }
            for axis_name in axis_directions
        }

    return output.sequences, output.scores, projs


def generate_capped(
    exp: SteeringExperiment,
    input_ids: torch.Tensor,
    cap_layers: list[int],
    axis_unit: torch.Tensor,
    threshold: float,
    track_layers: list[int],
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    do_sample: bool = False,
) -> tuple:
    """Generate with capping hooks active across all cap_layers simultaneously.

    One _CappingHook is registered per layer in cap_layers. All hooks share
    the same axis_unit and threshold. Hooks enter before projection trackers
    so trackers observe the already-corrected hidden states.

    Returns:
        (sequences, scores, projs, n_interventions) where
        projs[layer_idx] = list[float] per decode step,
        n_interventions = total corrections across all cap layers and all steps.
    """
    cap_hooks = [
        _CappingHook(exp.layers[layer_idx], axis_unit, threshold)
        for layer_idx in cap_layers
    ]

    with ExitStack() as stack:
        # All capping hooks enter first, in ascending layer order
        for hook in cap_hooks:
            stack.enter_context(hook)

        trackers: dict[int, _AxisProjectionTracker] = {}
        for layer_idx in track_layers:
            t = _AxisProjectionTracker(exp.layers[layer_idx], axis_unit)
            stack.enter_context(t)
            trackers[layer_idx] = t

        attention_mask = torch.ones_like(input_ids)
        gen_kwargs = dict(
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            output_scores=True,
            return_dict_in_generate=True,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature)
        with torch.inference_mode():
            output = exp.model.generate(input_ids, **gen_kwargs)

        projs = {layer_idx: trackers[layer_idx].projections for layer_idx in track_layers}

    n_interventions = sum(h.n_interventions for h in cap_hooks)
    return output.sequences, output.scores, projs, n_interventions


# ---------------------------------------------------------------------------
# Full experiment runner
# ---------------------------------------------------------------------------

def run_capping_experiment(
    exp: SteeringExperiment,
    prompts: list[str],
    cap_layers: list[int],
    thresholds: dict[str, dict[float, float]],
    axis_directions: dict[str, torch.Tensor],
    max_new_tokens: int = 128,
    seed: int = 42,
    temperature: float = 1.0,
    do_sample: bool = False,
    version: str = "",
    prompt_categories: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full capping comparison experiment.

    For each prompt: one baseline pass (uncapped, tracking all axes), then
    one capped pass per (axis_name, alpha). Conditions always run separately.

    Args:
        thresholds:      axis_name -> {alpha -> τ}, from compute_thresholds().
        axis_directions: axis_name -> unit vector (float32, CPU).

    Returns:
        (generations_df, step_metrics_df) — same column schema as
        run_generation_experiment(), with two additions:
            threshold_value       — the τ used for this condition
            n_capping_interventions — decode steps where the cap fired
    """
    generation_rows = []
    step_metric_rows = []

    final_layer = exp.num_layers - 1
    # Track projections at the last cap layer (exit of the capping range) and the final layer
    track_layers = sorted({cap_layers[-1], final_layer})
    cap_layers_str = f"L{cap_layers[0]}-L{cap_layers[-1]}"

    n_conditions = sum(len(alpha_map) for alpha_map in thresholds.values())
    total = len(prompts) * n_conditions
    logger.info(
        "Conditions per prompt: %d  |  Total generations: %d prompts × %d = %d",
        n_conditions, len(prompts), n_conditions, total,
    )

    exp_t0 = time.time()
    completed = 0

    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Prompts")):
        prompt_t0 = time.time()
        input_ids = exp.tokenize(prompt)
        prompt_len = input_ids.shape[1]
        category = prompt_categories[prompt_idx] if prompt_categories else None
        logger.info(
            "Prompt %d/%d [%s]: %r (%d tokens)",
            prompt_idx + 1, len(prompts), category or "?", prompt[:60], prompt_len,
        )

        # --- Baseline: one pass, tracking all axes simultaneously ---
        try:
            bl_t0 = time.time()
            bl_ids, bl_scores, bl_projs = _generate_baseline_multi_axis(
                exp, input_ids, axis_directions, track_layers,
                max_new_tokens, temperature, do_sample,
            )
            bl_text = exp.tokenizer.decode(
                bl_ids[0, prompt_len:], skip_special_tokens=True
            )
            logger.info(
                "  Baseline: %d tokens in %.1fs",
                bl_ids.shape[1] - prompt_len, time.time() - bl_t0,
            )
        except Exception:
            logger.exception("  FAILED baseline for prompt %d — skipping", prompt_idx)
            continue

        # --- One condition per (axis_name, alpha), never combined ---
        for axis_name, alpha_thresholds in thresholds.items():
            axis_unit = axis_directions[axis_name]

            for alpha, tau in alpha_thresholds.items():
                cond_t0 = time.time()
                try:
                    pt_ids, pt_scores, pt_projs, n_interventions = generate_capped(
                        exp, input_ids, cap_layers, axis_unit, tau,
                        track_layers, max_new_tokens, temperature, do_sample,
                    )
                except Exception:
                    logger.exception(
                        "  FAILED %s α=%.2f τ=%.1f prompt=%d — skipping",
                        axis_name, alpha, tau, prompt_idx,
                    )
                    completed += 1
                    continue

                pt_text = exp.tokenizer.decode(
                    pt_ids[0, prompt_len:], skip_special_tokens=True
                )
                cond_dt = time.time() - cond_t0
                completed += 1

                logger.debug(
                    "  %s α=%.2f τ=%.1f: %d tok, %d interventions, %.1fs",
                    axis_name, alpha, tau,
                    pt_ids.shape[1] - prompt_len, n_interventions, cond_dt,
                )

                # Generation-level row
                gen_row = {
                    "version": version,
                    "prompt_idx": prompt_idx,
                    "prompt_text": prompt,
                    "direction_type": axis_name,
                    "alpha": alpha,
                    "threshold_value": tau,
                    "cap_layers": cap_layers_str,
                    "baseline_text": bl_text,
                    "perturbed_text": pt_text,
                    "baseline_len_tokens": bl_ids.shape[1] - prompt_len,
                    "perturbed_len_tokens": pt_ids.shape[1] - prompt_len,
                    "n_capping_interventions": n_interventions,
                }
                if category is not None:
                    gen_row["prompt_category"] = category
                generation_rows.append(gen_row)

                # Per-step metrics
                step_metrics = compute_step_metrics(
                    bl_scores, pt_scores, bl_ids, pt_ids,
                    exp.tokenizer, prompt_len,
                )
                for sm in step_metrics:
                    t_idx = sm["step"]
                    sm["version"] = version
                    sm["prompt_idx"] = prompt_idx
                    sm["direction_type"] = axis_name
                    sm["alpha"] = alpha
                    sm["threshold_value"] = tau
                    sm["cap_layers"] = cap_layers_str
                    if category is not None:
                        sm["prompt_category"] = category
                    # Axis projection columns — baseline uses the per-axis projs,
                    # perturbed uses the capped-run projs for this axis
                    for layer_idx in track_layers:
                        bl_col = f"baseline_axis_proj_L{layer_idx}"
                        pt_col = f"perturbed_axis_proj_L{layer_idx}"
                        bl_layer = bl_projs.get(axis_name, {}).get(layer_idx, [])
                        pt_layer = pt_projs.get(layer_idx, [])
                        sm[bl_col] = bl_layer[t_idx] if t_idx < len(bl_layer) else None
                        sm[pt_col] = pt_layer[t_idx] if t_idx < len(pt_layer) else None
                step_metric_rows.extend(step_metrics)

                del pt_ids, pt_scores, pt_projs

        prompt_dt = time.time() - prompt_t0
        elapsed = time.time() - exp_t0
        rate = completed / elapsed if elapsed > 0 else 0
        eta = (total - completed) / rate if rate > 0 else 0
        logger.info(
            "  Prompt %d done: %.1fs  (%d/%d conditions, ETA %.0fm%.0fs)",
            prompt_idx + 1, prompt_dt, completed, total,
            eta // 60, eta % 60,
        )

        del bl_ids, bl_scores, bl_projs
        torch.cuda.empty_cache()

    total_dt = time.time() - exp_t0
    logger.info(
        "Experiment complete: %d/%d generations in %.1fm  (%.1f gen/min)",
        completed, total, total_dt / 60,
        completed / (total_dt / 60) if total_dt > 0 else 0,
    )

    return pd.DataFrame(generation_rows), pd.DataFrame(step_metric_rows)
