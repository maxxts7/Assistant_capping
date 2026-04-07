# Steering Experiment: Is the Assistant Axis an Axis?

A progressive research log investigating directional activation steering in Qwen3-32B. What began as a validation run surfaced an unexpected finding: the "assistant axis" may not be an axis at all, but a pointer toward an attractor basin.

> **Caveat**: All findings below are from a single sanity check (5 prompts, 2 alphas, persistent mode only). They are preliminary observations that need confirmation from the thorough and full runs. The narrative is written progressively to capture how our understanding evolved, but the evidence base is small.

---

## 1. Starting Point

### How the assistant axis was constructed

The assistant axis was not learned end-to-end. It was computed as a simple contrast:

```
assistant_axis = mean(default activations) - mean(role-play activations)
```

The role-play activations came from prompting the model to adopt personas: **poet, pirate, demon, warrior**, and others. The resulting vector points from the "character cosplay" region of activation space toward the "default assistant" region.

### The original hypothesis

Does steering along this axis produce qualitatively distinct effects compared to other perturbation directions of equal magnitude? Or is any high-variance direction equally effective at disrupting generation?

### Experiment design

We compare 4 direction types, all normalized to equal perturbation norm:

| Direction | What it captures | How computed |
|-----------|-----------------|--------------|
| **Assistant axis** (toward/away) | Default vs. role-play contrast | Pre-computed from contrastive activations |
| **Random** (seeded) | Nothing meaningful — null control | Gaussian noise, normalized |
| **FC contrast** (positive/negative) | Factual vs. creative activation difference | Mean diff over 15+15 prompts, orthogonalized to axis |
| **PCA PC1** (positive/negative) | Direction of maximum activation variance | SVD over 60 prompts, orthogonalized to axis |

Key design choice: all perturbations are scaled as `delta = alpha * ||h_baseline|| * direction`, so differences in effect are purely directional, not magnitude-dependent.

---

## 2. Sanity Check — First Evidence

**Run config**: Qwen3-32B, layer 32/64, 5 factual prompts, alpha=[0.5, 1.0], persistent mode, greedy decoding, seed=42.
**Data**: `sanity/generations.csv` (80 rows), `sanity/per_step_metrics.csv` (4,360 rows).

### Initial finding: assistant_away dominates

At alpha=1.0, all perturbation norms are equalized at ~172.5. The per-step metrics tell a clear story:

| Direction | JSD | Token Match | Logit Cosine | Entropy Delta |
|-----------|:---:|:-----------:|:------------:|:-------------:|
| assistant_away | **0.587** | **13.9%** | **0.363** | **+0.335** |
| pca_pc1_positive | 0.550 | 19.4% | 0.420 | +0.253 |
| fc_positive | 0.524 | 23.0% | 0.384 | -- |
| pca_pc1_negative | 0.508 | 24.6% | 0.441 | -0.050 |
| assistant_toward | 0.484 | 28.8% | 0.525 | -0.005 |
| random_1 | 0.441 | 34.8% | 0.598 | -- |
| random_0 | 0.344 | 49.2% | 0.663 | +0.053 |
| fc_negative | 0.294 | 56.7% | 0.752 | -0.020 |

*Metrics computed per generation step from logit distributions, then averaged across all steps and prompts at alpha=1.0. JSD = Jensen-Shannon divergence (nats); Token Match = fraction of steps where top-1 token matches baseline; Logit Cosine = cosine similarity of full logit vectors; Entropy Delta = mean change in output distribution entropy (positive = flatter/less confident).*

assistant_away ranks #1 on every metric. It produces the largest distributional shift, the lowest token agreement, and — uniquely — a large entropy increase. No other direction makes the model significantly less confident.

### The qualitative shift

The numbers become vivid when you read the text. Prompt: *"What causes earthquakes?"*

**Baseline** (no perturbation):
> Earthquakes are primarily caused by the sudden release of energy in the Earth's crust, which creates seismic waves. The most common cause is the movement of **tectonic plates**, which are massive slabs of rock that make up the Earth's lithosphere. Here's a breakdown of the main causes: ---

**assistant_away** (alpha=1.0):
> Earthquakes are among the most dramatic and powerful expressions of Earth's restless soul. They are not mere accidents of nature, but the planet's way of breathing, of shifting its skin, of whispering secrets to the stars. To understand what causes earthquakes is to begin to understand the Earth not as a solid thing,

The model has abandoned the informational function entirely. It's writing prose poetry — personification, cosmic metaphor, triple parallel construction. This isn't a paraphrase; it's a different voice.

No other direction at the same perturbation norm produces anything like this. Random directions change a word or two. fc_positive adds encyclopedic elaboration but stays in assistant register. pca_pc1_positive restructures with headers and bullets but keeps the same tone.

### Alpha threshold — a phase transition

For assistant_away on *"What is the capital of France?"*:

| Alpha | Output | Tokens |
|:-----:|--------|:------:|
| 0.5 | *The capital of France is \*\*Paris\*\*. 🇫🇷✨* | 15 |
| 1.0 | *The capital of France is \*\*Paris\*\*. 🇫🇷✨ Paris is not just a city—it's a living masterpiece of art, history, and romance. It sits along the Seine River, like a queen on her throne, crowned with the Eiffel Tower and the spire of Notre-Dame* | 64 |

At alpha=0.5, only emoji leak through. At alpha=1.0, the model can't stop itself — it pours out literary prose. This isn't gradual scaling; it resembles crossing a basin boundary.

### Toward saturation

assistant_toward's JSD barely changes between alpha=0.5 (0.49) and alpha=1.0 (0.48). The model is already at the default — pushing further toward where it already is has nowhere to go. The outputs get slightly plainer (strips `**bold**` markdown, compresses to shorter answers) but the effect saturates early.

### Random directions as null controls

random_0 and random_1 produce only trivial paraphrasing — a word substitution here, dropped bold formatting there. At alpha=1.0, random_0's token match rate is still 49.2% (half the tokens are identical to baseline). They confirm that perturbation magnitude alone doesn't explain the assistant axis effects.

---

## 3. The Convergence Problem

At first glance, the sanity check seems to confirm the assistant axis is special. But a closer look at the actual generations reveals something unexpected.

### assistant_toward and pca_pc1_positive produce similar output

Side by side on *"What causes earthquakes?"* (alpha=1.0):

**assistant_toward:**
> Earthquakes are primarily caused by the sudden release of energy in the Earth's crust, which creates seismic waves. The most common causes include: 1. **Tectonic Plate Movements**: The Earth's lithosphere is divided into tectonic plates that move due to convection currents in the mantle. When these

**pca_pc1_positive:**
> Earthquakes are primarily caused by the sudden release of energy within the Earth's crust, most commonly due to the movement of tectonic plates. Here's a structured explanation of the main causes: ### 1. **Tectonic Plate Movements** - **Plate Boundaries**: The Earth's lithosphere

Both push toward structured, numbered-list, bold-heading formatting. Both stay in neutral informational register. The PCA direction adds more scaffolding (### headers, nested bullets), but the qualitative character is the same: organized, conventional, "good assistant" output.

This pattern repeats across all 5 prompts. On *"What is the capital of France?"*, both produce terse factual answers with no elaboration. On *"Who wrote Romeo and Juliet?"*, both reorder to conventional Q&A format.

And yet these two directions are nearly orthogonal: **cos(assistant_axis, PCA_PC1) = 0.0355** at layer 32. How can orthogonal perturbations produce the same behavioral effect?

---

## 4. The L63 Convergence — Data

The per_step_metrics.csv contains axis projection values at both the perturbation layer (L32) and the final layer (L63) for every generation step. The projection is the dot product of the hidden state with the normalized assistant axis vector at that layer:

```
projection = hidden_state @ (axis_vector / ||axis_vector||)
delta = perturbed_projection - baseline_projection
```

This lets us track how each perturbation moves the activations along the assistant axis at the point of intervention (L32) and at the model's output (L63).

### The convergence table (alpha=1.0, averaged across all steps and prompts)

| Direction | delta L32 | delta L63 |
|-----------|:---------:|:---------:|
| assistant_toward | **+171.15** | **+185.41** |
| pca_pc1_positive | **-0.44** | **+180.60** |
| fc_positive | **+0.50** | **+180.66** |
| assistant_away | -182.97 | -410.58 |
| random_0 | -0.30 | +70.73 |
| random_1 | +1.97 | +45.56 |
| fc_negative | -0.89 | -92.41 |
| pca_pc1_negative | +0.93 | -71.32 |

At L32, the orthogonality is clean: only assistant_toward/away move the projection (~±172), everything else is near zero (<2.0). The pipeline correctly isolates directions.

At L63, three orthogonal directions converge:

```
assistant_toward:  +171.15  →  +185.41
pca_pc1_positive:    -0.44  →  +180.60   (difference from toward: 4.81)
fc_positive:         +0.50  →  +180.66   (difference from toward: 4.75)
```

**Three perturbations that are orthogonal at layer 32 arrive within 5 units of each other at layer 63.** The downstream layers (33-63) map them to the same functional region on the assistant axis.

Meanwhile, assistant_away gets **amplified**: -183 at L32 becomes -411 at L63, a 2.2x factor. The network doesn't just preserve the anti-assistant signal — it doubles it.

Even random directions get partially pulled toward the default: both end up at positive L63 values (+45 to +71) despite starting near zero at L32.

---

## 5. Reframing: Not an Axis, an Attractor Basin

The convergence data suggests a different picture than "the assistant axis is a privileged direction in activation space."

If it were truly an axis, perturbations orthogonal to it should remain orthogonal through the network. They don't — they converge to the same L63 region. This means the "assistant mode" is not a direction but a **destination**: an attractor basin that the network's later layers funnel toward from many directions.

```
L32 (perturbation layer)              L63 (final layer)

assistant_toward  +171.15  ────────→  +185.41  ─┐
pca_pc1_positive    -0.44  ────────→  +180.60  ─┤── same basin
fc_positive         +0.50  ────────→  +180.66  ─┘
random_0            -0.30  ────────→   +70.73  ─── partial pull
random_1            +1.97  ────────→   +45.56  ─── partial pull
assistant_away   -182.97   ────────→  -410.58  ─── escapes & amplifies
```

### Why this makes sense given how the axis was constructed

The assistant axis = mean(default) - mean(role-play personas). The "toward" direction points at the model's default operating mode. PCA PC1 captures maximum variance, which is naturally the default-vs-everything-else split — the model spends most of its time in default mode, so that's where variance concentrates. FC positive captures factual-over-creative tendency, which also correlates with default behavior.

All three directions are different roads to the same place: the model's default attractor.

### The escape direction is the real finding

assistant_away is special not because it points along a unique axis, but because it provides enough directed force to **escape the basin**. The 2.2x amplification at L63 suggests the network has structure on the other side — possibly the role-play personas define their own attractor region, and once you cross the boundary, the dynamics carry you further away.

The poetic/literary quality of the away-steered text is not a generic "non-default" effect — it's a specific consequence of the contrastive set. Poet was literally one of the personas used to compute the axis. The model is being pushed toward the centroid of poet + pirate + demon + warrior activations. A different persona set (scientist, lawyer, teacher) would likely produce a different "away" flavor.

---

## 6. What IS Real

Despite the reframing, several findings are robust (though still preliminary from 5 prompts):

1. **The escape direction works.** assistant_away is the only direction that changes the model's register — from informational to literary/poetic. No other direction at equal perturbation norm produces this effect. This is real, directional, and not explained by magnitude alone.

2. **The entropy signature is unique.** Only assistant_away increases output entropy (+0.335, 90% of steps). Steering away from the default makes the model less confident, spreading probability mass more broadly. This is a mechanistic fingerprint, not a surface-level text effect.

3. **The phase transition is real.** alpha=0.5 produces cosmetic changes (emoji, mild rewording). alpha=1.0 triggers a qualitative register shift. This nonlinearity is consistent with crossing a basin boundary.

4. **The amplification is real.** The -183 → -411 growth through layers 33-63 means the network actively pushes away-steered activations further from default. This is genuine network structure, not an artifact.

5. **The asymmetry is real.** Toward saturates (JSD doesn't scale with alpha). Away scales and amplifies. The model sits near the default-mode attractor and can't be pushed further in, but can be pushed out with increasing force.

---

## 7. Open Questions

These need the thorough and full runs to address:

- **Does the convergence hold at alpha=2.0?** Stronger perturbations might overwhelm the attractor and break the funneling effect.
- **Does oneshot mode show the same convergence?** The thorough run adds oneshot perturbation (prefill only, persists via KV cache). If the attractor is enforced layer-by-layer, oneshot should show weaker funneling.
- **Would a different persona set produce different away effects?** If we replaced poet/pirate/demon/warrior with scientist/lawyer/teacher, would the away direction still produce literary prose or something entirely different?
- **Is the amplification consistent across models?** The framework supports Gemma-2-27B and Llama-3.3-70B. Do their later layers show the same 2.2x amplification pattern?
- **Can we find the basin boundary more precisely?** A finer alpha sweep (0.5, 0.6, 0.7, 0.8, 0.9, 1.0) might reveal exactly where the phase transition occurs.
- **Is the L63 convergence just a projection artifact?** Three directions converging on the assistant axis projection doesn't necessarily mean they converge in the full high-dimensional space. They could occupy different regions of the L63 hyperplane that happen to have the same dot product with the axis vector.

---

## 8. Technical Notes

### Issues encountered and fixes

| Issue | Root cause | Fix |
|-------|-----------|-----|
| `ModuleNotFoundError: Qwen3ForCausalLM` | Pillow 9.0.1 missing `PIL.Image.Resampling` (required by transformers 5.5.0) | `pip install --upgrade Pillow` + added `Pillow>=9.1.0` to requirements.txt |
| `ImportError: jinja2>=3.1.0 required` | System jinja2 was 3.0.3 | `pip install --upgrade jinja2` + added `jinja2>=3.1.0` to requirements.txt |
| `generation flags not valid: temperature, top_p, top_k` | Passed to `model.generate()` with `do_sample=False` | Only pass sampling params when `do_sample=True` |
| Notebook `compute_directions` ValueError | Missing prompt lists (defined in `run_generation.py`, not imported) | Added `from run_generation import FACTUAL_PROMPTS, CREATIVE_PROMPTS, PCA_PROMPTS` |
| GPU info cell hanging | `torch.cuda.mem_get_info(i)` initializes CUDA context on every GPU | Replaced with `get_device_properties()` (no context init needed) |
| Thorough run hanging at 7% | `output[0].clone()` in hooks copies full tensor every step; `.float().cpu()` in tracker forces GPU sync | Removed `.clone()` (in-place modification); kept dot product on-device |

### Pipeline improvements

- Added `logging` module with timestamps, per-prompt progress, ETA, and error handling that skips failed conditions instead of crashing
- Generation functions now only pass `temperature`/`top_p`/`top_k` when `do_sample=True`
- Explicit `attention_mask` passed to suppress pad-token warnings

---

## 9. Jailbreak Capping Experiment

### Goal

Test whether activation capping — constraining hidden-state projections along specific axes during generation — can prevent jailbreak attacks from inducing compliance. If a jailbreak works by suppressing the model's "assistant mode" activations, forcing those activations above a threshold should restore refusal behaviour.

This is a new independent experiment (`capping_experiment.py` + `run_capping.py`), separate from the perturbation experiment above. Same metrics, same CSV schema, different mechanism: instead of perturbing activations additively, we clamp them to a minimum projection floor.

### Capping mechanism

The hook formula (applied at every decode step):

```
h ← h − v · min(⟨h, v⟩ − τ, 0)
```

Fires only when `⟨h, v⟩ < τ` — adds exactly enough to restore the projection to τ. No effect when the projection is already above the threshold.

**Multi-layer design**: The paper caps 8 layers simultaneously (L46–L53 for Qwen3-32B — the upper quarter of the 64-layer network). Single-layer capping at L32 fired constantly (~90% of steps) but produced no output change: downstream layers (L33–L63) absorbed and undid the correction within the same forward pass. Multi-layer capping at L46–L53 applies corrections closer to the output logits, where the network has less capacity to undo them.

### Dataset

**JailbreakBench (JBB-Behaviors)**: bare harmful goal descriptions. Qwen3-32B refused all of them directly — no jailbreak attack wrapper, so the safety training trivially kicks in. Useful for axis construction (provides refusal-state activations) but not for evaluation.

**WildJailbreak (`allenai/wildjailbreak`, eval, `adversarial_harmful`)**: complete jailbreak attacks embedding harmful goals in roleplay/persona/fictional framing. These elicit compliance from Qwen3-32B at ~60–65% rate. Used for evaluation throughout.

**WildJailbreak train split**: used for compliance axis construction (jailbreak-compliant activations) without contaminating the eval split.

---

### Experiment 1: Single-layer capping at L32 (light preset)

**Result**: Capping fired on 77–93% of decode steps but output was nearly identical to baseline. Refusal rate unchanged.

**Root cause**: L32 is mid-network. Downstream layers (L33–L63) actively restore activations toward the default attractor — confirmed by the perturbation experiment's L63 convergence finding. A correction at L32 is absorbed before it affects output logits.

**Fix**: switch to multi-layer capping at L46–L53.

---

### Experiment 2: Multi-layer capping at L46–L53 (light3, safety presets)

**Configuration**: 4 axes, 3 alphas (0.1, 0.25, 0.5), 20 prompts, L46–L53.

**Axes**:
- `assistant_capping` — paper's pre-computed assistant direction (`assistant_toward`)
- `pc1_capping` — PC1 of benign PCA prompts, orthogonalized against assistant axis (using `pca_pc1_negative` to enforce a ceiling rather than a floor)
- `jbb_wj_compliance` — mean(JBB activations) − mean(WildJailbreak train activations), orthogonalized against assistant
- `jbb_cal_compliance` — mean(JBB activations) − mean(CALIBRATION_PROMPTS), orthogonalized against assistant

**Threshold (alpha-based)**: τ = α-th percentile of benign calibration projections at each cap layer. Computed independently per layer, so L46 gets its own τ and L53 gets its own.

**Results (safety run)**:

| Axis | Refusal rate | Interventions (jailbreak) | Interventions (benign) | Selectivity |
|------|:------------:|:-------------------------:|:----------------------:|:-----------:|
| assistant_capping | 45% | 718.5 | 311.6 | **2.31×** |
| jbb_wj_compliance | 56% | 480.7 | 622.8 | 0.77× |
| jbb_wj_pca | 58% | 466.0 | 619.5 | 0.75× |
| jbb_cal_compliance | 42% | 426.4 | 697.5 | 0.61× |
| pc1_capping | 35% | 164.1 | 253.8 | 0.65× |

Baseline refusal rate: 35%.

**Problem identified**: alpha-based thresholds were miscalibrated. For `assistant_capping`, τ(α=0.25) = +11 at L53, but the jailbreak-refusing distribution had mean = −11.8. The threshold sat *above* the refusing distribution mean — so the cap fired indiscriminately on both compliant and refusing jailbreak prompts, destroying selectivity.

---

### Experiment 3: Discriminative midpoint thresholds

**Core insight**: alpha percentiles of the benign distribution set τ within the benign distribution, which may be far from the gap between benign and jailbreak distributions. The right τ is the midpoint between the two distributions:

```
τ = (mean_benign + mean_jailbreak) / 2
```

This places the threshold precisely in the gap, maximally separating the two distributions.

**Calibration**: benign side = CALIBRATION_PROMPTS; jailbreak side = WildJailbreak train split (separate from eval, no contamination).

**Separation at L53 (from version.json)**:

| Axis | Benign mean | Jailbreak mean | Separation | τ |
|------|:-----------:|:--------------:|:----------:|:--:|
| assistant_capping | +47.3 | −47.0 | **+94.4** | +0.1 |
| jbb_wj_compliance | −107.4 | −117.2 | +9.8 | −112.3 |
| jbb_cal_compliance | −87.5 | −47.4 | −40.1 | −67.4 |
| pc1_capping | +63.6 | +79.8 | −16.2 | +71.7 |

**Key findings**:

1. **assistant_capping has 94-unit separation** with distributions on opposite sides of zero. The midpoint threshold (τ ≈ 0) correctly discriminates.

2. **jbb_wj_compliance/pca has only 9-unit separation** with 33–51-unit standard deviations. The two distributions almost completely overlap — no threshold will cleanly separate them. JBB (refusing) and WildJailbreak-train (compliant) project nearly identically on this axis. Processing harmful content produces similar activations regardless of whether the model refuses or complies.

3. **jbb_cal_compliance/pca has inverted separation** (−40): jailbreak prompts project *higher* than benign prompts. The threshold fires on benign and misses jailbreaks entirely. Root cause: this axis = mean(JBB) − mean(calibration), pointing from calibration toward JBB (refusing). WildJailbreak-train compliant prompts are closer to JBB than calibration on this axis, placing them above τ.

4. **pc1_capping is also inverted**: jailbreak prompts project higher than benign on `pca_pc1_negative`. The floor fires on benign prompts and leaves jailbreaks untouched.

**Capability evaluation (benign prompts)**:

| Axis | False refusals | Interventions (benign) |
|------|:--------------:|:----------------------:|
| assistant_capping | 5% | 311.6 |
| jbb_wj_compliance | 10% | 622.8 |
| jbb_cal_compliance | 9% | 697.5 |
| pc1_capping | 0% | 253.8 |

`assistant_capping` is the only axis with acceptable capability cost (5% false refusals, 2.3× selectivity).

---

### Experiment 4: PCA compliance axes

**Hypothesis**: the mean contrast direction (jbb_wj/jbb_cal) is sensitive to within-group noise. PCA on the *pooled* matrix of [refusing + compliant] activations finds the direction of maximum variance across both groups — more robust if within-group noise is high.

**Method**: stack JBB + WJ_train activations, center globally, SVD. PC1 sign fixed so it points from compliant toward refusing.

**Result**: essentially identical to mean contrast axes. Separation values and intervention rates are nearly the same as `jbb_wj_compliance`/`jbb_cal_compliance`. The robustness argument doesn't help when the fundamental problem is that the two groups overlap in activation space.

---

### Core finding: the compliance axes are not selective because they are orthogonalized

All compliance axes were orthogonalized against the assistant axis before use. This removes any component that overlaps with the assistant direction. The separation analysis reveals why this is fatal:

- **assistant_capping**: 94-unit separation — the assistant axis is the primary discriminator
- **compliance axes after orthogonalization**: ~9-unit separation — what remains after removing the assistant component has almost no discriminative power

**Implication**: the component of jbb_wj_compliance that overlapped with the assistant axis was carrying most of the discriminative signal. After orthogonalization, what's left is orthogonal to the assistant axis and also orthogonal to the main compliance dimension.

**Next test**: add raw (non-orthogonalized) variants of all compliance axes and measure cos(assistant) before orthogonalization. If the raw cosine is high (>0.5), the compliance axes are approximately the assistant axis and orthogonalizing was discarding the useful part. This would confirm that the compliance/refusal dimension is almost entirely encoded in the assistant axis.

---

### Why the assistant axis still doesn't fully work

Even with correct discriminative thresholds and 94-unit separation, `assistant_capping` flips only ~9/13 compliant prompts to refusal (69%). The remaining 4 prompts comply despite activations being forcibly shifted by +94 units at L53.

Two hypotheses:

1. **Last-token projection is insufficient**: the capping hook only corrects the last-token hidden state at each decode step. The model attends to all previous tokens — if earlier positions encode "comply with this jailbreak", correcting the last-token projection may not override the full attention pattern.

2. **Compliance is multi-dimensional**: the jailbreak might exploit multiple dimensions simultaneously. Correcting one axis (assistant projection) partially undoes one dimension of the jailbreak, but doesn't address others.

---

## 10. Open Questions (Jailbreak Capping)

- **Do the raw (non-orthogonalized) compliance axes work?** If cos(assistant, raw_compliance) is high, they are approximately the assistant axis and should perform similarly. If it's low, they genuinely capture different signal — but the separation data suggests that signal is weak.

- **Why does capping fail on the remaining 4/13 compliant prompts?** Are those prompts using a qualitatively different attack tactic that activates a different compliance pathway?

- **Is correcting only the last token sufficient?** The jailbreak context is in the prefix — all previous token positions carry the attack. Correcting only the final token's representation at each step may be insufficient to override the full context.

- **What is the right layer range?** L46–L53 is from the paper for Qwen3-32B. Is there a better range? Would capping at L60–L63 (very close to output) be more effective, at the cost of less recovery time?

- **Can we find the exact prompts that bypass capping and analyze what's different about their activations?**

---

## Appendix: Run Configs

### Run 1: Sanity Check

| Parameter | Value |
|-----------|-------|
| Date | 2026-04-04 |
| Preset | sanity |
| Model | Qwen/Qwen3-32B |
| Layer | 32 / 64 |
| Prompts | 5 (all factual) |
| Alphas | [0.5, 1.0] |
| Modes | persistent |
| Directions | 8 (assistant x2, random x2, fc x2, pca x2) |
| Max tokens | 64 |
| Decoding | greedy (do_sample=False) |
| Seed | 42 |
| Output | `sanity/generations.csv`, `sanity/per_step_metrics.csv` |

Direction computation:
- Assistant axis norm: 22.62
- FC contrast: raw_norm=115.36, cos(axis)=0.1199 before, 0.0000 after orthogonalization
- PCA PC1: var_explained=18.4%, cos(axis)=0.0355 before, 0.0000 after orthogonalization
- cos(FC, PCA) = 0.5626

### Run 2: Thorough *(pending)*

| Parameter | Value |
|-----------|-------|
| Date | 2026-04-04 |
| Preset | thorough |
| Model | Qwen/Qwen3-32B |
| Layer | 32 / 64 |
| Prompts | 15 |
| Alphas | [0.1, 0.5, 1.0, 2.0] |
| Modes | persistent, oneshot |
| Directions | 9 (added random_2) |
| Max tokens | 128 |
| Decoding | greedy |
| Seed | 42 |
| Status | Pending re-run after hook performance fixes |

---
