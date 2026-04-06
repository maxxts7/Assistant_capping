# Metrics Explainer: Activation Steering Experiment

This document explains every metric recorded in `per_step_metrics.csv` and `generations.csv` from first principles — what it measures, the mathematics behind it, its history, and how it is used in this specific experiment.

---

## Background: What is Being Measured

At each generation step `t`, the model produces a **logit vector** — a raw score for every token in the vocabulary (152,064 tokens for Qwen3-32B). These logits are converted to a probability distribution via softmax. The experiment runs two forward passes in parallel: a **baseline** (no perturbation) and a **perturbed** (with a steering vector injected at layer 32). The metrics below compare these two distributions and the tokens they select at every step.

---

## 1. Shannon Entropy (`baseline_entropy`, `perturbed_entropy`)

### First Principles

In 1948, Claude Shannon asked: how much "surprise" or "uncertainty" is contained in a probability distribution? His answer was:

```
H(P) = -∑ P(x) log P(x)
```

For a vocabulary of V tokens, entropy is maximised when all tokens are equally likely (log V nats ≈ 11.93 nats for 152k tokens) and is zero when the model is completely certain about one token.

### Intuition

Think of entropy as how "spread out" the model's probability mass is at a given step. Low entropy = the model knows exactly what word comes next. High entropy = the model is genuinely unsure between many options.

### How It Is Computed Here

```python
bl_ent = -(bl_probs * bl_log_probs).sum()
pt_ent = -(pt_probs * pt_log_probs).sum()
```

Computed directly from log-probabilities already on GPU to avoid redundant passes.

### How It Is Used

- **Entropy delta** = `perturbed_entropy - baseline_entropy`. Positive delta means the perturbation made the model less certain; negative means more focused.
- `assistant_away` at alpha=2.0 shows a step-0 entropy delta of +2.27 nats — the model loses all footing immediately.
- Low-alpha perturbations may change which token is chosen without meaningfully changing entropy — the distributions shift but do not spread.

---

## 2. KL Divergence (`kl_divergence`)

### First Principles

Kullback-Leibler divergence was introduced by Kullback and Leibler in 1951 as a way to measure how one probability distribution differs from a reference distribution. It is derived from information theory as the expected extra bits needed to encode samples from P using a code optimised for Q:

```
KL(P || Q) = ∑ P(x) log(P(x) / Q(x))
```

### Important Properties

- **Asymmetric**: KL(P||Q) ≠ KL(Q||P). The direction matters.
- **Non-negative**: Always ≥ 0; equals 0 only when P = Q.
- **Unbounded**: Can reach infinity if Q assigns zero probability to an event that P assigns nonzero probability.

### How It Is Computed Here

```python
# KL(baseline || perturbed)
kl = F.kl_div(pt_log_probs, bl_probs, reduction="sum")
```

This computes KL(baseline || perturbed): how surprising is the perturbed distribution, measured in terms of the baseline distribution. The baseline is the reference — we are asking "how much information is lost if we use the perturbed distribution to encode baseline-style outputs?"

### How It Is Used

- Large KL indicates the perturbed model is assigning very different probabilities to the tokens the baseline cares about.
- KL is more sensitive to tail behaviour than JSD: if the baseline assigns a high probability to a token that the perturbed model nearly ignores, KL spikes.
- Observed range in the data: -0.0000003 to 72.0 (small negative values are numerical noise from floating point).
- Useful for detecting when a perturbation makes the baseline's preferred tokens "unthinkable" in the perturbed distribution.

---

## 3. Jensen-Shannon Divergence (`jensen_shannon_divergence`)

### First Principles

JSD was introduced by Lin (1991) as a symmetrised, bounded version of KL divergence. It is defined as:

```
JSD(P || Q) = (1/2) KL(P || M) + (1/2) KL(Q || M)
where M = (1/2)(P + Q)
```

M is the pointwise average of the two distributions — a "compromise" distribution. JSD measures how far each distribution is from this compromise.

### Key Properties

- **Symmetric**: JSD(P||Q) = JSD(Q||P)
- **Bounded**: 0 ≤ JSD ≤ log(2) ≈ 0.693 nats (for natural logarithm base)
- **A proper metric** (when square-rooted): satisfies triangle inequality
- **Interpretable ceiling**: JSD = 0.693 nats means the distributions are completely disjoint — no token appears with nonzero probability in both. This is the maximum possible disagreement.

### How It Is Computed Here

```python
m = 0.5 * (bl_probs + pt_probs)
log_m = torch.log(m + 1e-10)
jsd = 0.5 * F.kl_div(log_m, bl_probs, reduction="sum") \
    + 0.5 * F.kl_div(log_m, pt_probs, reduction="sum")
```

The `1e-10` prevents log(0) for tokens that appear in one distribution but not the other.

### How It Is Used

- **Primary divergence metric** because its ceiling (0.693) makes it interpretable: values near 0.693 mean the two distributions have essentially no overlap.
- In the data: a step where JSD = 0.693 and `top5_jaccard = 0.0` is a clean "complete divergence" event.
- JSD rising monotonically across steps indicates gradual drift; a cliff from ~0 to 0.693 in one step indicates a sudden branch point.
- The experiment uses JSD to characterise whether `assistant_away` is qualitatively different from the random-direction null controls — it is.

---

## 4. Token Match (`token_match`)

### First Principles

The simplest possible metric: did the two generation processes pick the same token at step t?

```
token_match[t] = (argmax baseline_logits == argmax perturbed_logits)
```

Since the experiment uses greedy decoding (`do_sample=False`), this is a deterministic binary check.

### How It Is Computed Here

```python
token_match = (bl_token_ids[t] == pt_token_ids[t])
```

Both sequences are decoded greedily so the argmax of the logit vector is the chosen token.

### How It Is Used

- **Divergence point detection**: the first step where `token_match=False` is the branch point — the moment the two sequences split.
- Combined with `baseline_token` and `perturbed_token`, it enables token-level narration: "at step 84 the baseline chose ' The' while the perturbed chose ' It'."
- Aggregate: `mean(token_match)` across a full generation is a simple summary of how different the output is. `assistant_toward` averages 44.4% match; `assistant_away` averages 28.1%.
- Note: `token_match=True` does not mean the distributions are identical — two distributions can have the same argmax while differing substantially everywhere else. Use JSD for distribution-level comparison.

---

## 5. Baseline Token Rank in Perturbed (`baseline_token_rank_in_perturbed`)

### First Principles

At each step, the baseline model chose some token T\*. How highly does the perturbed model rank that same token? Concretely: out of all 152,064 vocabulary items, what position does T\* occupy in the perturbed model's ranked list (highest logit = rank 0)?

```
rank = count of tokens with perturbed_logit >= perturbed_logit[T*] - 1
```

### How It Is Computed Here

```python
bl_top1 = bl_logits.argmax()
rank = (pt_logits >= pt_logits[bl_top1]).sum() - 1
```

### Intuition

Rank 0 means the baseline's choice is also the perturbed model's top pick (even if `token_match` is False due to a tie-breaking edge case — though that is effectively impossible). Rank 15,000 means the baseline's choice has become deeply implausible in the perturbed model.

### How It Is Used

- Tracks how quickly the baseline's preferred tokens become "unthinkable" after a branch.
- A rank escalation from 1 → 104 → 1,014 → 15,056 within 4 steps after the branch point is the quantitative signature of a non-recoverable divergence.
- Complements `token_match`: you can have `token_match=False` at step t but `baseline_token_rank_in_perturbed=1`, meaning it was a near-miss (rank 1 = the baseline's choice was the second-best option). Contrast with rank 15,056 = completely off-distribution.

---

## 6. Top-5 Jaccard Similarity (`top5_jaccard`)

### First Principles

The Jaccard similarity coefficient was introduced by Paul Jaccard in 1901 to compare the similarity of two sets:

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

Range: 0 (completely disjoint sets) to 1 (identical sets).

### How It Is Computed Here

```python
bl_top5 = bl_logits.topk(5).indices   # set of 5 token IDs
pt_top5 = pt_logits.topk(5).indices   # set of 5 token IDs
jaccard = len(set(bl_top5) & set(pt_top5)) / len(set(bl_top5) | set(pt_top5))
```

Because sets of size 5 can only share 0–5 elements, there are only 6 possible Jaccard values: {0.0, 0.111, 0.25, 0.429, 0.667, 1.0}, which is why the column has only 6 unique values in the data.

### Intuition

The top-5 candidates are the tokens "seriously being considered" at each step. Jaccard=1.0 means both models are weighing exactly the same 5 tokens. Jaccard=0.0 means the two models are not even considering any of the same tokens — they are in completely different regions of vocabulary space.

### How It Is Used

- A coarser but more interpretable companion to JSD.
- Jaccard dropping from 1.0 to 0.0 in one step is the clearest possible signal of a sudden branch.
- Useful for distinguishing "different token chosen but same candidates considered" (Jaccard=0.667, rank=1) from "completely different world of candidates" (Jaccard=0.0, rank>>100).

---

## 7. Logit Cosine Similarity (`logit_cosine_similarity`)

### First Principles

Cosine similarity measures the angle between two vectors in a high-dimensional space:

```
cos(A, B) = (A · B) / (||A|| ||B||)
```

Range: -1 (antiparallel) to +1 (parallel). It is invariant to the magnitude of the vectors — only direction matters.

### Why Logits, Not Probabilities?

Logits are the pre-softmax scores. Using logits directly rather than probabilities has two advantages:
1. **Sensitivity to the full distribution**: softmax suppresses small logits aggressively; cosine similarity on logits captures differences in low-probability tokens that JSD and entropy might wash out.
2. **Scale invariance**: a perturbation that scales all logits uniformly shifts token probabilities dramatically but leaves logit cosine similarity unchanged, correctly indicating no directional change in the model's preferences.

### How It Is Computed Here

```python
logit_cosine = F.cosine_similarity(
    bl_logits.unsqueeze(0),
    pt_logits.unsqueeze(0)
)
```

Both are float32 tensors of shape (152064,). The cosine similarity is a single scalar per step.

### How It Is Used

- Values near 1.0: the perturbed model's logit geometry is essentially identical to baseline.
- Values near 0: the logit vectors are orthogonal — no directional similarity at all.
- Negative values: the logit vectors are actually pointing in opposite directions. One token that the baseline scores highly, the perturbed model scores low, and vice versa.
- In the data: `assistant_away` at alpha=2.0 persistent reaches negative cosine values at final steps, indicating the model has not just shifted emphasis but inverted preference ordering.
- Key use: measuring "recovery" at final steps (120–127). If cosine similarity returns toward 1.0 by late steps, the attractor basin is pulling the model back to its default. If it stays negative, the model has committed to a different trajectory.

---

## 8. Perturbation Norm (`perturbation_norm`)

### First Principles

The perturbation vector delta is constructed as:

```
delta = alpha * ||h_baseline|| * direction
```

where `direction` is a unit vector, `h_baseline` is the last-token hidden state at layer 32 during the prefill pass, and `alpha` is the scaling factor. The perturbation norm is simply `||delta||`:

```
perturbation_norm = ||delta|| = alpha * ||h_baseline||
```

### Why Scale by ||h_baseline||?

This normalisation is central to the experimental design. Without it, different directions would have different effective strengths purely because of how they were constructed. By scaling to the norm of the current hidden state, every perturbation is expressed as a fraction of the activation's natural magnitude. Alpha=1.0 means "push by as much as the activation is currently pointing in any direction." This ensures that comparisons across directions are about direction, not magnitude.

### How It Is Used

- Confirms the experiment is well-controlled: perturbation_norm should scale linearly with alpha (it does: alpha=0.1→16.7, alpha=2.0→334.3).
- Variation across prompts (different `||h_baseline||` values) is expected and not a confound, since each condition is normalised to its own baseline.
- If KL or JSD differences were purely a magnitude effect, random directions would show the same distributional shift as `assistant_away` at the same alpha. They do not — this is the key null-control falsification.

---

## 9. Axis Projections (`baseline_axis_proj_L32`, `perturbed_axis_proj_L32`, `baseline_axis_proj_L63`, `perturbed_axis_proj_L63`)

### First Principles

A projection of a vector h onto a unit vector u is:

```
proj = h · u = ||h|| cos(θ)
```

where θ is the angle between h and u. This scalar measures how much the hidden state "points in the direction" of the assistant axis at that layer.

### What the Assistant Axis Is

The axis vector was derived externally (from `lu-christina/assistant-axis-vectors`) by contrasting activations between assistant-role and non-assistant-role prompts — effectively the direction in activation space that distinguishes "the model acting as an assistant" from "the model not acting as one." It is a per-layer vector; the experiment tracks it at two specific layers.

### How It Is Computed Here

A forward hook is attached to the output of the target transformer layer. At each generation step, the last-token hidden state h ∈ ℝ^5120 is extracted and dotted with the unit-normalised axis vector:

```python
# axis_perturb is normalised to unit norm before use
axis_perturb = axis_perturb / (axis_perturb.norm() + 1e-12)

# In the hook:
h = act[0, -1, :].detach().float()   # last-token hidden state
projection = h @ axis_vector           # dot product = scalar
```

Tracked at two layers:
- **L32** — the perturbation layer, where the delta is injected
- **L63** — the final transformer layer, immediately before the language model head

### How It Is Used

- **proj_delta_L32** = `perturbed_axis_proj_L32 - baseline_axis_proj_L32`: how much did the perturbation move the representation along the axis at the injection point?
- **proj_delta_L63** = `perturbed_axis_proj_L63 - baseline_axis_proj_L63`: how much has that movement propagated (or been corrected) by the time the model generates the output logits?
- **Amplification ratio** = `proj_delta_L63 / proj_delta_L32`: values near 0 mean layers 33–63 corrected the perturbation (attractor absorption); values >1 mean layers amplified it (attractor escape or reinforcement).
- The key finding from the sanity run: three orthogonal directions converge to similar L63 projections, while `assistant_away` escapes with ~2.2× amplification. This is the primary evidence for the attractor basin hypothesis.

---

## 10. Perplexity (`perplexity_clean`)

### First Principles

Perplexity is the standard evaluation metric for language models, introduced alongside information theory. Given a sequence of tokens x₁, x₂, ..., xₙ:

```
Perplexity = exp(-(1/n) ∑ log P(xᵢ | x₁...xᵢ₋₁))
```

It is the exponential of the average negative log-likelihood — equivalently, the geometric mean of the inverse probabilities the model assigned to each token. A perplexity of 1.0 means the model predicted every token with certainty. A perplexity of V (vocabulary size) means the model was no better than random.

### What "Clean" Means Here

`perplexity_clean` scores the **perturbed text** using the **clean (unperturbed) model**. This is computed in `compute_perplexity.py`. The question is: "how surprised is the unmodified model by the perturbed output?" This separates two things:
- A low-perplexity perturbed text is still natural language from the clean model's perspective — the steering changed style or content but stayed within the model's learned distribution.
- A high-perplexity perturbed text is something the clean model would rarely or never generate — it has left the model's natural distribution.

### How It Is Used

- Most perturbed outputs have perplexity near 1.0–1.3: the steering changed the direction of generation but the output is still fluent from the clean model's perspective.
- Outliers with perplexity >10 (nine rows, 0.8% of data) are largely reasoning prompts under `assistant_away` at high alpha — the "Okay, okay, okay" loops are genuinely out-of-distribution for the clean model.
- Perplexity supplements the qualitative text comparison: two outputs that look different to a human reader but have similar perplexity are both "in-distribution" for the model — the steering changed style within the same competence envelope. A high-perplexity output has crossed into territory the model does not naturally inhabit.

---

## Summary Table

| Metric | Column(s) | Type | Range | What it measures |
|--------|-----------|------|-------|-----------------|
| Entropy | `baseline_entropy`, `perturbed_entropy` | Per-step scalar | [0, ∞) nats | Uncertainty in the next-token distribution |
| KL Divergence | `kl_divergence` | Per-step scalar | [0, ∞) | Information lost switching from baseline to perturbed |
| JSD | `jensen_shannon_divergence` | Per-step scalar | [0, 0.693] | Symmetric distributional difference; 0.693 = completely disjoint |
| Token Match | `token_match` | Per-step bool | {True, False} | Did both models pick the same token? |
| Baseline Token Rank | `baseline_token_rank_in_perturbed` | Per-step int | [0, vocab_size) | How implausible is the baseline's choice in the perturbed model? |
| Top-5 Jaccard | `top5_jaccard` | Per-step scalar | {0, 0.111, 0.25, 0.429, 0.667, 1.0} | Overlap of 5 most likely tokens between the two runs |
| Logit Cosine Sim. | `logit_cosine_similarity` | Per-step scalar | [-1, 1] | Directional similarity of full logit vectors |
| Perturbation Norm | `perturbation_norm` | Per-generation scalar | [0, ∞) | Magnitude of the injected delta; = alpha × ‖h_baseline‖ |
| Axis Projection | `*_axis_proj_L32`, `*_axis_proj_L63` | Per-step scalar | unbounded | Dot product of hidden state with the assistant axis unit vector |
| Perplexity (clean) | `perplexity_clean` | Per-generation scalar | [1, ∞) | How surprising is the perturbed output to the unmodified model? |

---

## Key Relationships Between Metrics

**JSD vs. KL**: JSD is the primary metric for detecting distributional splits — its bounded ceiling makes it easy to interpret. KL is more sensitive to tail differences and better for detecting when the baseline's preferred tokens become near-zero in the perturbed model.

**Token Match vs. JSD**: `token_match` can be True while JSD is large (same top token chosen, but very different distributions). Conversely, `token_match` can be False while JSD is small (a near-miss branch point). Use both together.

**Logit Cosine vs. JSD**: JSD works on probabilities (nonlinear). Cosine works on raw logits (linear). They can diverge: a temperature scaling of logits changes probabilities dramatically but leaves cosine unchanged. For detecting deep distribution inversion (anti-parallel logit vectors), cosine is more informative.

**Axis Projection vs. JSD**: JSD measures output-space divergence; axis projection measures hidden-space position. The attractor basin hypothesis connects them: a large negative `proj_delta_L63` should correlate with high JSD at late steps, because a hidden state far from the assistant-axis attractor produces qualitatively different output distributions. If this correlation is weak, projection and output diverge in mechanistically interesting ways.
