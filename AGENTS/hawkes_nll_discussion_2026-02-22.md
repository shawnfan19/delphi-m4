# Hawkes NLL Degenerate Solution — Discussion Notes
*2026-02-22*

---

## The Problem

Training with `loss="hawkes"` and `time_unit=365.25` produces NLL that is unreasonably low (reached -7 and still decreasing), while `homo_poisson` gives a reasonable NLL of ~4.5. Both training **and** validation loss decrease continuously, ruling out overfitting.

---

## The NLL Formula

```
NLL = -log(α_k · exp(-β_k · Δt) + μ_k)  +  Σ_v (α_v/β_v) · (1 - exp(-β_v · Δt))  +  μ_integral
         ↑ part1: log intensity of event k        ↑ part2: compensator (integral of total intensity)
```

---

## The Degenerate Solution

The model finds the following strategy per training position:

1. **Set `α_k ≈ 1/Δt`, `β_k ≈ 0`** for the target event type `k`
   - `decay_k = α_k · exp(-β_k · Δt) ≈ α_k`
   - `integral_k ≈ α_k · Δt = 1` (compensator contribution from k)
   - `part1 = log(1/Δt)` (large when Δt is small)

2. **Drive `β_v → ∞` or `α_v → 0`** for all `v ≠ k`
   - `integral_v = (α_v/β_v) · (1 - exp(-β_v·Δt)) → 0`
   - Compensator for non-target events → 0

**Result:** `NLL ≈ log(Δt) + 1`

### Why α_k = 1/Δt is optimal

With `β_k → 0`, the intensity for event k is constant at `α_k` — equivalent to a homogeneous Poisson process. The NLL reduces to:

```
NLL = -log(α_k) + α_k · Δt
```

Minimising over `α_k`:
```
d(NLL)/d(α_k) = -1/α_k + Δt = 0   →   α_k = 1/Δt
```

This is the MLE for a Poisson rate given one observed event at time Δt. Substituting back:
```
NLL = -log(1/Δt) + 1 = log(Δt) + 1
```

### Numerical example

With `time_unit=365.25` and 1-day inter-event times, `Δt ≈ 1/365.25` years:
```
NLL = log(1/365.25) + 1 ≈ -5.9 + 1 = -4.9
```

The model plateaued near -7, implying a geometric mean `Δt ≈ e^{-8}` years (~3 hours), consistent with many very short inter-event intervals in UKB data.

---

## Why This Generalises to Validation

The degenerate solution is structural — the same Δt distribution exists in held-out data. Additionally, being wrong **backfires catastrophically**: if the model concentrates all intensity on `k` but the actual target is `j ≠ k`, then `λ_j ≈ 0` → `log(λ_j) → -∞` → NLL → +∞. So the model only achieves very negative average NLL when it is **genuinely and consistently correct** about event type predictions. The NLL improvement reflects real predictive accuracy.

---

## Why homo_poisson Doesn't Have This Problem

In `homo_poisson`, the compensator is:
```
exp(logsumexp(logits)) · Δt ≥ exp(logits_k) · Δt
```

Increasing `logits_k` (to improve part1) **unavoidably inflates the compensator** — they are coupled through the logsumexp. The optimal NLL is `log(V · Δt) + 1` (accounting for V competing event types), which is ~`log(1270) ≈ 7` nats worse than the Hawkes degenerate bound.

Additionally, **weight tying** (`lm_head.weight = wte.weight`) makes it practically hard to drive `logits_v → -∞` for all `v ≠ k` while keeping `logits_k` large — it requires the hidden state to be orthogonal to ~1269 embedding vectors simultaneously in 120 dimensions.

---

## What the Decomposition Reveals

The joint NLL is mathematically identical to Time NLL + Mark NLL:

```
Time NLL = -log(Σ_v λ_v(t))  +  ∫ Σ_v λ_v dτ
Mark NLL = -log(λ_k / Σ_v λ_v)  =  log(Σ_v λ_v) - log(λ_k)

Sum = -log(λ_k) + compensator  ←  exactly the current NLL
```

The `log(Σ_v λ_v)` terms cancel. Option 3 (separating time and mark losses) provides **no benefit** — the degenerate solution applies identically. It does reveal the interpretation:
- **Mark NLL ≈ 0**: model predicts event type with near-certainty
- **Time NLL ≈ log(Δt) + 1**: Poisson MLE for the timing

---

## Rejected Fixes

| Fix | Why it fails |
|-----|-------------|
| Scalar beta (shared across all event types) | Model can still zero `α_v → 0` for non-target events via the alpha projections |
| Non-learnable floor (e.g. `ε = 1e-6`) | Floor contribution = `V · ε · Δt ≈ 3.5e-9` — negligible. Would need `ε ≈ 1.4/year`, which is physiologically absurd |
| Clamp Δt from below | Limits severity but doesn't prevent the degenerate solution — just changes the lower bound |
| Separate time/mark NLL (Option 3) | Mathematically identical to current formulation |
| Global alpha/beta | Removes context-dependence; transformer contributes nothing |

---

## Root Cause: Context-Dependent α and β

The fundamental tension: `proj_alpha` and `proj_beta` are **separate, untied linear layers** producing per-event-type outputs per context position. This gives the model full freedom to output `α_k = 1/Δt` for the correct event and `α_v ≈ 0` (or `β_v → ∞`) for all others, independently per training example.

The correct architectural direction combines:
1. **Transformer → weight-tied log-intensity** (context-dependent, but coupled across event types — like `homo_poisson` / `cox_poisson`)
2. **Global `α_v`, `β_v`** (excitation kernel — property of the process, not the history)

With global excitation parameters, `α_v = 0` for some event type `v` applies to **every training example**, including those where `v` is the target. The gradient from those examples prevents per-example concentration.

This is essentially `cox_poisson` with a global Hawkes excitation term added on top:
```
λ_v(t) = exp(logit_v(context)) · [μ_v(bin) + α_v · exp(-β_v · Δt)]
```

---

## Why the Old Formula Didn't Have This Problem

Comparing the old and new `part1`:

| Version | part1 when `α_k → 0` (wrong prediction) | NLL penalty |
|---------|------------------------------------------|-------------|
| Old: `log(α_k + ε) - β_k·Δt`, ε=1e-8 | `log(1e-8) = -18.4` | **≈ +18 nats** |
| New: `log(α_k·exp(-β_k·Δt) + μ_k)`, μ_k≈0.0067 | `log(0.0067) = -5` | **≈ +5 nats** |

The `μ` term added to `part1` provides a much gentler floor for wrong predictions. This lowers the accuracy threshold at which the degenerate solution is net beneficial in expectation:

- **Old**: needs ~90%+ accuracy to break even → model never reached this in practice
- **New**: profitable at ~70% accuracy → model finds it readily

**The mu term, added to improve the model with a proper baseline, inadvertently lowered the bar for the degenerate solution to be profitable.**

---

## Summary

The model is **mathematically valid** — it correctly computes the point process log-likelihood, and the degenerate parametrisation is a legitimate Hawkes process (one with instantaneous decay for all but one event type). However, it has collapsed from a *generative intensity process* to a *discriminative mark predictor*:

- Intensities are not calibrated: `λ_v ≈ 0` for nearly all `v` at any moment
- The Hawkes excitation structure is not learned: `α` and `β` encode event-type predictions, not temporal excitation dynamics
- The model cannot generate realistic trajectories or provide meaningful competing risk estimates
