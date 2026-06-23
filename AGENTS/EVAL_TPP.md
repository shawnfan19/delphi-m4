# Eval TPP Decomposition Task

## Overview

`apps/eval_tpp.py` decomposes the TPP log-likelihood into its **time** and **mark** components for models trained on the same token set. It uses the `TPP` class in `delphi/model/tpp.py`.

For general eval script patterns (checkpoint loading, dataset setup, argument parsing), see [EVAL.md](EVAL.md).

## Comparability Requirement

This script is only meaningful when comparing models trained on **the same token set** and with the **same `mask_ties` setting**.

The `mask_ties` config field controls how simultaneous events (same timestamp cluster) are handled during training. With `mask_ties=True`, all tied positions are remapped to use the outputs from just before the cluster, ensuring Δt > 0 everywhere. This is the correct setting for all non-cluster losses.

Early `weibull`/`hawkes` models were mistakenly trained with `mask_ties=False`, exposing them to Δt=0 positions during training (which corrupts the Weibull NLL since PDF(0)=0). Models trained under different `mask_ties` settings **cannot be fairly compared**.

---

## Motivation

`eval_nll.py` reports total NLL but conflates two distinct sources of model quality:
- **p(t|H)** — the model's ability to predict *when* the next event occurs
- **p(m|t,H)** — the model's ability to predict *which* event type fires, given that an event occurred at time t

Different TPP kernels encode time information very differently (logits = constant intensity for homo_poisson vs Weibull PDF for weibull). Comparing total NLL alone makes it hard to understand *where* expressive time kernels help. This script enables that comparison, provided models are trained on the same token set (with or without no-event tokens).

## Usage

```bash
python apps/eval_tpp.py --ckpt path/to/ckpt.pt --batch_size 64 --fname eval_tpp
```

## Mathematical Decomposition

The joint log-likelihood factorises exactly as:

```
log p(t, m | H) = log p(t | H)  +  log p(m | t, H)
```

**log p(t | H)** — time density of the next event occurring at Δt:
```
log p(t|H) = log[Σ_v λ_v(Δt)]  +  Σ_v log[surv_v(0, Δt)]
```
where `surv_v(0, Δt) = exp(-∫_0^Δt λ_v(τ) dτ)` is the per-type survival over the interval.

**log p(m | t, H)** — conditional probability of mark given event time:
```
log p(m|t,H) = log λ_m(Δt) - log[Σ_v λ_v(Δt)]
```

**Sanity check**: `nll_time + nll_mark = nll_total`, which should match `eval_nll.py`'s headline NLL for the same checkpoint (real events only scope).

## How It Works

1. Loads checkpoint, sets up `UKBDataset` on val fold with `perturb=False`, `deterministic=True`
2. Instantiates `TPP(loss=model.config.loss, time_unit=model.config.time_unit)`
3. For each batch, calls `model(x0, t0)` to get `outputs`, then:
   - `tpp.intensity(outputs, eval_timesteps=t1, timesteps=t0)` → `(B, L, V)` per-type λ at t1
   - `tpp.survival_prob(outputs, timesteps=t0, start_age=t0, end_age=t1)` → `(B, L, V)` per-type survival
   - `log p(t|H) = log(intensity.sum(-1)) + log(surv).sum(-1)` → `(B, L)`
   - `log p(m|t,H) = tpp.ll_mark_conditional(outputs, t0, t1, x1.unsqueeze(-1))` → `(B, L)`
4. Applies the same masking as `eval_nll.py`: excludes padding (token 0), `ignore_tokens`, and no-event target positions — **real events only**

## Supported Loss Types

| Loss | Notes |
|------|-------|
| `homo_poisson` | Intensity = exp(logits); for this loss `nll_mark` is equivalent to cross-entropy |
| `default` | Same as homo_poisson |
| `weibull` | Intensity = Weibull PDF; survival = exp(-weibull CDF) |

`hawkes` and `hawkes_weibull` are **not yet supported** — they raise `NotImplementedError` inside the `TPP` class.

## Collator: `TPPDecompositionCollator`

Defined in the script itself. Accumulates per-participant and global sums/counts for `time`, `mark`, and `total` components.

## Metric Schema

| Metric | Description |
|--------|-------------|
| `mean_nll_time` | Mean -log p(t\|H) per valid token (global) |
| `mean_nll_mark` | Mean -log p(m\|t,H) per valid token (global) |
| `mean_nll_total` | Mean -(log p(t\|H) + log p(m\|t,H)) per valid token |
| `mean_nll_{comp}_per_participant` | Mean of per-participant means, comp ∈ {time, mark, total} |
| `std_nll_{comp}_per_participant` | Std dev of per-participant means |
| `median_nll_{comp}_per_participant` | Median of per-participant means |
| `n_valid_tokens` | Total valid (real event) tokens evaluated |
| `n_participants` | Number of participants with at least one valid token |

Output is written to `{ckpt_dir}/{fname}.json` (default `fname = "eval_tpp"`).

## Interpretation

- A model with lower `nll_time` better captures *when* events happen
- A model with lower `nll_mark` better captures *which* events happen given the time
- For models trained without no-event tokens (e.g. weibull), reduced `nll_time` is the main expected gain
- `nll_mark` should be roughly equal across models trained on the same token set, since all see the same event type information; large differences may indicate the time kernel interferes with mark prediction
