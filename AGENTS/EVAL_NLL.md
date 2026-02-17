# Eval NLL Task

## Overview

`apps/eval_nll.py` computes negative log-likelihood on the validation dataset for unimodal (Delphi2M) models. It writes metrics to a `.json` file in the same directory as the model checkpoint.

For general eval script patterns (checkpoint loading, dataset setup, argument parsing), see [EVAL.md](EVAL.md).

## Usage

```bash
python apps/eval_nll.py --ckpt path/to/ckpt.pt --batch_size 64 --fname eval_nll
```

## How It Works

1. Loads checkpoint via `load_ckpt`, prepares `UKBDataset` on val fold with `perturb=False`, `deterministic=True`, keeping training `block_size` (currently `None` for all models, meaning full sequences are used)
2. For each batch, calls `model.forward(x0, t0)` (without targets) to get `outputs`, then `model.loss(outputs, x1, t0, t1)` to get per-position `(B, L)` NLL tensors
3. Applies the same masking as training: excludes padding (token 0) and `ignore_tokens` (tokens 2-12). See [MODEL.md](MODEL.md) for loss function details and special token semantics.
4. Passes loss tensors and masks to collators for aggregation

### Collators

The script uses two collator classes (defined in the script itself, see [EVAL.md](EVAL.md) for the collator pattern):

- **`NLLCollator(suffix)`** — one instance per masking scope. Accumulates global and per-participant NLL (total + per-component). `finalize()` returns `mean_nll{suffix}`, per-participant stats, and component breakdowns.
- **`PerTokenNLLCollator(idx_to_event)`** — accumulates per-token-type NLL sums/counts. `finalize()` returns the `per_token_nll` dict.

## Metric Schema

### Masking Scopes

The script reports metrics under three scopes, depending on whether the model was trained with no-event tokens (see [DATA.md](DATA.md) for no-event token semantics):

| Suffix | Positions included | When reported |
|--------|-------------------|---------------|
| *(none)* | Real events only (no-event targets excluded) | Always — these are the headline metrics |
| `_no_event` | No-event target positions only | Only if model trained with no-event tokens |
| `_all` | All valid positions (real + no-event) | Only if model trained with no-event tokens |

### Metrics Per Scope

| Metric | Description |
|--------|-------------|
| `mean_nll{suffix}` | Mean NLL per valid token (global) |
| `mean_nll_per_participant{suffix}` | Mean of per-participant mean NLLs |
| `std_nll_per_participant{suffix}` | Std dev of per-participant mean NLLs |
| `median_nll_per_participant{suffix}` | Median of per-participant mean NLLs |
| `n_valid_tokens{suffix}` | Total valid tokens evaluated |

### Loss Component Breakdown

If the model's loss has multiple components (e.g., `default` loss → `loss_ce` + `loss_dt`), each component is reported separately per scope:

| Metric | Description |
|--------|-------------|
| `mean_nll_{comp}{suffix}` | Mean of component (e.g., `mean_nll_ce`, `mean_nll_dt`) |
| `mean_nll_{comp}_per_participant{suffix}` | Per-participant mean of component |

### Other Metrics

| Metric | Description |
|--------|-------------|
| `n_participants` | Number of participants with at least one valid token |
| `per_token_nll` | Dict mapping token name → mean NLL (real events only) |

## Known Caveat: Excluded Participants

Approximately 5% of validation participants (~5188 out of 100426) have very short sequences (1-4 tokens) where **all targets are padding or ignored tokens**. These participants contribute zero valid positions and are excluded from all per-participant statistics.

This was verified with `apps/diagnose_no_event_participants.py`:
- These participants have zero valid targets regardless of no-event token settings
- Their sequences are too short to contain any real medical events after the input→target shift
- `n_participants` in the output reflects only participants with at least one valid token, not the full validation set size

This exclusion is inherent — NLL cannot be computed for participants with no valid prediction targets. It does not bias `mean_nll` (token-level), but `mean_nll_per_participant` is conditioned on participants with evaluable positions.

## Comparing NLL Across Loss Types

When comparing NLL between models with and without no-event tokens (e.g., `hawkes` with no-event vs `hawkes_weibull` without), excluding no-event target positions is necessary but **not sufficient** for a fair comparison. See [DESIGN.md](DESIGN.md) § "No-Event Tokens: Input-Side Effects on NLL" for a detailed discussion of why no-event tokens still confer an advantage through the input side.
