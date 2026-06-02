# Eval C-Index Task

## Overview

The **concordance index** (C-index, C-statistic) is an alternative to a binned AUC pipeline for evaluating disease risk ranking. It answers the same question — does the model assign higher predicted intensity to participants who develop disease X than to those who don't? — but with lower variance.

## Why C-Index Instead of Binned AUC

The current AUC pipeline (`EVAL_AUC.md`) has two independent sources of variance that the C-index avoids.

### 1. Equal-weighted averaging of unequal-reliability bins

The current pipeline computes AUC per age bin and takes an unweighted mean. The variance of a Mann-Whitney AUC with $m$ cases and $n$ controls is roughly $O(1/(mn))$, so:

$$\text{Var}(\bar{\text{AUC}}) = \frac{1}{k^2} \sum_{i=1}^k \text{Var}(\text{AUC}_i) \approx \frac{1}{k^2} \sum_{i=1}^k \frac{1}{m_i n_i}$$

A bin with 3 cases is given the same weight as a bin with 40 cases, even though its AUC estimate is roughly 13× noisier. For rare diseases, most bins will be sparse.

The C-index pools all comparable pairs with **risk-set weighting** — each event contributes proportional to the number of controls available at that moment:

$$C = \frac{\text{# concordant pairs}}{\text{# total comparable pairs}}$$

This is equivalent to a risk-set-weighted average of per-event AUC values, where sparse periods automatically contribute less. The effective variance is $O(1/(M \cdot N_\text{eff}))$ where $M$ and $N_\text{eff}$ grow with the full dataset.

### 2. Random sampling of control positions

The current pipeline randomly samples **one** control score per participant per age bin to avoid within-participant correlation. This discards information and adds run-to-run stochasticity — different random seeds produce different AUC estimates on the same model.

In the C-index framework, at each event time $t_k$, **every at-risk participant** serves as a control. No sampling is required. Each participant contributes to as many risk sets as there are events occurring after their observation begins, making much fuller use of the data.

### 3. Continuous age adjustment

Age stratification via 5-year bins is a coarse approximation. The C-index's risk-set construction provides continuous age adjustment for free: when participant $i$ has an event at age 63.2, the controls are all participants observable at that exact time, implicitly age-matched at far finer resolution than any binning scheme.

### Summary

| Issue | Current AUC | C-index |
|-------|-------------|---------|
| Sparse bins | Equal weight to noisy bins | Risk-set weighting — sparse periods contribute less |
| Control sampling | One random sample per bin | All at-risk participants, no sampling |
| Age adjustment | Coarse 5-year bins | Continuous, via risk sets |
| Run-to-run stochasticity | Yes (random sampling) | No |

**Core principle:** averaging AUCs across bins is statistically less efficient than a single pooled C-index estimate, because the former treats each sparse bin as an independent estimator and then averages, while the latter accumulates all comparable pairs into one estimator with $O(M \cdot N_\text{total})$ effective sample size.

## Connection to Time-Dependent AUC

The C-index can be derived from the time-dependent AUC framework. At each event time $t_k$ (continuous time, so exactly one case per event), the per-event AUC is the percentile rank of the case among its controls:

$$\text{AUC}(t_k) = \frac{1}{n_k} \sum_{j \in \mathcal{R}(t_k)} \mathbf{1}\!\left(\lambda_\text{case}(t_k) > \lambda_j(t_k)\right)$$

The C-index is the risk-set-weighted average of these per-event AUCs:

$$C = \frac{\sum_k n_k \cdot \text{AUC}(t_k)}{\sum_k n_k} = \frac{\text{concordant pairs}}{\text{total pairs}}$$

This is algebraically identical to the standard pairwise concordance formula, so the C-index is not a different metric — it is a more statistically efficient way to aggregate the same local discrimination signal.

## Adaptation to the Delphi TPP Setting

The standard survival C-index uses a static risk score (e.g., a hazard ratio). In Delphi, the model produces time-varying log-intensities. The natural adaptation is:

- At each event time $t_k$, use the model's log-intensity **at $t_k$** (with the same `min_time_gap` offset as the current AUC pipeline) as the discrimination score.
- The risk set $\mathcal{R}(t_k)$ consists of all participants observable at $t_k$ who have not yet had the disease — the continuous-time analogue of the current "disease-free controls in this age bin".

Sex stratification can be preserved by computing separate C-indices per sex (as is done for AUC), or by including sex as a covariate in a more principled adjustment.

## Implementation (`apps/c-index.py`)

### Two-Pass Algorithm

**Phase 1 — collect case scores and onset times**

One forward pass over the validation set:

- `DiseaseRatesCollator` (from `delphi/eval.py`): records, for each case participant, the model's log-intensity at the `min_time_gap`-corrected time point. Produces `dis_rates (N, V)` (case scores) and `dis_times (N, V)` (the exact t0 timestamps used to score each case).
- `EventTimeCollator` (from `delphi/eval.py`): records the raw (un-corrected) onset time from `(x1, t1)` for each disease. `finalize()` returns `(occur_time (N, V), exit_time (N,))`; only `occur_time` is used as `onset_times`, with NaN for participants who never develop the disease.
- `SexCollator` (from `delphi/eval.py`): records `is_female (N,)`.

After Phase 1, if `--after_modality` is enabled, case events scored before the biomarker cutoff are masked out (set to NaN in `dis_rates`). Remaining case events are then flattened to `(E_total,)` vectors.

**Phase 2 — accumulate concordant / total pairs**

Second forward pass without `correct_time_offset` (raw `t0` and `logits` needed for the searchsorted lookup). For each batch of `B` participants and each chunk of `E_c` events:

1. `torch.searchsorted(t0, t_q.expand(B, -1))` locates, for each participant, the t0 position corresponding to the case's query time — one batched CUDA kernel replaces the per-participant Python loop.
2. A flat fancy-index into `logits` reads the per-participant, per-disease scores without materialising a large `(B, E_c, V)` intermediate.
3. Validity masks are applied in order:
   - **Timeline validity**: index within bounds and not padding (`t_at > 0`)
   - **Max gap**: control score time must be within `max_gap` years of the case query time, filtering out age-mismatched pairs where searchsorted silently fell back to the control's last observation
   - **Biomarker cutoff** (if `--after_modality`): control score time must be after the control's first biomarker measurement
   - **At-risk**: participant has not yet developed disease `d` at the case's event time
   - **Anti-self**: participant is not the case itself
4. `concordant[e]` and `total_pairs[e]` are incremented on CPU via `.sum(0).cpu()`.

Event arrays and `onset_times` are moved to GPU once before Phase 2. Events are chunked (`chunk_size=8192`) to keep peak GPU memory bounded at ~20 MB per chunk, regardless of sequence length or vocabulary size.

### Key Collators Used

| Collator | Source | Output |
|----------|--------|--------|
| `DiseaseRatesCollator` | `delphi/eval.py` | `dis_rates (N, V)`, `dis_times (N, V)` |
| `EventTimeCollator` | `delphi/eval.py` | `occur_time (N, V)` (used as `onset_times`), `exit_time (N,)` (discarded) |
| `SexCollator` | `delphi/eval.py` | `is_female (N,)` |

### Query Time vs. Onset Time

Two different times are tracked per case event:

- **`event_query_times`** = `dis_times[p, d]` — the offset-corrected t0 position at which the model was actually queried for the case. Used for `searchsorted` so controls are scored at exactly the same point in time.
- **`event_actual_times`** = `onset_times[p, d]` — the raw onset time from the data. Used for the at-risk check: a control is eligible only if their own onset of disease `d` is after this time (or they never develop the disease).

### Max Gap Filter

The `--max_gap` flag (default 5 years) addresses a subtle issue with the searchsorted time-matching. When a control's timeline doesn't extend to the case's query time, searchsorted silently falls back to the control's last observation — which could be years earlier. The validity check `(idx_mat >= 0) & (t_at > 0)` does not catch this. The max gap filter adds `(t_q - t_at) < max_gap_days`, excluding pairs where the control's score comes from a time too far before the case's query time.

### After-Modality Filter

When `--after_modality` is used with `--modalities`, both case and control scores are restricted to time points after the first biomarker measurement:

- **Case side**: case events where `dis_times < bio_cutoff` are masked to NaN before flattening, so they are excluded entirely.
- **Control side**: in Phase 2, pairs where `t_at < bio_cutoff[j]` are marked invalid.

When multiple modalities are specified, the cutoff is the **earliest** first-occurrence time across all modalities (i.e. `np.fmin`), so evaluation begins as soon as any modality becomes available.

This is useful when comparing models trained with vs without a particular biomarker modality: it ensures scores are only compared at time points where the biomarker model actually has the biomarker data available. Without this filter, pre-biomarker time points (where both models have identical information) dilute the comparison.

## Output

A JSON file written alongside the checkpoint:

```json
{
    "E11": {
        "female": {"c_index": 0.71, "n_events": 200, "n_pairs": 800000},
        "male":   {"c_index": 0.68, "n_events": 300, "n_pairs": 1200000},
        "either": {"c_index": 0.70, "n_events": 500, "n_pairs": 2000000}
    },
    "...": "..."
}
```

- `c_index`: proportion of concordant pairs; `null` if no valid pairs exist
- `n_events`: number of case events for that disease × sex group
- `n_pairs`: total valid (case, control) pairs counted

Output filename: `{ckpt_dir}/cindex.json` (auto-generated, with `-modalities_*` suffix when `--modalities` is used; override with `--fname`)

## Usage

```bash
# Basic usage
python apps/c-index.py --ckpt path/to/ckpt.pt --batch_size 64

# Compare models on NMR subset, restricted to post-biomarker time points
python apps/c-index.py --ckpt path/to/ckpt.pt --modalities nmr --after_modality

# Custom max gap and lead time
python apps/c-index.py --ckpt path/to/ckpt.pt --max_gap 3 --min_time_gap 0.01
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--ckpt` | `delphi-m4/delphi-m4/ckpt.pt` | Path relative to `$DELPHI_CKPT_DIR` |
| `--batch_size` | 64 | Inference batch size |
| `--min_time_gap` | 0 | Lead time in years between prediction point and event |
| `--max_gap` | 5 | Maximum allowed gap in years between case query time and control score time |
| `--modalities` | None | Restrict to participants with all listed biomarker modalities |
| `--after_modality` | False | Only compute C-index at time points after first occurrence of specified modalities |
| `--fname` | auto | Override output filename stem |

## Sanity Checks

After the output is written, verify:

1. All C-index values in `[0, 1]`; values near 0.5 for weak/rare diseases are expected.
2. `n_pairs` should be large (millions) for common diseases and smaller for rare ones.
3. C-index and AUC should correlate across diseases — same discrimination signal, different aggregation — typically within ±0.05.
4. Re-running with the same checkpoint produces identical results (no randomness).

## Empirical Variance

Across 5 models trained with different random seeds on the same data, the C-index shows **lower variance** than the AUC for the same diseases. This is consistent with the theoretical expectation: the C-index pools all comparable pairs into a single estimator while the AUC averages across sparse age bins with random control sampling. See `notebook/eval_variance.py` for the analysis.

## Related Code

- `delphi/eval.py`: `DiseaseRatesCollator`, `EventTimeCollator`, `SexCollator`, `correct_time_offset`
- `delphi/experiment.py`: `eval_iter`, `load_ckpt`, `move_batch_to_device`
- `delphi/data/ukb.py`: `Biomarker.first_occurrence_times()` — used for `--after_modality` cutoff
- `delphi/multimodal.py`: `Modality` enum
