# Eval AUC Task

## Overview

`apps/auc_fast.py` evaluates the model's ability to **rank participants by disease risk**. For each disease, it asks: does the model assign higher predicted intensity to participants who actually develop the disease than to those who don't?

This is measured via **AUC** (area under the ROC curve), computed using the Mann-Whitney U statistic. AUC = 1.0 means perfect ranking; AUC = 0.5 means random.

For general eval script patterns, see [EVAL.md](EVAL.md).

## What This Task Measures (and What It Doesn't)

This task evaluates **risk ranking**, not **prognostic accuracy**. The distinction matters:

- **Risk ranking**: "Does the model correctly identify who is at higher risk?" — a relative comparison between participants
- **Prognostic accuracy**: "Does the model correctly predict the probability that a specific patient will develop disease X within Y years?" — an absolute prediction

The current task only addresses the former. Prognostic evaluation would require either:
1. Sampling trajectories from the model and checking disease occurrence rates, or
2. Integrating the predicted intensity over a time horizon to obtain calibrated risk probabilities

## How It Works

### 1. Forward Pass

Runs each validation participant through the model to get per-position log-intensities (logits) of shape `(B, L, V)`.

### 2. Time Offset Correction

For each target position (where a disease occurs), the score is taken not from the immediately preceding position but from a position at least `min_time_gap` years earlier (`correct_time_offset`). This ensures the model's prediction is based on information available *before* the event.

The default `min_time_gap` is 0.01 years (~3.65 days) — intentionally small, because the goal is risk ranking rather than long-horizon forecasting. A small gap maximizes the number of evaluable positions while still ensuring the model isn't scoring events using information from essentially the same time point.

### 3. Score Collection

Two types of scores are collected per participant:

- **Control scores** (`AgeStratRatesCollator`): For each age bin, one position is randomly sampled from the participant's sequence and the model's log-intensity for each disease at that position is recorded. This gives one score per participant per age bin per disease. The random sampling avoids within-participant correlation.

- **Disease scores** (`DiseaseRatesCollator`): For each participant who develops disease `k`, the model's log-intensity for `k` at the (time-corrected) position where the disease first appears is recorded.

### 4. AUC Computation

For each disease × age bin × sex combination:
- **Cases**: participants who developed the disease in that age bin
- **Controls**: disease-free participants with a score in that age bin
- **AUC**: Mann-Whitney U statistic comparing case scores vs control scores

The overall AUC for a disease is the **mean across age bins** (within each sex stratum).

### 5. Stratification

Results are stratified by:
- **Sex**: female, male, either
- **Age**: 5-year bins (default: 40-45, 45-50, ..., 80-85)

Stratification is important because some risk factors are trivially predictive — older age and male sex are strong predictors for many diseases (e.g., cardiovascular). Without stratification, a model could achieve high AUC simply by learning age and sex effects. Stratifying forces the model to demonstrate discrimination *within* age-sex groups.

## Output

A JSON file written alongside the checkpoint with structure:

```json
{
    "E11": {
        "female": {
            "40-45": {"auc": 0.72, "ctl_count": 1000, "dis_count": 50},
            "...": "...",
            "total": {"auc": 0.71, "ctl_count": 8000, "dis_count": 200}
        },
        "male": { "..." },
        "either": { "..." }
    },
    "...": "..."
}
```

## Usage

```bash
python apps/auc_fast.py --ckpt path/to/ckpt.pt --batch_size 64 --min_time_gap 0.01
```

Key arguments:
- `--min_time_gap`: Minimum lead time in years between the prediction point and the event (default: 0.01)
- `--age_start`, `--age_end`, `--age_gap`: Age stratification (default: 40-85 in 5-year bins)

## Limitations

### Only works for intensity-based losses

The script uses raw log-intensities (`logits`) as the discrimination score. This means it only works for models with intensity-based losses (`default`, `homo_poisson`, `homo_cluster_poisson`) that produce logits. Models with parametric losses (`hawkes`, `hawkes_weibull`, `weibull`) output distribution parameters instead of logits, and this script does not handle them. Extending it would require computing the instantaneous intensity λ from the model's parametric outputs.

### Binned age stratification introduces variance

The mean-across-bins approach for overall AUC can be noisy, especially for rare diseases with few cases per bin. A single bin with very few cases can produce an unreliable AUC estimate that disproportionately affects the average.

### No principled confounder adjustment

Age is handled by coarse 5-year bins, and sex by simple stratification. A more principled approach might use the **concordance index (C-statistic)** from survival analysis, which can incorporate continuous covariates. However, adapting the C-statistic to TPP models (where predictions are intensities rather than survival times or hazard ratios) is non-trivial and remains an open question.

### Instantaneous intensity vs integrated risk

Using the instantaneous intensity at a single time point as the score ignores the temporal dimension — it doesn't account for how long a participant is at elevated risk. Integrated risk over a time horizon (as in `integrate_risk` in `delphi/eval.py`) would be more appropriate for prognostic evaluation, but is not used here because the goal is ranking, not prognosis.

## Related Code

- `delphi/eval.py`: Contains all collator classes, `correct_time_offset`, `mann_whitney_auc`, and related utilities
- `apps/auc.py`: Older version of the AUC task (similar logic, less optimized)
