# Shap Analysis Pipeline (delphi/shap.py)

This module contains infrastructure for running SHAP analysis. It interfaces with the [EXTERNAL] `shap` package and `MultimodalUKBDataset`.

---

## Types

```python
MultimodalOut = tuple[
    np.ndarray,                        # x    — token ids, shape (L,)
    np.ndarray,                        # t    — token timestamps, shape (L,)
    dict[Modality, list[np.ndarray]],  # bio_x_dict — per-modality measurement arrays
    np.ndarray,                        # bio_t — measurement timestamps, shape (M,)
    np.ndarray,                        # bio_m — modality index per measurement, shape (M,)
]
```

`MultimodalOut` is the direct output of `MultimodalUKBDataset.__getitem__` (first 5 elements). It is the native format throughout the SHAP pipeline — no intermediate conversion is needed.

---

## MultimodalShapMasker

Masker for measurement-level SHAP attribution with **missingness background**.

Each SHAP feature corresponds to one **biomarker measurement** (modality × time-point). Tokens are not SHAP features; they are always present in the model input.

### Constructor

```python
MultimodalShapMasker()
```

No arguments. The masker is stateless and can be shared across participants.

### Methods

#### `shape(bio_t: np.ndarray)` / `mask_shapes(bio_t: np.ndarray)`

Both return `(1, n_measurements)` / `[(n_measurements,)]` where `n_measurements = len(bio_t)`.

#### `__call__(mask, bio_t: np.ndarray)`

- **`mask`**: boolean array of shape `(n_measurements,)` from SHAP's coalition sampler. `True` = present, `False` = absent.
- **`bio_t`**: the biomarker timestamp array for the participant (passed by `shap.Explainer`). Used only to standardize the mask shape; the actual data flows through `multimodal_shap_forward` via `partial`.
- **Returns**: `((mask.astype(np.int8),),)` — the binary mask wrapped for the SHAP batching protocol.

Baseline = all measurements absent (complete missingness). SHAP values measure each measurement's contribution relative to having no biomarker data.

---

## multimodal_shap_forward

```python
@torch.no_grad
def multimodal_shap_forward(
    masks: list[np.ndarray],
    *,
    out: MultimodalOut,
    model,
)
```

Forward function for the SHAP explainer. Bake `out` and `model` in with `functools.partial` before passing to `shap.Explainer`.

### Arguments

- **`masks`**: list of binary arrays (one per coalition in SHAP's internal batch), each of length `n_measurements`. `1` = present, `0` = absent. Supplied automatically by `shap.Explainer`.
- **`out`**: `MultimodalOut` for the participant being explained. Bound via `partial`.
- **`model`**: multimodal model with `.forward(idx, age, biomarker, mod_age, mod_idx)`.

### Returns

`np.ndarray` of shape `[batch_size, vocab_size]` — logits at the last position.

### Flow

For each binary mask:
1. Boolean-index `bio_t` and `bio_m` to keep only present measurements.
2. Walk `bio_m` with a per-modality counter to select the matching rows from `bio_x_dict[M]` (measurements in `bio_x_dict[M]` are in the same chronological order as their occurrences in `bio_m`).
3. Accumulate masked samples into a batch.

Then collate the batch and run a single model forward pass.

---

## Complete SHAP Analysis Pipeline

```python
from functools import partial
import numpy as np
import shap
from delphi.shap import MultimodalShapMasker, multimodal_shap_forward
from delphi.multimodal import Modality

# 1. Extract single participant data
x, t, bio_dict, bio_t, bio_m, _, _ = ds[i]
out = (x, t, bio_dict, bio_t, bio_m)

# 2. Build measurement-level feature labels (one per position in bio_m)
meas_features = [Modality(int(mval)).name for mval in bio_m]
meas_timesteps = [float(ts) for ts in bio_t]

# 3. Setup explainer (masker is stateless, reuse across participants)
masker = MultimodalShapMasker()
shap_model = partial(multimodal_shap_forward, out=out, model=model)
explainer = shap.Explainer(
    shap_model,
    masker,
    feature_names=np.array([meas_features]),
    output_names=list(tokenizer.keys()),
)

# 4. Compute SHAP values
# bio_t is passed as the "sample": a uniform 1D array satisfying SHAP's
# dimension consistency requirement. The actual data flows via partial.
shap_values = explainer([bio_t])
# shap_values.values[0]: [n_measurements, vocab_size]
# shap_values.base_values: [1, vocab_size] — baseline = all biomarkers absent
```

### Data Flow Diagram

```
ds[i] → MultimodalOut
             │
             ├─ bio_t ──────────────────────────────────────────────────────┐
             │   (1D array; passed to explainer([bio_t]) as the "sample"    │
             │    so SHAP sees uniform dimensions)                           │
             │                                                               ▼
             │                                           MultimodalShapMasker.__call__(mask, bio_t)
             │                                             uses len(bio_t) to standardise mask
             │                                             returns binary mask (1=present, 0=absent)
             │                                                               │
             └─ out=(x,t,bio_x_dict,bio_t,bio_m) ──── via partial ─────────┘
                                                                             │
                                         multimodal_shap_forward([mask1, mask2, ...], out=out, model=model)
                                           applies each mask directly to out (no intermediate format)
                                           collates masked samples → single batched forward pass
                                                                             │
                                                                             ▼
                                                            logits → SHAP values [n_measurements, vocab]
```

---

## ShapMasker

Masker for discrete token sequences (non-multimodal models). Unchanged.

### Token IDs
- `0`: padding
- `1`: no_event
- `2`: female (sex token)
- `3`: male (sex token)
- `>3`: event tokens (diseases, procedures, lifestyle, etc.)

### Masking Strategies

| Token Type | Strategy | Rationale |
|------------|----------|-----------|
| Event (non-last) | **Drop**: set `x=0, t=-1e4`, sort pushes to front as padding | Removing events from history |
| Event (last) | **Replace** with `no_event` token | Preserve elapsed time semantics for prediction |
| Sex token | **Swap** as counterfactual (`male↔female`) | Sex is always present; measure effect of alternative |

---

## shap_pickle

Serialized SHAP analysis results for a cohort of participants.

### File Output
- `shap_missingness.pickle.gz`: gzip-compressed dictionary of per-participant SHAP results (overridable with `--fname`)

### Schema

```python
shap_pickle: dict[int | str, ...]
# Key: participant ID (int)  →  per-participant results
# Key: "tokenizer"          →  the dataset tokenizer (name → int mapping)
```

Per-participant value:
```python
{
    "shap":      np.ndarray,  # shape [n_measurements, vocab_size], dtype float16
    "features":  list[str],   # length n_measurements — e.g. ["WBC", "PRS", "WBC"]
    "timesteps": np.ndarray,  # shape [n_measurements], dtype float16 — days
}
```

All arrays are position-aligned. Features are at **measurement granularity** (one per modality × time-point).

### Field Descriptions

| Field | Type | Description |
| --- | --- | --- |
| `shap` | `np.ndarray [n_measurements, vocab_size]` | SHAP attribution in logit-space. `shap[i, j]` = contribution of measurement `i` to predicting token `j`. Baseline = all biomarkers absent. |
| `features` | `list[str]` | Modality names per measurement, e.g. `["WBC", "PRS", "WBC"]`. |
| `timesteps` | `np.ndarray [n_measurements]` | Timestamp (in days) for each measurement. |

### Notes

- `n_measurements` varies per participant
- SHAP values are in **logit-space** (attributions sum to difference in logits from baseline)
- Baseline = complete biomarker missingness (all measurements absent)
- Only biomarker measurements are SHAP features; discrete tokens are always present

### Example Usage

```python
import gzip, pickle
import numpy as np

with gzip.open("shap_missingness.pickle.gz", "rb") as f:
    shap_pickle = pickle.load(f)

tokenizer = shap_pickle["tokenizer"]

pid = 123456
result = shap_pickle[pid]
shap_values = result["shap"]    # [n_measurements, vocab_size], float16
features    = result["features"] # ["WBC", "PRS", ...]
timestamps  = result["timesteps"]

# Top biomarker measurements contributing to a specific disease
outcome_idx = tokenizer["heart_failure"]
contributions = shap_values[:, outcome_idx].astype(float)
top_idx = np.argsort(np.abs(contributions))[::-1][:10]
for idx in top_idx:
    print(f"{features[idx]} (day {timestamps[idx]:.0f}): {contributions[idx]:.4f}")

# All WBC measurements across time
wbc_mask = np.array([f == "WBC" for f in features])
wbc_times = timestamps[wbc_mask]
wbc_contrib = shap_values[wbc_mask, outcome_idx].astype(float)
plt.scatter(wbc_times, wbc_contrib)
plt.xlabel("Days"); plt.ylabel("SHAP value for heart_failure")
```

### Creation Script

`apps/run_shap_m4.py` — runs the full pipeline for a checkpoint and saves results.

---

# SHAP Visualization Module Summary

## Purpose

A Python module for visualizing SHAP (SHapley Additive exPlanations) values from a disease prediction model. The module answers three core questions:

1. **Feature → Diseases**: Given a feature, which diseases does it contribute to most?
2. **Disease → Features**: Given a disease, which features are most predictive of it?
3. **Temporal dynamics**: For a (feature, disease) pair, how does the contribution change over time?

## Architecture

### Computation vs. Visualization Separation

Each task has a **compute function** that aggregates raw data into a DataFrame, and a **plot function** that visualizes it. This allows users to access the underlying data without plotting.

| Task | Compute Function | Plot Function |
|------|------------------|---------------|
| Feature → Diseases | `compute_feature_disease_contributions()` | `plot_feature_to_diseases()` |
| Disease → Features | `compute_disease_feature_importance()` | `plot_disease_predictive_features()` |
| Temporal | `compute_feature_disease_temporal()` | `plot_feature_disease_temporal()` |

### Utility Functions

- `load_shap_data(filepath)`: Loads gzip-compressed pickle file
- `compute_feature_counts(shap_data)`: Counts occurrences of each feature across all participants
- `suggest_min_samples_threshold(shap_data)`: Suggests reasonable `min_samples` thresholds based on percentile distribution
- `get_data_summary(shap_data)`: Returns summary statistics including feature counts and threshold suggestions

## Key Design Decisions

### 1. Filtering by `min_samples`

High-variance features with few occurrences can be noisy. The `min_samples` parameter filters these out.

- **Applied to**: `compute_disease_feature_importance()` and `plot_disease_predictive_features()`
- **Not applied to**: Single-feature functions (`compute_feature_disease_contributions`, `compute_feature_disease_temporal`) — user has full freedom to visualize any feature

### 2. Filtering by `contribution_direction`

Users can filter to see only risk-increasing or risk-decreasing contributions.

- **Parameter**: `contribution_direction: Literal['all', 'positive', 'negative']`
- **Filter logic**: Based on `median_shap` (more robust than mean)
- **Applied to**: `plot_feature_to_diseases()` and `plot_disease_predictive_features()`
- **Visual behavior**: When filtering by direction, all bars use a single color (red for positive, blue for negative)

### 3. SHAP Value Interpretation

- SHAP values are in **logit-space**
- `exp(shap)` gives odds ratios (risk multipliers)
- Positive SHAP → increases disease risk (red in plots)
- Negative SHAP → decreases disease risk (blue in plots)

## Known Issues & Fixes

### float16 Incompatibility with Pandas

The source data stores arrays as `float16`/`float32` for memory efficiency. Pandas does not support `float16` for indexing operations like `pd.cut()`.

**Fix applied in two places:**

1. `compute_feature_disease_temporal()`: Convert values to Python `float()` when building records
2. `_compute_binned_statistics()`: Cast x-values to `np.float64` before calling `pd.cut()`

## Function Signatures Reference

```python
# Task 1: Feature → Diseases
plot_feature_to_diseases(
    shap_data: dict,
    feature_name: str,
    disease_names: Optional[Dict[int, str]] = None,
    top_k: int = 15,
    contribution_direction: Literal['all', 'positive', 'negative'] = 'all',
    figsize: Tuple[int, int] = (14, 6)
) -> pd.DataFrame

# Task 2: Disease → Features
plot_disease_predictive_features(
    shap_data: dict,
    disease_idx: int,
    disease_name: Optional[str] = None,
    top_k: int = 20,
    min_samples: int = 10,
    contribution_direction: Literal['all', 'positive', 'negative'] = 'all',
    figsize: Tuple[int, int] = (14, 8)
) -> pd.DataFrame

# Task 3: Temporal Dynamics
plot_feature_disease_temporal(
    shap_data: dict,
    feature_name: str,
    disease_idx: int,
    disease_name: Optional[str] = None,
    time_unit: str = 'years',
    n_bins: int = 10,
    figsize: Tuple[int, int] = (14, 5),
    xlim: Optional[Tuple[float, float]] = None,
    sample_size: Optional[int] = 2000
) -> pd.DataFrame
```
