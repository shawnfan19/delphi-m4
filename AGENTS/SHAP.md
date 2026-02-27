# Shap Analaysis Pipeline (delphi/shap.py)
this module contains infrastructure for running SHAP analysis and interfaces with the [EXTERNAL] shap package and MultimodalUKBDataset

## ShapArray
ShapArray is an intermediate data structure for interfacing with shap.Explainer. shap.Explainer returns a shap.Explanation object which expects its features to have consistent dimensions. MultimodalUKBDataset outputs a tuple of different types (torch.Tensor and dict[str, torch.Tensor) and incompatible dimensions.

https://github.com/shap/shap/blob/984932ee48f9646eae77cb2801aeb815e0d865a5/shap/_explanation.py#L855
shap AssertionError: Arrays in Explanation objects must have consistent inner dimensions!

### Why it exists
`shap.Explainer` returns `shap.Explanation` objects that require features with consistent dimensions. `MultimodalUKBDataset.__getitem__` returns heterogeneous types (`np.ndarray` and `dict[Modality, list[np.ndarray]]`) with incompatible shapes. `ShapArray` flattens this into a uniform representation.

### Type Definition
```python
ShapArray = tuple[np.ndarray, np.ndarray, np.ndarray]
#                 all_x,      all_t,      all_m
```
All three arrays are 1D and position-aligned:

-   `all_x`: Feature values (tokens cast to float, followed by biomarker values)
-   `all_t`: Timestamps in days for each feature
-   `all_m`: Modality index for each feature (1 = discrete token, 2+ = biomarker modality per `Modality` enum)

A `ShapArray` holds data for a **single participant**. SHAP analysis runs multiple forward passes with different features masked; `shap.Explainer` handles batching internally.

### Example

```python
# Input (MultimodalOut from __getitem__, excluding x1/t1):
x = [2, 101, 1269]           # 3 tokens
t = [0, 2000, 3000]          # timestamps
bio_x_dict = {
    Modality.PRS: [[1.0, -1.0, 0.5]],   # 1 measurement, 3 features
    Modality.WBC: [[0.0, 2.0, 0.5]],    # 1 measurement, 3 features
}
bio_t = [0, 2500]            # PRS at day 0, WBC at day 2500
bio_m = [2, 3]               # Modality indices

# Output (ShapArray):
all_x = [2, 101, 1269, 1.0, -1.0, 0.5, 0.0, 2.0, 0.5]
all_t = [0, 2000, 3000, 0, 0, 0, 2500, 2500, 2500]
all_m = [1, 1, 1, 2, 2, 2, 3, 3, 3]
```

---

## to\_shap\_array(s, detokenizer, biomarker\_features, biomarker\_background=None)

Converts per-participant data from `MultimodalUKBDataset.__getitem__` format to `ShapArray`.

### Arguments

-   `s`: `MultimodalOut` tuple `(x, t, bio_x_dict, bio_t, bio_m)` — first 5 elements from `dataset[i]`
-   `detokenizer`: `dict[int, str]` mapping token IDs to names (from `dataset.detokenizer`)
-   `biomarker_features`: `dict[Modality, list[str]]` mapping modality to list of feature names
-   `biomarker_background`: `dict[Modality, np.ndarray] | None` — optional background values (training cohort means). Pass `None` (default) when using `MultimodalShapMasker` (missingness background).


### Returns

Tuple of `(shap_array, features, bio_bg)`:
-   `shap_array`: `ShapArray` tuple `(all_x, all_t, all_m)`
-   `features`: `list[str]` of feature names, position-aligned with `shap_array` (scalar-level)
    -   Token features: `["diabetes", "stroke", ...]`
    -   Biomarker features: `["PRS.feature1", "PRS.feature2", ..., "WBC.feature1", ...]`
-   `bio_bg`: `np.ndarray` of background values (empty array when `biomarker_background=None`)


### Ordering

Biomarkers are appended in **first-appearance order** based on `bio_m` (the order modalities appear chronologically for this participant), then chronologically within each modality. This ordering is deterministic and invertible via `from_shap_array`.

---

## from\_shap\_array(s, biomarker\_features=None)

Reconstructs `MultimodalOut` from `ShapArray`. Inverse of `to_shap_array`.

### Arguments

-   `s`: `ShapArray` tuple `(all_x, all_t, all_m)`
-   `biomarker_features`: `dict[Modality, list[str]]` (optional) — if provided, enables correct reconstruction when multiple measurements of the same modality occur at the same timestamp


### Returns

`MultimodalOut` tuple `(x, t, bio_x_dict, bio_t, bio_m)`


## ShapMasker

Masker for discrete token sequences (non-multimodal models).

### Token IDs
- `0`: padding
- `1`: no_event
- `2`: female (sex token)
- `3`: male (sex token)
- `>3`: event tokens (diseases, procedures, lifestyle, etc.)

### Masking Strategies

When a feature is masked (`mask[i] = False`):

| Token Type | Strategy | Rationale |
|------------|----------|-----------|
| Event (non-last) | **Drop**: set `x=0, t=-1e4`, sort pushes to front as padding | Removing events from history |
| Event (last) | **Replace** with `no_event` token | Preserve elapsed time semantics for prediction |
| Sex token | **Swap** as counterfactual (`male↔female`) | Sex is always present; measure effect of alternative |

### Implementation Detail
After masking, tokens are sorted by timestamp. Dropped tokens (with `t=-1e4`) sort to the front and appear as padding to the model.

---

## MultimodalShapMasker

Masker for measurement-level SHAP attribution with **missingness background**.

Each SHAP feature corresponds to one **biomarker measurement** (modality × time-point). A masked measurement is fully absent from the model input (modality index set to 0, causing `from_shap_array` to skip it). Tokens are always passed through unchanged.

### Constructor
```python
MultimodalShapMasker(biomarker_features: dict[Modality, list])
```

-   `biomarker_features`: mapping from modality to its list of scalar feature names. Used to determine the size (in scalars) of each measurement.

### Methods

#### `_measurement_sizes(s: ShapArray) -> list[int]`

Returns `[feature_dim, feature_dim, ...]` — one entry per biomarker measurement in the flat bio-array, where `feature_dim = len(biomarker_features[modality])`. Used to expand the measurement-level mask to scalar-level.

#### `shape(s)` / `mask_shapes(s)`

Both return `(1, n_measurements)` where `n_measurements` is the total number of biomarker measurements for this participant (summed across all modalities).

### Masking Strategy

| Feature Type | Strategy |
| --- | --- |
| Tokens | Always passed through unchanged |
| Biomarker measurement | **Absent**: modality index set to 0 → `from_shap_array` skips it entirely |

Baseline = all measurements absent (complete missingness). SHAP values measure each measurement's contribution relative to having no biomarker data at all.

### `__call__(mask, s)`

#### Arguments

-   `mask`: Boolean array of shape `(n_measurements,)` where `True` = measurement present, `False` = measurement absent
-   `s`: `ShapArray` tuple `(all_x, all_t, all_m)`

#### Returns

Nested tuple `((all_x,), (all_t,), (all_m,))` — SHAP API requirement for masker outputs.

---

## multimodal\_shap\_forward(all\_x\_lst, all\_t\_lst, all\_m\_lst, model)

Forward function for SHAP explainer. Converts masked `ShapArray` data back to model input format and runs inference.

### Arguments

-   `all_x_lst`: List of `all_x` arrays (one per sample in SHAP's internal batch)
-   `all_t_lst`: List of `all_t` arrays
-   `all_m_lst`: List of `all_m` arrays
-   `model`: The multimodal model with `.forward(idx, age, biomarker, mod_age, mod_idx)` signature


### Returns

`np.ndarray` of shape `[batch_size, vocab_size]` — logits for the next token prediction (last position only).

### Flow

1.  Convert each `ShapArray` back to `MultimodalOut` via `from_shap_array`
2.  Collate into batched tensors (same format as `MultimodalUKBDataset.get_batch`)
3.  Run model forward pass
4.  Return logits for last position


---

## Complete SHAP Analysis Pipeline

```python
# 1. Extract single participant data
x, t, bio_dict, bio_t, bio_m, _, _ = ds[i]

# 2. Convert to SHAP-compatible flat format (no background needed)
sample, _, _ = to_shap_array(
    (x, t, bio_dict, bio_t, bio_m),
    detokenizer=ds.detokenizer,
    biomarker_features=biomarker_features,   # dict[Modality, list[str]]
)
all_x, all_t, all_m = sample

# 3. Build measurement-level feature labels
masker = MultimodalShapMasker(biomarker_features=biomarker_features)
sizes = masker._measurement_sizes(sample)    # [feat_dim, feat_dim, ...]
bio_m_flat = all_m[all_m != 1]
bio_t_flat = all_t[all_m != 1]
meas_features, meas_timesteps = [], []
offset = 0
for size in sizes:
    modval = int(bio_m_flat[offset])
    t_meas = float(bio_t_flat[offset])
    meas_features.append(f"{Modality(modval).name}@{t_meas:.0f}")
    meas_timesteps.append(t_meas)
    offset += size

# 4. Setup explainer
shap_model = partial(multimodal_shap_forward, biomarker_features=biomarker_features, model=model)
explainer = shap.Explainer(
    shap_model,
    masker,
    feature_names=np.array([meas_features]),   # measurement-level names
    output_names=list(tokenizer.keys()),
)

# 5. Compute SHAP values
shap_values = explainer([sample])
# shap_values.values[0]: [n_measurements, vocab_size]
# shap_values.base_values: [1, vocab_size] — baseline = all biomarkers absent
```

### Data Flow Diagram

```text
ds[i] → MultimodalOut → to_shap_array() → ShapArray (scalar-level flat arrays)
                                              ↓
                         masker._measurement_sizes() → measurement-level feature labels
                                              ↓
                         MultimodalShapMasker.__call__(mask, ShapArray)
                           (expands measurement mask → scalar mask via repeat)
                                              ↓
                                        masked ShapArray
                                              ↓
                         multimodal_shap_forward() → from_shap_array()
                           (skips measurements where modval == 0)
                                              ↓
                                      model.forward()
                                              ↓
                              logits → SHAP values [n_measurements, vocab]
```


## shap_pickle

Serialized SHAP analysis results for a cohort of participants.

### File Output
- `shap_missingness.pickle.gz`: Gzip-compressed dictionary of per-participant SHAP results (overridable with `--fname`)

### Schema

```python
shap_pickle: dict[int | str, ...]
# Key: participant ID (int)  →  per-participant results
# Key: "tokenizer"  →  the dataset tokenizer (name → int mapping)
```

Per-participant value:
```python
{
    "shap":       np.ndarray,  # shape [n_measurements, vocab_size], dtype float16
    "features":   list[str],   # length n_measurements — measurement labels
    "timesteps":  np.ndarray,  # shape [n_measurements], dtype float16 — days
}
```

All arrays are position-aligned. Features are at **measurement granularity** (one per modality × time-point), not scalar granularity.

### Field Descriptions

| Field | Type | Description |
| --- | --- | --- |
| `shap` | `np.ndarray [n_measurements, vocab_size]` | SHAP attribution in logit-space. `shap[i, j]` = contribution of measurement `i` to predicting token `j`. Baseline = all biomarkers absent. |
| `features` | `list[str]` | Measurement labels in `"{MODALITY}@{timestep:.0f}"` format, e.g. `"WBC@2500"`, `"PRS@0"`. |
| `timesteps` | `np.ndarray [n_measurements]` | Timestamp (in days) for each measurement. |

### Notes

-   `n_measurements` varies per participant (different biomarkers available at different timepoints)
-   SHAP values are in **logit-space** (attributions sum to difference in logits from baseline)
-   Baseline = complete biomarker missingness (all measurements absent)
-   Only biomarker measurements are SHAP features; discrete tokens are always present

### Example Usage

```python
import gzip
import pickle

with gzip.open("shap_missingness.pickle.gz", "rb") as f:
    shap_pickle = pickle.load(f)

tokenizer = shap_pickle["tokenizer"]

pid = 123456
result = shap_pickle[pid]
shap_values = result["shap"]    # [n_measurements, vocab_size], float16
features = result["features"]   # ["WBC@2500", "PRS@0", ...]
timestamps = result["timesteps"]

# Top biomarker measurements contributing to a specific disease
outcome_idx = tokenizer["heart_failure"]
contributions = shap_values[:, outcome_idx].astype(float)
top_idx = np.argsort(np.abs(contributions))[::-1][:10]
for idx in top_idx:
    print(f"{features[idx]} (day {timestamps[idx]:.0f}): {contributions[idx]:.4f}")

# All WBC measurements across time
wbc_mask = np.array([f.startswith("WBC@") for f in features])
wbc_times = timestamps[wbc_mask]
wbc_contrib = shap_values[wbc_mask, outcome_idx].astype(float)
plt.scatter(wbc_times, wbc_contrib)
plt.xlabel("Days"); plt.ylabel("SHAP value for heart_failure")
```

### Creation Script

`apps/run_shap_m4.py` — runs the full pipeline for a checkpoint and saves results.

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

## Extension Points

Potential areas for future work:

- **Vocabulary mapping**: Functions accept optional `disease_names: Dict[int, str]` for human-readable labels. Currently defaults to "Disease {idx}".
- **Heatmap visualization**: `plot_feature_disease_heatmap()` exists for multi-feature, multi-disease overview (less polished than main functions)
- **Temporal analysis**: Could add statistical tests for trend significance, or support for different binning strategies
- **Export functionality**: Could add functions to export plots or DataFrames to files

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
