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

## to\_shap\_array(s, detokenizer, biomarker\_features, biomarker\_background)

Converts per-participant data from `MultimodalUKBDataset.__getitem__` format to `ShapArray`.

### Arguments

-   `s`: `MultimodalOut` tuple `(x, t, bio_x_dict, bio_t, bio_m)` — first 5 elements from `dataset[i]`
-   `detokenizer`: `dict[int, str]` mapping token IDs to names (from `dataset.detokenizer`)
-   `biomarker_features`: `dict[Modality, list[str]]` mapping modality to list of feature names
-   `biomarker_background`: `dict[Modality, np.ndarray]` mapping modality to background values (typically training cohort means)


### Returns

Tuple of `(shap_array, features, bio_bg)`:
-   `shap_array`: `ShapArray` tuple `(all_x, all_t, all_m)`
-   `features`: `list[str]` of feature names, position-aligned with `shap_array`
    -   Token features: `["diabetes", "stroke", ...]`
    -   Biomarker features: `["PRS.feature1", "PRS.feature2", ..., "WBC.feature1", ...]`
-   `bio_bg`: `np.ndarray` of background values for biomarker features (for SHAP baseline)


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

Masker for discrete token sequences. Used internally by `MultimodalShapMasker`.

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

Combined masker for multimodal data (tokens + biomarkers). Wraps `ShapMasker` for token handling.

### Constructor
```python
MultimodalShapMasker(biomarker_background: np.ndarray)
```

-   `biomarker_background`: 1D array of background values for all biomarker features, position-aligned with the biomarker portion of `ShapArray`. Typically training cohort means (from `to_shap_array`'s `bio_bg` output).


### Masking Strategies

| Feature Type | Strategy |
| --- | --- |
| Tokens | Delegated to `ShapMasker` (drop/swap/replace) |
| Biomarkers | **Substitute** with background mean |

### **call**(mask, s)

#### Arguments

-   `mask`: Boolean array where `True` = feature participates, `False` = feature is masked

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

# 2. Convert to SHAP-compatible format
sample, features, bio_bg = to_shap_array(
    (x, t, bio_dict, bio_t, bio_m),
    detokenizer=ds.detokenizer,
    biomarker_features=biomarker_features,      # dict[Modality, list[str]]
    biomarker_background=biomarker_background,  # dict[Modality, np.ndarray]
)

# 3. Setup masker and explainer
masker = MultimodalShapMasker(bio_bg)
shap_model = partial(multimodal_shap_forward, model=model)
explainer = shap.Explainer(
    shap_model,
    masker,
    feature_names=np.array([features]),
    output_names=list(tokenizer.keys()),
)

# 4. Compute SHAP values
shap_values = explainer([sample])
# shap_values.values: [1, num_features, vocab_size] — attribution per feature per output
# shap_values.base_values: [1, vocab_size] — baseline prediction
# shap_values.data: the input features
```

### Data Flow Diagram

```text
ds[i] → MultimodalOut → to_shap_array() → ShapArray
                                              ↓
                            MultimodalShapMasker.__call__(mask, ShapArray)
                                              ↓
                                        masked ShapArray
                                              ↓
                            multimodal_shap_forward() → from_shap_array()
                                              ↓
                                      model.forward()
                                              ↓
                                    logits → SHAP values
```


## shap_pickle

Serialized SHAP analysis results for a cohort of participants.

### File Output
- `shap.pickle.gz`: Gzip-compressed dictionary of per-participant SHAP results

### Schema

```python
shap_pickle: dict[int, dict]
# Key: participant ID (int)
# Value: dict with keys:
#   "shap": np.ndarray, shape [n_features, vocab_size], dtype float32
#   "features": list[str], length n_features — feature names
#   "timesteps": np.ndarray, shape [n_features], dtype float32 — timestamps in days
```

All arrays are position-aligned. The `no_event` token is excluded as it functions as padding/augmentation and carries no semantic meaning.

### Field Descriptions

| Field | Type | Description |
| --- | --- | --- |
| `shap` | `np.ndarray [n_features, vocab_size]` | SHAP attribution values in logit-space. `shap[i, j]` = contribution of feature `i` to predicting output token `j`. |
| `features` | `list[str]` | Feature names. Discrete tokens use model tokenizer names (e.g., `"diabetes"`). Biomarker features use `"{MODALITY}.{feature}"` format (e.g., `"WBC.rbc"`). |
| `timesteps` | `np.ndarray [n_features]` | Timestamp (in days) when each feature occurred. |

### Notes

-   `n_features` varies per participant (different sequence lengths, different biomarkers available)

-   SHAP values are in **logit-space** (attributions sum to difference in logits from baseline)

-   Features include discrete tokens (diseases, procedures, sex) and biomarker components


### Example Usage

```python
import gzip
import pickle

# Load results
with gzip.open("shap.pickle.gz", "rb") as f:
    shap_pickle = pickle.load(f)

# Access participant results
pid = 123456
result = shap_pickle[pid]
shap_values = result["shap"]        # [n_features, vocab_size]
features = result["features"]        # list of strings
timestamps = result["timesteps"]     # [n_features]

# Use case 1: Top contributors to a specific disease
outcome_idx = tokenizer["heart_failure"]
contributions = shap_values[:, outcome_idx]
top_idx = np.argsort(np.abs(contributions))[::-1][:10]

for idx in top_idx:
    print(f"{features[idx]} (day {timestamps[idx]:.0f}): {contributions[idx]:.4f}")

# Use case 2: Track contribution of a feature type over time
diabetes_mask = np.array([f == "diabetes" for f in features])
diabetes_times = timestamps[diabetes_mask]
diabetes_contrib = shap_values[diabetes_mask, outcome_idx]
plt.scatter(diabetes_times, diabetes_contrib)
plt.xlabel("Days"); plt.ylabel("SHAP value for heart_failure")
```

### Creation Code

```python
shap_pickle = dict()
for i in trange(len(ds)):
    x, t, bio_dict, bio_t, bio_m, _, _ = ds[i]
    pid = ds.participants[i]

    sample, features, bio_bg = to_shap_array(
        (x, t, bio_dict, bio_t, bio_m),
        detokenizer=ds.detokenizer,
        biomarker_features=biomarker_features,
        biomarker_background=biomarker_background,
    )
    all_x, all_t, all_m = sample
    no_event = np.array(["no_event" in f for f in features]).astype(bool)

    masker = MultimodalShapMasker(bio_bg)
    shap_model = partial(multimodal_shap_forward, model=model)
    explainer = shap.Explainer(
        shap_model,
        masker,
        feature_names=np.array([features]),
        output_names=list(tokenizer.keys()),
    )
    shap_values = explainer([sample])

    shap_pickle[int(pid)] = {
        "shap": shap_values.values[0, ~no_event, :].astype(np.float32),
        "features": np.array(features)[~no_event].tolist(),
        "timesteps": all_t[~no_event].astype(np.float32),
    }

with gzip.open(ckpt.parent / "shap.pickle.gz", "wb") as f:
    pickle.dump(shap_pickle, f)
```

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
