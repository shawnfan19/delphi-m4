# Eval Saliency Task

## Overview

`apps/saliency_biomarker.py` computes **gradient-based saliency** of the model's predicted log-intensities with respect to the raw continuous features of a chosen biomarker modality. For each participant and each model target disease, it records:

$$s_{p,d,f} = \frac{\partial \hat{\lambda}_d}{\partial x_f}\bigg|_{x = x_p}$$

where $\hat{\lambda}_d$ is the model's predicted log-intensity for disease $d$ at the last sequence position (conditioned on all of participant $p$'s events and biomarkers), and $x_f$ is the $f$-th feature of the chosen biomarker modality.

This is a **local, individual-level** sensitivity: a large $|s_{p,d,f}|$ means that, at participant $p$'s current biomarker values, a small change to feature $f$ would meaningfully shift their predicted risk of disease $d$.

## Interpretation

### Signed gradients

By default the signed gradient is stored. The sign indicates direction: $s_{p,d,f} > 0$ means increasing feature $f$ raises predicted risk of disease $d$ for this participant; $s_{p,d,f} < 0$ means it lowers risk. Use `--abs` to store magnitudes only.

### Saturation

A near-zero gradient can arise from two distinct situations:

1. **Saturated regime**: the biomarker value is extreme enough that the model's output is locally flat there. This is informative — it means small perturbations to this feature won't move the prediction, because the model is already "maximally alarmed" (or maximally reassured). Biologically plausible for very high LDL, very low eGFR, etc.
2. **Genuine non-use**: the model does not rely on this feature for this participant's prediction regardless of value.

These two cases are indistinguishable from the local gradient alone. If distinguishing them matters, consider Integrated Gradients (with a population-mean baseline), which accumulates the gradient along the path from baseline to the observed value and would show a large total attribution even when the endpoint gradient is near zero.

## Implementation (`apps/saliency_biomarker.py`)

### Algorithm

For each batch of $B$ participants:

1. **Forward-mode Jacobian** (`torch.func.jacfwd`): compute the full Jacobian of the target logits with respect to the biomarker feature tensor in one call. The forward function returns only target disease logits `(B, n_targets)` rather than the full vocabulary, and `jacfwd` propagates `n_features` tangent vectors through the forward pass to produce the `(B, n_targets, B, n_features)` Jacobian.

2. **Batch diagonal**: since samples are independent (no cross-attention between participants), only the batch diagonal is non-zero. Extract it to get `(B, n_targets, n_features)`.

3. **Reshape and store**: transpose to `(B, n_features, n_targets)`, then write one dict entry per participant.

### Forward-mode vs reverse-mode AD

Computing the Jacobian $J \in \mathbb{R}^{n_\text{targets} \times n_\text{features}}$ requires either:

- **Reverse-mode** (`jacrev`): one backward pass per output → `n_targets` backward passes
- **Forward-mode** (`jacfwd`): one forward pass per input → `n_features` forward passes

Since `n_features` $\ll$ `n_targets` for all biomarker modalities (e.g. LIPID has 4 features vs ~1270 targets), forward-mode is dramatically faster:

| Modality | n_features | n_targets | Forward passes (jacfwd) | Backward passes (jacrev) | Speedup |
|----------|-----------|-----------|------------------------|-------------------------|---------|
| LIPID    | 4         | ~1270     | 4                      | ~1270                   | ~300x   |
| WBC      | 31        | ~1270     | 31                     | ~1270                   | ~40x    |
| NMR      | 251       | ~1270     | 251                    | ~1270                   | ~5x     |

Forward-mode also avoids storing the computation graph (no `retain_graph`), giving lower peak GPU memory.

### B^2 Jacobian overhead

`jacfwd` (and `jacrev`) over a batched function `(B, F) → (B, n_targets)` produces a `(B, n_targets, B, F)` Jacobian. The cross-sample entries are all zero but still materialised. This limits batch size at moderate B. A `vmap` + per-sample `jacfwd` approach would produce `(B, n_targets, F)` directly, eliminating the $B^2$ factor, but is impractical here because `bio_X_dict` is flattened across the batch (the model does not expose a per-sample interface).

### Query position

`logits[:, -1, :]` is the prediction at the **last position of the fused sequence** — the position after the most recent event or biomarker measurement, whichever occurs latest in the timeline. This is the natural "current state" query point for a static set of biomarker inputs and gives the same prediction that `multimodal_shap_forward` uses.

### Forward function and `functools.partial`

`_sal_forward(bio_x, *, ...)` is a standalone function whose fixed inputs (all batch tensors except the target modality) are frozen per-batch via `functools.partial`. Only `bio_x` — the target modality tensor — participates in the computation graph.

## Output

A gzip-compressed pickle file written alongside the checkpoint:

```python
{
    pid: {
        "jacobian":  np.float16 array, shape (n_meas * n_features, n_targets),
        "timestamp": float,  # age at last target position (model output)
    },
    "targets":   ["e11_(type_2_diabetes)", "i21_(acute_myocardial_infarction)", ...],
    "tokenizer": {...},
    "modality":  "LIPID",  # modality name string
}
```

- `results[pid]["jacobian"]` is the full Jacobian flattened across measurements. When a participant has `n_meas` biomarker measurements, reshape to `(n_meas, n_features, n_targets)` to recover per-measurement gradients. Average across axis 0 if a single gradient per feature is needed.
- `results[pid]["timestamp"]` is the age (in years) at the model's last prediction position — used together with biomarker measurement time to compute the **time horizon** of prediction.
- `results["targets"][d]` gives the disease name of the $d$-th target, matching the last axis of the Jacobian.

Output filename: `{ckpt_dir}/saliency-{MODALITY}-ckpt-{ckpt_stem}.pkl.gz`

## Usage

```bash
python apps/saliency_biomarker.py --modality LIPID
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--ckpt` | `delphi-m4/delphi-m4/ckpt.pt` | Path relative to `$DELPHI_CKPT_DIR` |
| `--batch_size` | 1 | Inference batch size (limited by B^2 Jacobian memory) |
| `--modality` | *(required)* | Biomarker modality to analyse (e.g. `LIPID`, `WBC`, `NMR`) |
| `--abs` | False | Store absolute value of gradients instead of signed values |
| `--subsample` | None | Limit to first N participants |
| `--fname` | auto | Override output filename |

## Sanity Checks

After the output is written, verify:

1. Gradient arrays are finite (`np.isfinite`) for all participants and features — NaN/Inf indicates a numerical issue in the forward pass.
2. Gradients should not be identically zero across all features for a given participant — this would indicate the biomarker tensor is not on the computation graph.
3. Sign patterns for well-understood features should be biologically plausible: e.g., higher LDL should increase gradient for cardiovascular diseases ($s > 0$), not decrease it.
4. Re-running with the same checkpoint and same `--modality` produces identical results (no randomness in the pipeline).

## Visualisation (`apps/vis_saliency.py`)

Loads saliency output and raw biomarker values, then plots **saliency vs biomarker value** scatter plots (one subplot per feature) for a chosen target disease. Key details:

- **Rescaling**: gradients are divided by `bio_std[f]` to convert from per-z-unit to per-raw-unit ($\partial \log\lambda / \partial x_f$). The y-axis formatter shows the corresponding percent change in hazard.
- **Multiple measurements**: for participants with `n_meas > 1`, the Jacobian is reshaped to `(n_meas, n_features, n_targets)` and averaged across measurements.
- **Time-horizon stratification**: the **time horizon** = `sal_timestamp − biomarker_measurement_time` (years between when the blood was drawn and the model's prediction target age) is computed per participant. Scatter points and LOWESS trend lines are stratified into three bins:
  - **<5 yr** (red) — short-term predictions
  - **5–10 yr** (orange) — medium-term
  - **>10 yr** (blue) — long-term
- **LOWESS**: optional smoothed trend lines per stratum (controlled by `lowess_frac`). Each stratum gets its own coloured line.

### Biomarker class methods used

- `Biomarker.first_occurrence_times(pids)` → `np.ndarray` of first measurement timestamps (one per pid; NaN if missing). Used to compute the time horizon.
- `Biomarker.occurrence_times(pids)` → `dict[int, np.ndarray]` of all measurement timestamps per pid. Available for more fine-grained temporal analyses.

## Related Code

- `delphi/model/multimodal.py`: `BiomarkerEmbedding`, `DelphiEmbedding`, `fuse_embed`, `DelphiM4.forward` — the differentiable path from biomarker features to logits
- `delphi/data/ukb.py`: `Biomarker` (`.features`, `.first_occurrence_times()`, `.occurrence_times()`), `MultimodalUKBDataset` (`.mod_ds` dict)
- `delphi/experiment.py`: `eval_iter`, `load_ckpt`, `move_batch_to_device`
- `apps/vis_shap.py`: visualisation patterns applicable to saliency output
