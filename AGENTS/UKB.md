# UKBDataset (delphi/data/ukb.py)

## Constructor Arguments

### Data Loading
| Argument | Description |
|----------|-------------|
| `data_dir` | Directory containing binary files, tokenizer, and p2i.csv |
| `subject_list` | Path to participant list (relative to `data_dir`), e.g., `"participants/train_fold.bin"` |
| `memmap` | Use memory-mapped files instead of loading into RAM |
### Preprocessing Toggles
| Argument | Description |
|----------|-------------|
| `no_event_interval` | Interval (in days) for inserting no-event tokens. `None` disables. |
| `no_event_mode` | Strategy for placing no-event tokens: `"random"`, `"regular"`, `"legacy-random"`, `"exponential"` |
| `perturb` / `perturb_list` | Enable timestamp perturbation for specified tokens (default: lifestyle). Rarely used. |
| `exclude` / `exclude_list` | Remove specified tokens entirely (default: lifestyle). Rarely used. |
| `block_size` | Max sequence length. `None` disables cropping. |
| `crop_mode` | How to crop: `"left"`, `"right"`, or `"random"` |
| `break_clusters` | Enable cluster dissolution (see Preprocessing Pipeline) |
| `additional_dx_token` | If `True`, add new `dx` token to vocab. If `False`, reuse `NO_EVENT_TOKEN`. |
### Reproducibility
| Argument | Description |
|----------|-------------|
| `seed` | RNG seed for stochastic transforms |
| `deterministic` | If `True`, seed RNG per-patient (`pid + seed`) for reproducible augmentation |


# MultimodalUKBDataset (delphi/data/ukb.py)
MultimodalUKBDataset is responsible for loading multi-modal data from the UK Biobank
– flexibly loads data from different bioamrkers and expansion packs
  – a biomarker (also referred to as Modality) contains continuous vectors; examples: whole blood count
  – an expansion pack contains additional discrete tokens beyond the base vocabulary; examples: tokens for surgical operations

## get_batch(batch_idx)

Collates multiple samples into a batch for model training/inference.

### Arguments
- `batch_idx`: An iterable of integer indices corresponding to samples in the dataset

### Returns
A tuple of 7 elements: `(X0, T0, bio_X_dict, bio_T, bio_M, X1, T1)`

**Discrete token sequences (for next-token prediction):**
- `X0`: Input tokens. Shape: `[batch_size, seq_len]`, dtype: `torch.long`, padding: `0`
- `T0`: Timestamps for input tokens (in days). Shape: `[batch_size, seq_len]`, dtype: `torch.float32`, padding: `-1e4`
- `X1`: Target tokens (X0 shifted by 1). Shape: `[batch_size, seq_len]`, dtype: `torch.long`, padding: `0`
- `T1`: Timestamps for target tokens. Shape: `[batch_size, seq_len]`, dtype: `torch.float32`, padding: `-1e4`

`seq_len` is the maximum sequence length across samples in the batch.

**Continuous biomarker data:**
- `bio_X_dict`: Dictionary mapping `Modality` → `torch.Tensor` containing feature vectors for that modality across the entire batch
  - Shape: `[num_measurements_for_modality_in_batch, feature_dim]`
  - `feature_dim` is modality-specific
  - Data is ordered by sample index, then chronologically within each sample
  - Only modalities present in the batch appear as keys
- `bio_T`: Timestamps for biomarker measurements (in days). Shape: `[batch_size, multimodal_seq_len]`, dtype: `torch.float32`, padding: `-1e4`
- `bio_M`: Modality index for each position (values from `Modality` enum). Shape: `[batch_size, multimodal_seq_len]`, dtype: `torch.long`, padding: `0`

`multimodal_seq_len` is the maximum number of biomarker measurements per participant in the batch.

### Alignment between bio_X_dict and bio_T/bio_M

`bio_T` and `bio_M` are position-aligned: `bio_T[i, j]` is the timestamp and `bio_M[i, j]` is the modality for the j-th biomarker measurement of sample i (sorted chronologically across all modalities).

`bio_X_dict[modality]` contains feature vectors for that modality, concatenated across samples in sample order (sample 0 first, then sample 1, etc.), chronologically within each sample.

To reconstruct a dense tensor for fusion:
```python
mask = (bio_M == modality.value)
dense_tensor[mask] = bio_X_dict[modality]
```
