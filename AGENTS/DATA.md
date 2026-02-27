## Data Storage Architecture

This project uses a memory-efficient flat-file storage scheme for patient event sequences.

### Binary Files

| File | Dtype | Description |
|------|-------|-------------|
| `data.bin` | `uint32` | Contiguous token sequences (all patients concatenated, order arbitrary) |
| `time.bin` | `uint32` | Matching timestamps as **age in days since birth** |
| `participants/*.bin` | `uint32` | Person IDs for train/val/test splits |

**Key invariant**: Each patient's tokens are stored contiguously. `data.bin[i]` and `time.bin[i]` always correspond to the same event.

### Index File

`p2i.csv` maps each `person_id` to their location in the binary files:
- `pid`: person_id
- `start_pos`: byte offset where this person's sequence begins
- `seq_len`: number of tokens for this person

### Tokenizer

`tokenizer.yaml` is a flat `name → int` mapping:
- `0`: Padding token
- `1`: No-event token (for TPP intensity recomputation)
- Remaining: ICD disease codes, lifestyle factors (BMI, smoking, alcohol), sex

### Design Rationale

- **Flat binary storage**: Enables fast random access per patient without loading entire dataset
- **Separate participant lists**: Allows flexible train/val/test splits without duplicating data
- **Pre-sorted sequences**: Tokens stored in temporal order per patient (though preprocessing may require re-sorting)

---

## Preprocessing Pipeline

`__getitem__` applies transforms in a specific order. Each step has a corresponding `identity_transform` fallback when disabled.

### Pipeline Order

| Step | Function | Purpose |
|------|----------|---------|
| 1 | `exclude_tokens` | Remove tokens by blacklist (e.g., lifestyle tokens). Rarely used. |
| 2 | `append_no_event` | Insert synthetic no-event tokens for TPP intensity recomputation |
| 3 | `perturb_time` | Data augmentation via timestamp noise. Rarely used. |
| 4 | `sort_by_time` | Restore temporal order (steps 2-3 may disrupt it) |
| 5 | `crop_block_size` | Truncate to fixed context length (left/right/random crop) |
| 6 | `dissolve_clusters` | Handle same-day disease clusters for exponential inter-event times |

### Cluster Dissolution

**Problem**: Multiple diseases on the same day → inter-event time of 0 → outside exponential distribution support.

**Solution**:
1. Identify disease tokens (anything not in whitelist: no-event, sex, lifestyle)
2. Insert `dx_token` at each unique disease timestamp
3. Perturb each disease backward into the preceding interval: `t_disease → t_disease - ε`, where `ε ∈ (0, Δt_prev)`
4. Re-sort by time

**Result**: Model sees scattered disease tokens, then `dx_token` signals "a diagnosis cluster occurred at this moment." All inter-event times become strictly positive.

**Inverse**: `pack_clusters` reverses this for inference — restores original timestamps and removes `dx_token`s.

### Whitelist

Tokens exempt from cluster dissolution:
- `NO_EVENT_TOKEN` (synthetic)
- Sex tokens (demographic)
- Lifestyle tokens (survey-based, not diagnoses)

### Determinism

When `deterministic=True`:
- RNG seeded per-patient (`pid + seed`) for reproducible augmentation
- Stable sort used to ensure consistent ordering of ties

---

## UKBDataset (`delphi/data/ukb.py`)

### Constructor Arguments

**Data Loading**
| Argument | Description |
|----------|-------------|
| `data_dir` | Directory containing binary files, tokenizer, and p2i.csv |
| `subject_list` | Path to participant list (relative to `data_dir`), e.g., `"participants/train_fold.bin"` |
| `memmap` | Use memory-mapped files instead of loading into RAM |

**Preprocessing Toggles**
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

**Reproducibility**
| Argument | Description |
|----------|-------------|
| `seed` | RNG seed for stochastic transforms |
| `deterministic` | If `True`, seed RNG per-patient (`pid + seed`) for reproducible augmentation |

---

## MultimodalUKBDataset (`delphi/data/ukb.py`)

Extends `UKBDataset` to load multi-modal data from the UK Biobank:
- **Biomarkers** (also referred to as Modalities): continuous feature vectors, e.g. whole blood count
- **Expansion packs**: additional discrete tokens beyond the base vocabulary, e.g. tokens for surgical operations

### `get_batch(batch_idx)`

Collates multiple samples into a batch for model training/inference.

**Arguments**: `batch_idx` — an iterable of integer indices corresponding to samples in the dataset

**Returns** a 7-tuple: `X0, T0, bio_x_dict, bio_T, bio_M, X1, T1`

| Tensor | Shape | Description |
|--------|-------|-------------|
| `X0` | `(B, L)` | Token sequences (disease codes, no-event, lifestyle, etc.) |
| `T0` | `(B, L)` | Token timestamps in days; `-1e4` for padding |
| `bio_x_dict` | `dict[Modality, Tensor]` | Raw biomarker feature vectors (see below) |
| `bio_T` | `(B, L_bio)` | Biomarker timestamps; `-1e4` for padding |
| `bio_M` | `(B, L_bio)` | Modality index per position (0 = padding) |
| `X1` | `(B, L)` | Target tokens (next event) |
| `T1` | `(B, L)` | Target timestamps |

All tensors are **left-padded** (valid entries right-aligned; `collate_batch` uses `pad_left=True` by default). Padding sentinels: token `0` and timestamp `-1e4`.

`L` is the maximum sequence length across the batch; `L_bio` is the maximum number of biomarker measurements per participant in the batch.

#### Alignment between `bio_x_dict` and `bio_T`/`bio_M`

`bio_T` and `bio_M` are position-aligned: `bio_T[i, j]` is the timestamp and `bio_M[i, j]` is the modality for the j-th biomarker measurement of sample i (sorted chronologically across all modalities).

### `bio_x_dict` Row-Indexing Invariant

`bio_x_dict[M]` is a **flat 2-D tensor** of shape `(K, input_size)` where `K` is the total number of modality-M measurements across the entire batch — it is **not** batched by sample.

**Invariant**: row `k` of `bio_x_dict[M]` corresponds to the `k`-th `True` entry of `(bio_M == M.value)` in **row-major order** (iterating over batch dim first, then position dim).

This arises from `get_batch`:
```python
for idx in batch_idx:
    x0, t0, bio_x_dict, bio_t, bio_m, x1, t1 = self[idx]
    for modality in bio_x_dict.keys():
        bio_X_dict[modality].extend(bio_x_dict[modality])  # appends per-sample rows
```

The model recovers per-position embeddings via:
```python
mod_mask = mod_idx == modality.value   # (B, L_bio) boolean
biomarker_emb[modality] += mod_age_emb[mod_mask]   # selects K positions
```

**Consequence**: any operation that changes which positions in `bio_M` carry a given modality value — masking entries to 0, removing positions — **silently breaks** `bio_x_dict` unless you reindex it to match. The row count in `bio_x_dict[M]` must always equal `(bio_M == M.value).sum()`.

#### Reindexing pattern

After modifying `bio_M` (e.g. masking entries to 0), reindex `bio_x_dict` as follows:

```python
for mod, bio_x in bio_x_dict.items():
    old_mask = bio_m_orig == mod.value   # (B, L_bio) — before modification
    new_mask = bio_m_new  == mod.value   # (B, L_bio) — after modification
    keep = new_mask[old_mask]            # 1-D bool over original rows
    if keep.any():
        new_bio_x_dict[mod] = bio_x[keep]
    # if not keep.any(): omit this modality from the dict entirely
```

To **drop a modality entirely** (e.g. for ablation), also set `bio_M[bio_M == M.value] = 0` and `bio_T[...] = -1e4`, then exclude the key from `bio_x_dict`. Do not leave a mismatched key in the dict.

### `must_have_biomarkers` — Filtering Participants

```python
data_args["must_have_biomarkers"] = ["wbc"]   # lowercase strings
ds = MultimodalUKBDataset(**data_args)
```

Accepts a list of **lowercase modality name strings**; internally calls `Modality[name.upper()]`. Filters `self.participants` to only those present in every listed modality's `Biomarker.pids`. Apply before the eval loop to avoid per-batch `has_modality` checks.

The modality must also appear in `data_args["biomarkers"]` (loaded from the checkpoint) — if it wasn't part of the training biomarkers list, `self.mod_ds` won't contain it and the filter will fail.

---

## Model Forward Pass — Multimodal (`DelphiM4`)

```python
out_dict, loss, att = model(X0, T0, bio_x_dict, bio_T, bio_M, X1, T1)
```

Argument names in the model signature: `(idx, age, biomarker, mod_age, mod_idx, targets, targets_age)`.

### Output shape depends on whether targets are passed

| Call | `out_dict["logits"]` shape | Notes |
|------|---------------------------|-------|
| With `targets` / `targets_age` | `(B, L, V)` | Logits filtered to token positions only (biomarker positions excluded via `fuse_targets_mask`) |
| Without targets (eval mode) | `(B, L_fused, V)` | Logits for **all** fused positions; `L_fused = L_bio + L` |

### Fused Sequence — Sort Semantics

Inside `DelphiM4.forward`, `fuse_embed` concatenates biomarker and token embeddings then sorts ascending by timestamp:

```python
fused_age_unsorted = torch.cat((mod_age, age), dim=1)   # biomarkers first, then tokens
sort_indices = torch.argsort(fused_age_unsorted, stable=True, dim=1)
```

Two consequences of `stable=True` + biomarkers-first concat order:

1. **At equal timestamps, biomarkers precede tokens.** A biomarker and a disease token recorded on the same day are ordered: biomarker → disease token. This is intentional: the model sees the biomarker value before predicting the disease event at the same timestep.

2. **Padding (`-1e4`) sorts to the front.** All padding positions land at the beginning of the sorted sequence. Valid positions are always right-aligned.

### `logits[:, -1, :]` is the last real position

Because padding sorts to the front, the **last column of the logits tensor** (`logits[:, -1, :]`, shape `(B, V)`) is always the model's prediction after the most recent real event for every sample in the batch — regardless of how much padding exists. This holds both when targets are omitted and when the sequence has been manually truncated by masking tokens to `-1e4`.

---

## `move_batch_to_device`

```python
from delphi.experiment import move_batch_to_device
batch = move_batch_to_device(batch, device=device)
```

Handles **tensors** and **dicts of tensors** only. Raises `NotImplementedError` for any other type. The 7-tuple batch format satisfies this contract (all elements are tensors or `bio_x_dict` which is a dict).

When constructing a sub-batch to pass directly to the model (e.g. after `remove_after`), move the full batch to device first, then manipulate — cloning device tensors is cheaper than moving CPU tensors after manipulation.
