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
