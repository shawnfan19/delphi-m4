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
- `start_pos`: element index into the flat arrays where this person's sequence begins
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

## AoU Storage and Cross-Dataset Alignment (UKB ↔ AoU)

The repo supports two datasets — UK Biobank and All of Us — and the
data layer is designed so downstream code (eval scripts, model training,
plotting) can run dataset-agnostically against either via
`delphi.data.auto.multimodal_reader_cls()`. Alignment is enforced at
three levels.

### Storage layout — UKB vs AoU side by side

| Aspect | UKB | AoU |
|--------|-----|-----|
| Base dir | `data/ukb_real_data/` | `data/aou_uk/` |
| Token stream | `data.bin` + `time.bin` (`uint32` flat arrays) + `p2i.csv` (`pid, start_pos, seq_len`) | `data.parquet` (rows: `person_id, age_in_days, token`); indexed at load time via `np.unique` |
| Biomarker storage | `biomarkers/{name}/data.bin` (float32) + `features.yaml` + `p2i.csv` (`pid, time, start_pos, seq_len`) | `biomarkers/{name}/data.parquet` (columns: `person_id, age_in_days`, plus per-feature `{feat}` and `{feat}_raw_value/_unit_id/_unit_name/_concept_id/_concept_name`) |
| Expansion packs | `expansion_packs/{name}/{data.bin, time.bin, p2i.csv, tokenizer.yaml}` | `expansion_packs/{name}/{data.parquet, tokenizer.yaml}` |
| Disease labels | `labels_chapters_colours.csv` | `labels_chapters_colours.csv` (a copy of UKB's; see below) |

Despite the different on-disk formats, downstream code is dataset-agnostic:
`delphi.data.auto.multimodal_reader_cls()` returns the concrete
`MultimodalReader` subclass for the active dataset (env `DELPHI_DATASET`, else
auto-detected from the data dir), and code is written against the one reader
contract — the 5-tuple `__getitem__` and the trajectory / `participants(fold)` /
`labels()` surface. See [Readers and Datasets](#readers-and-datasets). The
per-dataset storage difference is confined to each reader's `_load*` methods.

### Disease token alignment

Both datasets use the **same tokenizer** — AoU's vocabulary is a strict
subset of UKB's:

- The canonical UKB disease tokenizer is `data/ukb/dictionary/tokenizer.yaml` —
  lowercase `e78_(disorders_of_lipoprotein_...)` form, one ICD-10 3-char prefix
  per token. (Git-tracked in the repo so the AoU prep can read it without the
  UKB data volume.)
- `data/aou/core.py` reads that tokenizer, extracts the ICD-10 prefix (regex
  `[a-z]\d{2}`) from the keys, and maps AoU's SNOMED→ICD-10 path onto those
  tokens. AoU-only codes are dropped; unmappable codes are logged to
  `aou_uk/missing_icd_codes.yaml`.

**Practical implication**: a token name like
`e78_(disorders_of_lipoprotein_metabolism_and_other_lipidaemias)` resolves
to a token in BOTH `data/ukb_real_data/tokenizer.yaml` AND
`data/aou_uk/tokenizer.yaml`. Cross-dataset comparisons keyed on disease
token names work without translation.

**Disease labels** (ICD chapter + color per token, read by `reader.labels()`)
are vocabulary metadata keyed on this shared tokenizer, so they are dataset-
independent. The canonical file is `data/ukb/dictionary/labels_chapters_colours.csv`
(git-tracked); `data/aou/core.py` uploads a byte-faithful copy to
`aou_uk/labels_chapters_colours.csv` so `MultimodalAOUReader.labels()` resolves.

### Biomarker alignment

`data/biomarker.yaml` is the master cross-reference between UKB field
IDs and AoU OMOP concept IDs. Each biomarker entry has dual keys:

```yaml
cholesterol:
  ukb: 30690                # UKB field ID
  aou:
    id: [3027114]           # OMOP concept ID(s)
    unit: {8840: 0.02586}   # unit conversion factor → mmol/L
  range: [0, 20]            # QC bounds (post-conversion)
```

Both prep pipelines (`data/ukb/codon/gather.py` for UKB;
`data/aou/biomarker.py` for AoU) consume this YAML and produce
files where:
- Feature names are identical across datasets (`cholesterol`, `hdl`,
  `ldl_direct`, `triglycerides`, etc.).
- Numeric values are in the same units (e.g., mmol/L for lipids).
- Plausibility-range bounds are applied identically.

**Modality directory naming** differs slightly between datasets. UKB has
`lipid/` (6 features: cholesterol, hdl, ldl_direct, triglycerides,
apolipoprotein_a, apolipoprotein_b). AoU has `lipid_panel/` (4 features:
cholesterol, hdl, ldl_direct, triglycerides — common subset). Callers
that need to work across datasets should resolve the modality via the
dataset's panel YAML (`data/panel/aou.yaml`) or by searching the active
reader's `biomarkers` dict, not by hardcoding the dir name.

**Distribution shifts** between datasets at the feature level are
documented in `data/aou/biomarker_stats.md`; e.g., cholesterol shows
~−0.82σ in AoU vs UKB (likely reflecting statin therapy prevalence in
the AoU cohort). This is the kind of artifact downstream analyses need
to account for.

### Expansion-pack alignment

Each pack's tokenizer is shared across datasets:

- UKB writes the canonical tokenizer (e.g.,
  `data/ukb/dictionary/prescriptions_tokenizer.yaml`,
  `ops_tokenizer.yaml`).
- AoU's `data/aou/medications.py` and `operations.py` load these UKB
  tokenizers directly. AoU-only codes are dropped; shared codes get
  identical local token IDs across datasets.

**Practical implication**: expansion-pack token IDs (within an
offset-adjusted merged vocab) are directly comparable across UKB and AoU.

### Pair-list convention for cross-dataset analyses

For analyses that iterate (disease, biomarker_feature) pairs (e.g.,
`plot/data/expansion_packs.py`, biomarker-distribution-by-disease
plots), pairs can be specified dataset-agnostically:

- Disease: a token name from the shared tokenizer (e.g.,
  `e78_(disorders_of_lipoprotein_metabolism_and_other_lipidaemias)`).
- Biomarker feature: a feature name shared across modalities (e.g.,
  `ldl_direct`).

The script resolves the modality directory per-dataset (`lipid` for UKB,
`lipid_panel` for AoU) by searching the active dataset's biomarker
catalog.

---

## Preprocessing Pipeline

`TokenTransform.__call__` applies transforms in a specific order. Each step has a corresponding `identity_transform` fallback when disabled.

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

## Readers and Datasets

The data layer is one shared slicing engine composed by per-dataset readers, all
obtained through the factory:

```python
from delphi.data.auto import multimodal_reader_cls
reader = multimodal_reader_cls()(biomarkers=["wbc"], expansion_packs=["ops"])
```

### Reader hierarchy (`delphi/data/reader.py`, `ukb.py`, `aou.py`)

- **`TokenReader`** — a pure pid-indexed `(token, time)` slicing engine over
  in-memory arrays: `__getitem__(pid) -> (x, t)`, plus `start_pos`, `seq_len`,
  `tokenizer`, `detokenizer`. It is **composed**, never subclassed per dataset.
- **`MultimodalReader(abc.ABC)`** — composes a main-stream `TokenReader` +
  expansion packs + biomarkers. The base owns everything dataset-agnostic: the
  5-tuple `__getitem__`, the trajectory queries over the main stream
  (`is_female`, `event_times`, `exit_times`, `participants_with_event`), the
  modality participant filters, the shared `labels()`, and the public
  `__init__(expansion_packs, biomarkers)` (which builds the components via the
  abstract seam + the `*_cls` hooks). The per-dataset **seam** is two abstract
  classmethods — `_load_token_reader()` (load the main stream into a
  `TokenReader`) and `participants(fold)` — plus the `base_dir` /
  `biomarker_cls` / `expansion_pack_cls` class attrs.
- **`MultimodalUKBReader` / `MultimodalAOUReader`** — the concrete per-dataset
  subclasses, essentially pure declarations: `base_dir`, the three bindings, the
  lifestyle/sex key lists, the two seam methods, plus genuine extras (UKB
  `recruitment_times` / `labels`; AoU `first_biomarker_times`).
- **`BiomarkerReader(abc.ABC)` → `Biomarker`** and
  **`ExpansionPackReader(abc.ABC)` → `ExpansionPack`** — the composed modality
  stores (own sections below). Both concretes are named the same in `ukb.py` and
  `aou.py`: only one dataset is ever live in a given secure environment, so the
  same-name classes are mutually exclusive at runtime and the factory resolves
  the right one.

There is no longer a unimodal reader or unimodal `Dataset`: the old
`UKBReader`/`AOUReader` were folded into the multimodal readers, and
`MultimodalDataset` is the only dataset class. Code that needs the raw main-stream
`(x, t)` 2-tuple (rather than the 5-tuple) reaches it via `reader.token_reader[pid]`.

### Biomarker indexing — `reader.biomarker2idx`

The integer modality ids in `bio_M` come from `reader.biomarker2idx` (a per-reader
`name -> int` map), **not** a global enum. `0` / `1` are reserved (padding / event
token); biomarkers start at `RESERVED_MOD_IDX = 2`. Pass `biomarkers=` as either:
- a **list** of names → indices auto-assigned from sorted order, or
- a **dict** `{name: idx}` → used as-is (e.g. a checkpoint's
  `model_args["biomarker2idx"]`, so eval matches training).

Names are lowercased. (The `Modality` enum in `delphi/multimodal.py` is a
model-side registry; it is not the source of `bio_M` ids — the biomarker set
churns, so prefer the per-reader / per-checkpoint `biomarker2idx`.)

### `MultimodalDataset` (`delphi/data/dataset.py`)

`MultimodalDataset(reader, pids, token_transform=None, biomarker_transform=None,
prompt_transform=None)` wraps a reader + a participant array, applies the
transforms in `__getitem__`, and collates batches. It is the only dataset class.

`__getitem__(idx) -> (x0, t0, bio_x_dict, bio_t, bio_m, x1, t1)`. The optional
transforms (all in `delphi/data/transform.py`) are:

- **`TokenTransform`** — `no_event_interval` / `no_event_mode`, `block_size` /
  `crop_mode`, `perturb_tokens`, `blacklist_tokens`, `break_clusters` (see
  Preprocessing Pipeline). Applied to `(x, t)`.
- **`BiomarkerTransform`** — `first_time_only`, `dropout`, `z_score` (+ `mean` /
  `std`), keyed by the reader's `biomarker2idx`. This is where biomarker
  normalization / first-visit selection / dropout now live (**not** on the
  `Biomarker` class).
- **`Prompt` / `MultimodalPrompt`** — split a sequence into prompt + ground truth
  at a cutoff age.

Transforms round-trip through a checkpoint via `.config` / `from_ckpt` / `to_ckpt`.

### `get_batch(batch_idx)`

Collates multiple samples into a batch for model training/inference (`collate`
doubles as a torch `DataLoader` `collate_fn`).

**Arguments**: `batch_idx` — an iterable of integer indices corresponding to samples in the dataset

**Returns** a 7-tuple: `X0, T0, bio_x_dict, bio_T, bio_M, X1, T1`

| Tensor | Shape | Description |
|--------|-------|-------------|
| `X0` | `(B, L)` | Token sequences (disease codes, no-event, lifestyle, etc.) |
| `T0` | `(B, L)` | Token timestamps in days; `-1e4` for padding |
| `bio_x_dict` | `dict[str, Tensor]` | Raw biomarker feature vectors keyed by lowercase modality name (see below) |
| `bio_T` | `(B, L_bio)` | Biomarker timestamps; `-1e4` for padding |
| `bio_M` | `(B, L_bio)` | Modality index per position (0 = padding; values from `reader.biomarker2idx`) |
| `X1` | `(B, L)` | Target tokens (next event) |
| `T1` | `(B, L)` | Target timestamps |

All tensors are **left-padded** (valid entries right-aligned; `collate_batch` uses `pad_left=True` by default). Padding sentinels: token `0` and timestamp `-1e4`.

`L` is the maximum sequence length across the batch; `L_bio` is the maximum number of biomarker measurements per participant in the batch.

#### Alignment between `bio_x_dict` and `bio_T`/`bio_M`

`bio_T` and `bio_M` are position-aligned: `bio_T[i, j]` is the timestamp and `bio_M[i, j]` is the modality for the j-th biomarker measurement of sample i (sorted chronologically across all modalities).

### `bio_x_dict` Row-Indexing Invariant

`bio_x_dict[name]` is a **flat 2-D tensor** of shape `(K, input_size)` where `K` is the total number of measurements of that modality across the entire batch — it is **not** batched by sample.

**Invariant**: row `k` of `bio_x_dict[name]` corresponds to the `k`-th `True` entry of `(bio_M == reader.biomarker2idx[name])` in **row-major order** (iterating over batch dim first, then position dim).

This arises from `collate`:
```python
for x0, t0, bio_x_dict, bio_t, bio_m, x1, t1 in samples:
    for modality in bio_x_dict.keys():
        bio_X_dict[modality].extend(bio_x_dict[modality])  # appends per-sample rows
```

The model recovers per-position embeddings via:
```python
mod_mask = mod_idx == idx               # (B, L_bio) boolean
biomarker_emb[name] += mod_age_emb[mod_mask]   # selects K positions
```

**Consequence**: any operation that changes which positions in `bio_M` carry a given modality value — masking entries to 0, removing positions — **silently breaks** `bio_x_dict` unless you reindex it to match. The row count in `bio_x_dict[name]` must always equal `(bio_M == reader.biomarker2idx[name]).sum()`.

#### Reindexing pattern

After modifying `bio_M` (e.g. masking entries to 0), reindex `bio_x_dict` as follows:

```python
for name, bio_x in bio_x_dict.items():
    old_mask = bio_m_orig == biomarker2idx[name]   # (B, L_bio) — before modification
    new_mask = bio_m_new  == biomarker2idx[name]   # (B, L_bio) — after modification
    keep = new_mask[old_mask]                       # 1-D bool over original rows
    if keep.any():
        new_bio_x_dict[name] = bio_x[keep]
    # if not keep.any(): omit this modality from the dict entirely
```

The helpers `filter_biomarker_array` / `dropout_biomarkers` in
`delphi/data/utils.py` already maintain this invariant — prefer them over hand-rolled masking.

### Filtering participants by modality

Filtering is a **reader** concern (classmethods); the kept pids are then passed to `MultimodalDataset`:

```python
R = multimodal_reader_cls()
pids = R.participants("all")
pids = R.filter_participants_with_modalities(pids, biomarkers=["wbc"], expansion_packs=["ops"])
```

`filter_participants_with_biomarkers` / `..._with_expansion_packs` keep pids
present in (any of) the named modalities; `..._with_modalities` chains both. They
route through the `biomarker_cls` / `expansion_pack_cls` hooks, reading each
modality's `participants(name)` from disk without building a full reader.

---

## Model Forward Pass — Multimodal (`DelphiM4`)

```python
out_dict, loss, att = model(X0, T0, bio_x_dict, bio_T, bio_M, X1, T1)
```

Argument names in the model signature: `(idx, age, biomarker, mod_age, mod_idx, targets, targets_age)`.

### Output shape depends on whether targets are passed

| Call | `out_dict["logits"]` shape | Notes |
|------|---------------------------|-------|
| With `targets` / `targets_age` | `(B, L, V)` | Logits filtered to token positions only (biomarker positions, `fused_mod_idx > 1`, excluded) |
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

When constructing a sub-batch to pass directly to the model, move the full batch to device first, then manipulate — cloning device tensors is cheaper than moving CPU tensors after manipulation.

---

## Modality (`delphi/multimodal.py`)

`Modality` is a **model-side** `Enum` (with `Modality.PRS.name.lower() -> "prs"`,
used for the model's per-modality `nn.ModuleDict` keys). It is **not** how the data
layer assigns `bio_M` ids — those come from `reader.biomarker2idx` (see
[Biomarker indexing](#biomarker-indexing--readerbiomarker2idx)). Because the
biomarker set churns, prefer the per-reader / per-checkpoint `biomarker2idx` over
the enum's fixed values.

---

## BiomarkerReader / Biomarker (`delphi/data/reader.py`, `ukb.py`, `aou.py`)

`BiomarkerReader` is an ABC for **continuous-valued measurements** of a single
modality (e.g. a blood test panel). The concrete subclass is named `Biomarker` in
both `ukb.py` and `aou.py` (storage adapters); all shared logic lives on the ABC.
It is distinct from `ExpansionPack` (discrete tokens) and is composed into a
`MultimodalReader` via `self.biomarkers[name]`.

### Canonical in-memory layout

Each `_load` normalizes its on-disk storage to a single layout:
- `data`: `(n_measurements, n_features)` float32, rows grouped by pid then ordered by time.
- `times`: `(n_measurements,)` float32, aligned row-for-row to `data`.
- `pid2idx` / `pid2cnt`: `pid -> first row index / number of rows` in `data`.

UKB stores a flat `data.bin` + `p2i.csv` (`pid, time, start_pos, seq_len`) and its
`_load` gathers the ragged-flat rows into the 2-D layout; AoU's `data.parquet` is
already 2-D. (No `memmap` — readers always load into memory.)

### Per-dataset seam (the only subclass code)

| Member | Role |
|--------|------|
| `base_dir`, `_marker` | class attrs: data dir + the file that marks a biomarker dir (`data.bin` / `data.parquet`) |
| `_load(name) -> (values, times, pids, features)` | read + normalize to the canonical layout |
| `_read_features(name) -> list[str]` | feature names, without building an instance |
| `_read_index(name) -> DataFrame[pid, time]` | the (pid, time) table, without the feature payload |

### Shared surface (on the ABC)

- `Biomarker(name)` — construct; the ABC `__init__` derives `pid2idx`/`pid2cnt` via `np.unique`.
- `__getitem__(pid) -> (list[np.ndarray] | None, np.ndarray | None)` — per-measurement feature vectors + times, or `(None, None)` if absent.
- `to_array(subjects) -> (len(subjects), n_features)` — first-occurrence vector per subject (NaN where absent); `stats(subjects) -> (mean, std)`.
- `features`, `n_features`, `feat2idx`.
- classmethods `catalog()`, `input_size(name)`, `participants(name)`, `first_occurrence_times(name, pids)`.

Normalization (`z_score`), first-visit selection (`first_time_only`) and `dropout`
are **not** on `Biomarker` — they live in `BiomarkerTransform` (see MultimodalDataset).

---

## ExpansionPackReader / ExpansionPack (`delphi/data/reader.py`, `ukb.py`, `aou.py`)

`ExpansionPackReader` is an ABC for **additional discrete EHR tokens** (e.g.
surgical operations via OPCS-4/OMOP) that are not in the base vocabulary. Unlike
biomarkers, pack tokens join the disease token timeline as ordinary discrete
events. The concrete subclass is named `ExpansionPack` in both `ukb.py` and
`aou.py`.

`ExpansionPackReader` **composes** a `TokenReader` (`self.reader`) rather than
subclassing it, and exposes only the slicing surface the composer reads:
`__getitem__(pid) -> (x, t)`, `start_pos`, `seq_len`, `tokenizer`, plus `pids`.

### Per-dataset seam

| Member | Role |
|--------|------|
| `base_dir` | data dir; `_marker = "tokenizer.yaml"` is shared on the ABC |
| `_load(name) -> (tokens, timesteps, start_pos, seq_len, tokenizer)` | read the pack |
| `participants(name)` / `first_occurrence_times(name, pids)` | abstract classmethods — UKB pack times live in `time.bin`, AoU's in the parquet, so they genuinely differ |

`catalog()` is shared on the ABC. UKB disk: `data.bin` + `time.bin` (uint32) +
`p2i.csv` (`pid, start_pos, seq_len`) + `tokenizer.yaml`; AoU: `data.parquet` +
`tokenizer.yaml`.

### Token offset + merge into the vocabulary

A pack's local token IDs must not collide with the base vocabulary, so
`MultimodalReader.__init__` assigns each pack an `offset` via `update_tokenizer`:

```python
for name in sorted(expansion_packs or []):
    pack = self.expansion_pack_cls(name=name)
    self.tokenizer, offset = update_tokenizer(self.tokenizer, pack.tokenizer)
    self.expansion_offset[name] = offset
```

In `MultimodalReader.__getitem__`, each pack's tokens are offset and concatenated
with the base stream before assembly:

```python
exp_x, exp_t = expansion_pack[pid]
x_lst.append(exp_x + self.expansion_offset[name])   # then concatenate + assemble the 5-tuple
```

The merged tokens participate in the same autoregressive loss as base tokens. The
shifted IDs are available via `reader.expansion_tokens`.
