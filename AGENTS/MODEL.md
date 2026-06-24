# MODEL.md — DelphiM4 (model, losses, sampling)

> This doc covers the **stable** model surface — input contract, TPP loss theory,
> the fusion invariants, and the tie/sampling helpers. The model's loss machinery
> is actively evolving (a `tpp_dispatch` layer in `delphi/model/tpp.py`, plus
> experimental `neural_tpp` / `neural_ode` / `multitask` paths); for those
> internals read the code rather than relying on prose here.

## Overview

**DelphiM4** (`delphi/model/multimodal.py`, `model_type = "delphi-m4"`) is the only
model — a transformer temporal point process (TPP) over EHR `(token, age)`
sequences that fuses biomarker modalities with disease-event tokens early in the
forward pass and predicts **what** event happens next and **when**. Unimodal use is
just `DelphiM4(biomarkers={})`; legacy `delphi-2m` checkpoints upgrade-load to a
zero-biomarker `DelphiM4`.

- **`delphi/model/multimodal.py`** — `DelphiM4`, `DelphiM4Config`, the biomarker
  embedding/fusion, `loss`, `sample_next`.
- **`delphi/model/transformer.py`** — shared primitives (`AgeEncoding`, `Block`,
  `LayerNorm`) and the standalone `generate()`.
- **`delphi/model/tpp.py`** — the TPP likelihood layer (`tpp_dispatch`, the per-loss
  `log_likelihood`) and the neural intensity heads (`NeuralIntensity`,
  `NeuralODEIntensity`).
- **`delphi/model/utils.py`** — attention masks (`causal_attention_mask`,
  `incremental_attention_mask`) and the tie / cluster / self-termination /
  sampler helpers (below).

---

## Inputs & Semantics

| Tensor | Shape | Description |
|--------|-------|-------------|
| `idx` | `(B, L)` | Token indices (event types) per position |
| `age` | `(B, L)` | Timestamp (days since birth) of each event |
| `biomarker` | `dict[str, (N_mod, input_size)]` | Per-modality raw feature vectors, keyed by **lowercase modality name**; `N_mod` = total measurements of that modality across the batch |
| `mod_age` | `(B, n_bio)` | Timestamps of each biomarker slot; padding sentinel `-1e4` |
| `mod_idx` | `(B, n_bio)` | Modality id per biomarker slot, from `config.biomarker2idx` (`0` = padding) |
| `targets` | `(B, L)` | Next event type |
| `targets_age` | `(B, L)` | Next event timestamp |

### Special tokens

| Token ID | Meaning |
|----------|---------|
| `0` | Padding — excluded from loss |
| `1` | No-event — synthetic token to advance time without an event (piecewise-constant intensity) |
| `2–12` | Reserved (sex / lifestyle); excluded as targets via `config.ignore_tokens` |

### Time

- Timestamps are **age in days since birth**; `delta_t = targets_age - age` is the time-to-next-event.
- `config.time_unit` (default `365.25`) normalises time inside the losses.

### Modality ids

`bio_M` / `mod_idx` integers come from `config.biomarker2idx` (a per-checkpoint
`name -> int` map, `0`/`1` reserved, biomarkers from `2`). This is saved with the
checkpoint so eval reconstructs the same mapping. The `Modality` enum in
`delphi/multimodal.py` is a separate model-side registry, **not** the source of
these ids (the biomarker set churns — prefer `biomarker2idx`).

---

## Temporal Point Process Theory

A TPP models events at times $t_1 < t_2 < \cdots$ via the conditional intensity
$\lambda(t \mid \mathcal{H}_t)$:

$$\lambda(t)\,dt = P(\text{event in } [t, t+dt) \mid \mathcal{H}_t)$$

**Log-likelihood** of event type $k$ at $t_{i+1}$:

$$\ell = \underbrace{\log \lambda_k(t_{i+1})}_{\text{event}} - \underbrace{\int_{t_i}^{t_{i+1}} \textstyle\sum_v \lambda_v(\tau)\,d\tau}_{\text{compensator (survival)}}$$

The compensator penalises high intensity over event-free intervals.

**Homogeneous Poisson** (constant intensity per type) is the default model:
$\lambda_v(t) = \lambda_v$, giving $-\ell = -\log\lambda_k + \Delta t \sum_v \lambda_v$.
It decomposes cleanly into a **cross-entropy** over the next event type and an
**exponential NLL** over the time-to-event.

**No-event tokens**: constant intensity over long gaps is unrealistic (risk at age
60 ≠ age 40), so synthetic no-event tokens (id `1`) are inserted to break long
intervals into segments — turning the model into a *piecewise-constant* intensity
process that re-evaluates intensity at each no-event token. (Insertion is a data-
layer transform; see `AGENTS/DATA.md`.)

---

## Losses

`DelphiM4.loss(outputs, targets, targets_age)` builds an NLL via
`tpp_dispatch(self, outputs)` (`delphi/model/tpp.py`) → `tpp.log_likelihood(...)`,
masks out `targets == 0` and `config.ignore_tokens`, and reduces with `nanmean`.
The active variant is set by **`config.loss`**:

| `config.loss` | Head | Notes |
|---------------|------|-------|
| `homo_poisson` *(default)* | `lm_head` (log-intensities) | cross-entropy + exponential-NLL decomposition; weights `ce_beta` / `dt_beta` |
| `neural_tpp` | `NeuralIntensity` | neural intensity; **experimental** |
| `neural_ode` | `NeuralODEIntensity` | ODE intensity; **experimental** |
| `dynamic_dpp` | `DPPSetHead` | set-valued marked TPP (Chang et al., AISTATS 2024): scalar ground intensity λ* + a history-dependent **DPP** over co-occurring marks (same-age clusters via `multi_hot`); mark term `log det L_S − log det(L+I)`. **experimental**. NLL is **per-set**, not per-token — not directly comparable to the other losses. Train + NLL only; no sampler yet. See `delphi/test/test_dynamic_dpp.py`. |

A `multitask` flag adds a biomarker-reconstruction decoder (+ optional EMA encoder)
whose loss is appended (`mse_beta`, `multitask_beta`). The non-`homo_poisson` paths
and the multitask machinery are evolving — read `multimodal.py` / `tpp.py` for their
current form. (`hawkes` and the old standalone `neural_tpp` head described in
earlier revisions of this doc were removed.)

A standalone NLL helper lives in `delphi/model/utils.py`:
`nll_homogeneous_cluster_poisson`.

---

## Tie & Sampling Helpers (`delphi/model/utils.py`)

### `multi_hot` — cluster encoding

`multi_hot(targets, targets_age, vocab_size) -> (hot_targets (B,L,V), cooccur (B,L))`
groups co-occurring events into multi-hot clusters; `cooccur` marks
cluster-continuation positions.

### `self_terminate_single` / `self_terminate` — no-repeat masking

Suppress already-seen tokens (set logits to `-inf`), except `terminate_except` ids
(e.g. `[1]` for no-event):
- `self_terminate_single(idx, logits (B,V), terminate_except)` — last-position
  logits; called inside `sample_next` during generation.
- `self_terminate(idx, logits (B,L,V), terminate_except)` — causal all-position
  variant; used in post-generation analysis.

### `sample_competing_exponentials(logits) -> (next_token (B,1), time_til_next (B,1))`

Inverse-CDF sampling for competing exponentials (each event type races; earliest
wins). Used by the `homo_poisson` sampler.

---

## Multimodal Fusion (stable mechanics)

The defining design of DelphiM4 — how biomarkers and tokens become one sequence:

1. **Embed.** Tokens: `x = token_drop(wte(idx)) * (1 - token_dropout) + wae(age)`
   (inline in `forward`). Biomarkers: `mod_emb = self.bio_embed(biomarker)` where
   `bio_embed` is a `BiomarkerEmbeddingDict` — one `BiomarkerEmbedding`
   (linear or MLP, per `BiomarkerEmbedConfig`) per modality name, projecting
   `(N_mod, input_size) -> (N_mod, n_embd)`; plus `wae(mod_age)` and a learned
   `mod_embedding(mod_idx)` when `config.modality_emb`.
2. **Fuse.** `fuse_embed(...)` (`multimodal.py`) scatters the sparse per-modality
   embeddings into dense slots, concatenates **biomarkers then tokens**, and sorts
   the merged sequence by timestamp with `argsort(stable=True)`. Returns the fused
   embedding/age/`mod_idx`/`idx` views; the pad mask is `fused_mod_idx > 0`.
3. **Attend.** `causal_attention_mask(pad, timestep=fused_age)` (time-causal; or
   `"triangular"`, or `incremental_attention_mask` with a KV cache). Transformer
   `Block`s support `past_kv` for incremental decoding.
4. **Read out + loss** via the TPP layer (above).

Two consequences of `stable=True` + biomarkers-first concat:

- **At equal timestamps, biomarkers precede tokens** — the model sees a biomarker
  value before predicting a same-day event.
- **Padding (`-1e4`) sorts to the front** — valid positions are right-aligned, so
  `logits[:, -1, :]` is always the prediction after the most recent real event.

---

## `DelphiM4Config` (`delphi/model/multimodal.py`)

Key fields (see the dataclass for the full, current list):

- **Transformer**: `block_size` (256), `vocab_size` (1270), `n_layer`/`n_head`/
  `n_embd`, `dropout`, `token_dropout`, `bias`, `weight_tying` (default `False`).
- **TPP / loss**: `loss` (`"homo_poisson"` default), `t_min`, `time_unit` (365.25),
  `mask_ties` (`True`), `attn_mask` (`"time"`), `ignore_tokens` (`[0, 2..12]`),
  `ce_beta`, `dt_beta`, `self_terminate_except` (`[1]`).
- **Multimodal**: `biomarkers: dict[str, BiomarkerEmbedConfig]`,
  `biomarker2idx: dict[str, int]` (saved with the checkpoint), `modality_emb`,
  `fuse` (`"early"`).
- **Experimental** (evolving — confirm against the code): `multitask`, `ema`,
  `n_integrate_grid`, `integrate_method`, `ode_method`, `ode_step_size`,
  `multitask_beta`, `mse_beta`, `spectral_norm`.

`BiomarkerEmbedConfig` is a `TypedDict`: `input_size` (int), `projector`
(`"linear"`/`"mlp"`), `n_layers`/`n_hidden` (for MLP), `bias`.

---

## Sampling / Generation

- `generate()` (`delphi/model/transformer.py`) autoregressively simulates
  trajectories from a trained model — see `AGENTS/GEN.md` for its invariants.
- `DelphiM4.sample_next(outputs, idx)` produces the next token(s) + time; it is
  responsible for **self-termination** (masking already-seen tokens via
  `self_terminate_single`, exempting `config.self_terminate_except`). `generate()`
  does not mask logits externally.

---

## Numerical Conventions

- **Log-space**: `log_prob = log_intensity - exp(log_intensity) * dt`, not
  `intensity * exp(-intensity*dt)`.
- **`logsumexp`** for summing intensities across event types.
- **Clamp `delta_t`** only when 0 is outside the distribution's support
  (e.g. exponential: `dt = clamp(targets_age - age, min=config.t_min)`).
- Type-hint all functions.

---

## File Locations

| What | Where |
|------|-------|
| Model + config | `delphi/model/multimodal.py` (`DelphiM4`, `DelphiM4Config`) |
| Shared transformer + `generate()` | `delphi/model/transformer.py` |
| TPP likelihood + neural heads (`tpp_dispatch`, `log_likelihood`, `NeuralIntensity`, `NeuralODEIntensity`) | `delphi/model/tpp.py` |
| Attention masks, tie / cluster / self-terminate / samplers / NLL helpers | `delphi/model/utils.py` |
| `Modality` enum | `delphi/multimodal.py` |
| Training | `apps/train-delphi-m4.py` |
