# CLAUDE.md

> Single source of truth for agent guidance — edit this file. `AGENTS.md` is a
> one-line stub pointing here (Claude Code auto-loads `CLAUDE.md`; other tools
> follow the `AGENTS.md` pointer).

**Delphi** is a transformer temporal point process (TPP) model of disease natural
history from EHR sequences of `(token, age-in-days)` events — it predicts what
medical event happens next and when. Trained on UK Biobank (~500K patients); the
data layer also supports All of Us. The current model is multimodal (`DelphiM4`,
`delphi/model/multimodal.py`); legacy `delphi-2m` checkpoints upgrade-load to a
zero-biomarker `DelphiM4`.

## Where the detailed docs live (`AGENTS/`)

Per-subsystem documentation lives in `AGENTS/`. **Read the relevant doc before
changing that part of the codebase.**

**Start here**
- `AGENTS/DESIGN.md` — project scope, architecture overview, design philosophy.

**Running jobs**
- `AGENTS/LAUNCHPAD.md` — submitting training/eval/plot scripts to SLURM via the `submit` launcher (resources, sweeps, the `.env`/`slurm/` prerequisites, gotchas).

**Data** (`delphi/data/`, prep in `data/`)
- `AGENTS/DATA.md` — the data layer: on-disk formats (UKB flat `data.bin`/`time.bin`/`p2i.csv`, AoU `data.parquet`), the reader/dataset classes (`TokenReader` slicing engine; `MultimodalReader` ABC + per-env concretes; `BiomarkerReader`/`ExpansionPackReader` + same-name `Biomarker`/`ExpansionPack` concretes; `MultimodalDataset`), the `transform.py` transforms, the 7-tuple batch + `bio_x_dict` row-indexing invariant, and UKB↔AoU alignment.
- `AGENTS/UKB.md` — UKB raw-phenotype extraction → flat binary format (`data/ukb/`).
- `AGENTS/MIMIC.md` — MIMIC-IV preprocessing.

**Model & generation** (`delphi/model/`)
- `AGENTS/MODEL.md` — model architecture, loss functions, config dataclasses, checkpoint schema.
- `AGENTS/GEN.md` — `generate()` autoregressive trajectory simulation: invariants and design (`delphi/model/transformer.py`).

**Evaluation** (`apps/` eval scripts; utilities in `delphi/eval/`)
- `AGENTS/EVAL.md` — shared eval patterns: checkpoint structure/loading, dataset setup, argument parsing. Read first before touching any eval script.
- `AGENTS/EVAL_NLL.md` — negative log-likelihood (`apps/eval_nll.py`).
- `AGENTS/EVAL_CINDEX.md` — concordance index for disease-risk ranking.
- `AGENTS/EVAL_TPP.md` — TPP log-likelihood decomposed into time vs mark components (`delphi/model/tpp.py`).
- `AGENTS/EVAL_SALIENCY.md` — gradient saliency of predicted intensities w.r.t. biomarker features (`apps/saliency_biomarker.py`).

**Explainability** (`delphi/explain/`)
- `AGENTS/SHAP.md` — SHAP analysis pipeline (`delphi/explain/shap.py`, `plot/vis_shap.py`).

## Repo layout

- `delphi/` — core library: `model/`, `data/`, `eval/`, `explain/`, `multimodal.py` (`Modality` enum), `experiment.py` (config + checkpoints), `optim.py`, `env.py`.
- `apps/` — runnable scripts (training, eval, sampling, saliency/IG).
- `plot/` — figure / analysis plotting scripts.
- `data/` — data-prep pipelines (`ukb/`, `aou/`) and shared dictionaries (`ukb/dictionary/`: tokenizers, labels).
- `config/`, `reproducibility/` — experiment configs and reproduction recipes.

## Conventions

- Timestamps are always **age in days since birth** (`uint32`), never calendar dates.
- Runtime env vars: `DELPHI_DATA_DIR` (data root) and `DELPHI_CKPT_DIR` (checkpoints); the active dataset is auto-detected, or forced with `DELPHI_DATASET=ukb|aou`.
- `pre-commit` hooks (black, isort, codespell, …) gate commits — run `pre-commit run --files <changed>` first.
