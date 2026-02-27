# AGENTS.md

## Project Summary

**Delphi** is a transformer-based temporal point process (TPP) model that learns the natural history of human disease from electronic health records. It processes sequences of medical events — each a (token, timestamp) pair — and predicts what event happens next and when. The model is a modified GPT-2 architecture trained on UK Biobank data (~500K patients).

## Repository Layout

```
delphi/          # Core library
  model/         # Transformer architecture, loss functions, configs
  data/          # Dataset classes (UKBDataset), data loading, preprocessing
  eval.py        # Evaluation utilities (AUC, NLL, calibration)
  shap.py        # SHAP explainability infrastructure
  optim.py       # Optimizer and scheduler setup
  experiment.py  # Experiment config and checkpoint management
apps/            # Runnable scripts (training, eval, sampling, SHAP)
data/            # Data preparation scripts and UKB processing
configs/         # Experiment configuration files
```

## Component Documentation

Detailed technical documentation for each subsystem lives in `AGENTS/`. Read the relevant file before modifying that part of the codebase.

| File | Covers | Read before touching |
|------|--------|----------------------|
| [AGENTS/DESIGN.md](AGENTS/DESIGN.md) | Architecture overview, design philosophy, project scope | Any structural or cross-cutting change |
| [AGENTS/DATA.md](AGENTS/DATA.md) | Binary storage format (`data.bin`, `time.bin`, `p2i.csv`), index structure, participant splits | `delphi/data/`, data loading, preprocessing |
| [AGENTS/MODEL.md](AGENTS/MODEL.md) | Model architecture, loss functions, config dataclasses, checkpoint format | `delphi/model/`, training scripts, loss functions |
| [AGENTS/EVAL.md](AGENTS/EVAL.md) | General eval script patterns: checkpoint loading, dataset setup, argument parsing | Any `apps/` eval script |
| [AGENTS/EVAL_AUC.md](AGENTS/EVAL_AUC.md) | AUC evaluation task — risk ranking via Mann-Whitney U | `apps/auc*.py` |
| [AGENTS/EVAL_NLL.md](AGENTS/EVAL_NLL.md) | NLL evaluation task — negative log-likelihood on validation set | `apps/eval_nll.py` |
| [AGENTS/SHAP.md](AGENTS/SHAP.md) | SHAP analysis pipeline, `ShapArray` adapter, explainer integration | `delphi/shap.py`, `apps/run_shap*.py`, `apps/vis_shap.py` |
| [AGENTS/UKB.md](AGENTS/UKB.md) | `UKBDataset` constructor arguments, preprocessing toggles, data transforms | `delphi/data/ukb.py` |
| [AGENTS/DATA_MULTIMODAL.md](AGENTS/DATA_MULTIMODAL.md) | `MultimodalUKBDataset` batch format, `bio_x_dict` row-indexing invariant, fused sequence sort semantics | Any eval script using multimodal batches, `delphi/data/ukb.py` multimodal paths |
| [AGENTS/EVAL_TPP.md](AGENTS/EVAL_TPP.md) | TPP log-likelihood decomposition into time and mark components, `mask_ties` comparability requirement | `apps/eval_tpp.py` |
| [AGENTS/GEN.md](AGENTS/GEN.md) | `generate()` design — active batch management, temporal sorting invariant, self-termination, multi-token sampling | `delphi/model/transformer.py` `generate()`, any script that calls `generate()` |

## Key Conventions

- **Environment variables**: `DELPHI_DATA_DIR` (training data), `DELPHI_CKPT_DIR` (checkpoints). These are required at runtime.
- **Model variants**: `delphi-2m` (unimodal ICD codes) and `delphi-m4` (multimodal). Check `model_type` in checkpoint to know which you're working with.
- **Timestamps**: Always age in days since birth (`uint32`). Never calendar dates.
- **No-event tokens**: Synthetic tokens inserted between real events to model the absence of events. Controlled by `no_event_interval` and `no_event_mode` in the dataset config.
- **Checkpoints**: Saved as `ckpt.pt` files containing model weights, tokenizer, data args, and model args. See [AGENTS/MODEL.md](AGENTS/MODEL.md) for the full schema.

## Workflow Notes

- Training scripts live in `apps/train-delphi-*.py`.
- Eval scripts follow a shared pattern documented in [AGENTS/EVAL.md](AGENTS/EVAL.md) — read that first before modifying any eval script.
- The codebase uses pre-commit hooks for code quality. Run `pre-commit run --all-files` before committing.
