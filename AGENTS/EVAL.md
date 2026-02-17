## Checkpoint Structure

Each `ckpt.pt` file contains:

| Key | Type | Purpose |
|-----|------|---------|
| `model` | `dict` | Model `state_dict` |
| `model_type` | `str` | Identifies model class (e.g., `"delphi-2m"`, `"delphi-m4"`) |
| `model_args` | `dict` | Arguments to instantiate model config dataclass |
| `tokenizer` | `dict[str, int]` | Event name → token index mapping |
| `data_args` | `dict` | Arguments used to instantiate `UKBDataset` during training |
| `config` | `dict` | Full experiment config (for debugging only) |
| `optimizer` | `dict` | Training only |
| `scheduler` | `dict` | Training only |
| `iter_num` | `int` | Training only |
| `best_val_loss` | `float` | Training only |

---

## Loading a Checkpoint

```python
from pathlib import Path
from delphi.experiment import load_ckpt
from delphi.env import DELPHI_CKPT_DIR

model, ckpt_dict = load_ckpt(Path(DELPHI_CKPT_DIR) / "path/to/ckpt.pt")
```

This returns:
- `model`: Loaded model in eval mode, on GPU if available
- `ckpt_dict`: Full checkpoint dict (access `tokenizer`, `data_args`, etc.)

**Reverse tokenizer lookup** (if needed):
```python
idx_to_event = {v: k for k, v in ckpt_dict["tokenizer"].items()}
```

---

## Argument Parsing
Both approaches support overriding defaults in interactive environments (Jupyter), which is essential for prototyping and debugging before running from CLI.

### Approach 1: argparse (use for prototyping / new evals)

```python
import argparse
import pprint
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="path/to/ckpt.pt")
# add task-specific arguments...

if "ipykernel" in sys.modules:
    print("running in jupyter notebook")
    args = parser.parse_args([])
    args.ckpt = "debug/ckpt.pt"  # override for interactive use
else:
    args = parser.parse_args()

print("args:")
pprint.pp(vars(args))
```

### Approach 2: Dataclass + OmegaConf (for mature evals)

```python
from dataclasses import dataclass
from omegaconf import OmegaConf
import sys

@dataclass
class MyEvalConfig:
    ckpt: str = "path/to/ckpt.pt"
    # add task-specific arguments...

    @classmethod
    def auto(cls, **overrides):
        if "ipykernel" in sys.modules or "IPython" in sys.modules:
            print("Running in interactive environment")
            schema = OmegaConf.structured(cls)
            override_conf = OmegaConf.create(overrides)
            merged = OmegaConf.merge(schema, override_conf)
            return OmegaConf.to_object(merged)
        else:
            schema = OmegaConf.structured(cls)
            cli = OmegaConf.from_cli()
            merged = OmegaConf.merge(schema, cli)
            return OmegaConf.to_object(merged)

args = MyEvalConfig.auto(ckpt="debug/ckpt.pt")
```

---

## Evaluation Script Pattern

```python
# 1. Parse config
args = ...

# 2. Load model
model, ckpt_dict = load_ckpt(Path(DELPHI_CKPT_DIR) / args.ckpt)

# 3. Prepare dataset (modify data_args as needed)
data_args = ckpt_dict["data_args"]
data_args["subject_list"] = "participants/val_fold.bin"  # use val set
data_args["perturb"] = False
data_args["deterministic"] = True
# ... other modifications as needed
ds = UKBDataset(**data_args)

# 4. Instantiate collators
collator_a = MyCollator(...)
collator_b = AnotherCollator(...)

# 5. Run evaluation loop
for batch_idx in eval_iter(...):
    batch_input = ds.get_batch(batch_idx)
    # ... forward pass, compute masks ...
    collator_a.step(...)
    collator_b.step(...)

# 6. Aggregate and output results
metrics = {}
metrics.update(collator_a.finalize())
metrics.update(collator_b.finalize())
```

### Collator Pattern

Eval scripts use **collators** to encapsulate batch-level accumulation logic. Each collator follows a simple `step()` / `finalize()` interface:

- `step(...)`: processes one batch, updates internal state
- `finalize()`: computes and returns final metrics (typically a dict)

This keeps the data loop clean and makes it easy to compose multiple independent analyses in the same pass. Each collator owns its own state and aggregation logic.

**Conventions:**
- Collators do not share a formal base class — follow the convention by convention
- For new tasks, define collator classes in the script itself (e.g., at the top of `apps/eval_nll.py`). Move to `delphi/eval.py` once the task matures.
- Existing mature collators live in `delphi/eval.py` (e.g., `AgeStratRatesCollator`, `DiseaseRatesCollator`, `SexCollator`)

## Modifying `data_args` for Evaluation

When loading a checkpoint, `data_args` reflects training settings. Some arguments **must** be modified for evaluation, some **must not** be changed, and others are context-dependent.

### Must Modify

| Argument | Eval Setting | Reason |
|----------|--------------|--------|
| `subject_list` | `"participants/val_fold.bin"` or `"participants/test_fold.bin"` | Evaluate on held-out data |
| `perturb` | `False` | Disable augmentation; use clean timestamps |
| `deterministic` | `True` | Ensure reproducible results |

### Must NOT Modify

| Argument | Reason |
|----------|--------|
| `data_dir` | Must match training data (same tokenizer, same binary files) |
| `no_event_interval` | Model trained expecting specific no-event density |
| `no_event_mode` | Model trained expecting specific no-event placement strategy |
| `exclude` / `exclude_list` | Changing token inclusion would create train/eval mismatch |
| `break_clusters` | Model expects dissolved (or not) clusters as trained |
| `additional_dx_token` | Affects vocab size; mismatch will cause errors |

### Context-Dependent

| Argument | Consideration |
|----------|---------------|
| `block_size` | Match training for perplexity. Set `None` for full-sequence tasks. |
| `crop_mode` | Use `"right"` for most-recent history. Some tasks may need `"left"` or full sequence. |
| `seed` | Keep consistent across eval runs for reproducibility; change for variance estimation. |
| `memmap` | No effect on results; use based on memory constraints. |

---

## Notes

- **Eval scripts** live in `apps/`
- **Model forward signatures** vary by model class. See model-specific documentation for input/output formats.
- **`model_type`** determines which config/model classes are used during loading (`"delphi-2m"` → `Delphi2MConfig`/`Delphi2M`, `"delphi-m4"` → `DelphiM4Config`/`DelphiM4`)

## Eval Task Documentation

| Task | Script | Docs |
|------|--------|------|
| NLL evaluation | `apps/eval_nll.py` | [EVAL_NLL.md](EVAL_NLL.md) |
| AUC evaluation | `apps/auc_fast.py` | [EVAL_AUC.md](EVAL_AUC.md) |

---
