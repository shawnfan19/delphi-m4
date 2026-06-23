"""Extract last-layer hidden states at a forecasting anchor, for downstream Cox.

Mirrors the ``hazards`` path of ``forecast-m4.py``: build a prompt truncated to
the anchor age, run one forward pass, and take the post-``ln_f`` hidden state at
the last prompt position (``outputs["h"][:, -1, :]``) as the representation "as
of" the anchor. Bundles the survival labels (per-target first-occurrence times,
exit time, sex) into the same ``.npz`` so ``delphi.eval.cox`` can fit/evaluate
without re-deriving the participant set or risking misalignment.

One split + one anchor per run. For multiple anchors, re-run with a different
``--prompt_age`` (each writes its own file).
"""

import math
import pprint
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from delphi.data import MultimodalDataset
from delphi.data.auto import multimodal_reader_cls
from delphi.data.transform import BiomarkerTransform, MultimodalPrompt, TokenTransform
from delphi.env import DELPHI_CKPT_READ, DELPHI_CKPT_WRITE
from delphi.experiment import GenerateConfig, eval_iter, load_ckpt, move_batch_to_device


@dataclass(kw_only=True)
class TaskConfig(GenerateConfig):
    fold: str = "val"  # train (fit Cox) / val / test (evaluate)
    prompt_age: int | str = "recruitment"  # fixed anchor age (years), or "recruitment"
    fname: None | str = None

    def __post_init__(self):
        assert self.fold in {"train", "val", "test"}
        if self.prompt_age != "recruitment":
            self.prompt_age = int(self.prompt_age)  # fixed anchor age in years
        if not self.fname:
            tag = (
                "recruitment"
                if self.prompt_age == "recruitment"
                else f"age{self.prompt_age}"
            )
            self.fname = f"embed_{self.fold}_{tag}"


args = TaskConfig.from_cli()
args.print()

ckpt = Path(DELPHI_CKPT_READ) / args.ckpt
model, ckpt_dict = load_ckpt(ckpt)
device = "cuda" if torch.cuda.is_available() else "cpu"
assert model.config.block_size is None


reader_args = ckpt_dict["reader_args"]
pprint.pp({"reader_args": reader_args})

# dataset-aware: UKB on the cluster, AoU on the workbench. Honors
# DELPHI_DATASET (set in the dsub env), else auto-detects from the data dir.
ReaderCls = multimodal_reader_cls()

# pass dict (not list) so reader uses the checkpoint's index assignments
reader = ReaderCls(
    biomarkers=model.config.biomarker2idx or None,
    expansion_packs=reader_args["expansion_packs"],
)

pids = ReaderCls.participants(args.fold)
pids, prompt_age = reader.resolve_prompt_age(pids, args.prompt_age)

token_transform = TokenTransform.from_ckpt(ckpt_dict)
token_transform.describe()
biomarker_transform = BiomarkerTransform.from_ckpt(ckpt_dict)
if biomarker_transform is not None:
    biomarker_transform = biomarker_transform.replace(dropout=None)
    biomarker_transform.describe()

prompt_transform = MultimodalPrompt(
    prompt_age={pid: age for pid, age in zip(pids, prompt_age.tolist())},
    biomarker2idx=reader.biomarker2idx,
    append_no_event=False,  # mirror forecast-m4 hazards: h at last event before anchor
)

ds = MultimodalDataset(
    reader=reader,
    pids=pids,
    token_transform=token_transform,
    biomarker_transform=biomarker_transform,
    prompt_transform=prompt_transform,
)


hidden = list()
it = eval_iter(total_size=len(ds), batch_size=args.batch_size)
pbar = tqdm(it, total=math.ceil(len(ds) / args.batch_size))
for batch_idx in pbar:

    pmt_idx, pmt_age, pmt_bio_x_dict, pmt_bio_t, pmt_bio_m, X1, T1 = (
        move_batch_to_device(ds.get_batch(batch_idx), device=device)
    )

    with torch.no_grad():
        outputs, _, _ = model(
            pmt_idx,
            pmt_age,
            biomarker=pmt_bio_x_dict,
            mod_age=pmt_bio_t,
            mod_idx=pmt_bio_m,
        )

    hidden.append(outputs["h"][:, -1, :].detach().cpu().to(torch.float32).numpy())

hidden = np.concatenate(hidden, axis=0)

# survival labels, aligned row-for-row with `hidden` (same `pids`, same order)
targets = model.targets.detach().cpu().numpy()
all_event_times = reader.event_times(pids=pids)  # (N, full vocab)
event_times = all_event_times[:, targets]  # (N, n_targets) aligned to target_tokens
died = ~np.isnan(all_event_times[:, 1269])  # 1269 = death token (matches forecast-m4)
exit_time = reader.exit_times(pids=pids)
is_female = reader.is_female(pids=pids)
target_names = np.array([reader.detokenizer[int(t)] for t in targets])

path = Path(DELPHI_CKPT_WRITE) / Path(args.ckpt).parent / f"{args.fname}.npz"
path.parent.mkdir(parents=True, exist_ok=True)  # don't lose a GPU run to a missing dir
np.savez(
    path,
    h=hidden,  # (N, n_embd) float32 — last-layer hidden state at the anchor
    pids=pids,  # (N,)
    prompt_age=prompt_age,  # (N,) anchor age in days (left-truncation entry time)
    target_tokens=targets,  # (n_targets,) disease token ids -> columns of event_times
    target_names=target_names,  # (n_targets,) disease names (downstream logbook keys)
    event_times=event_times,  # (N, n_targets) first-occurrence age in days, NaN if none
    exit_time=exit_time,  # (N,) last-seen age in days (censoring time)
    is_female=is_female,  # (N,) bool
    died=died,  # (N,) bool — any death event (follow-up completeness)
)
print(f"wrote {hidden.shape} hidden states + labels -> {path}")
