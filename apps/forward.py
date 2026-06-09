"""Forward pass over a fold, saving last-position intensities to disk.

Loads a pretrained checkpoint, rebuilds its dataset (always inheriting the
biomarkers and expansion packs the model was trained with), cuts every
participant's sequence at a prompt age, runs a single forward pass, and saves
the last prompt position's per-token intensities (``exp(logits)``) over the
full vocabulary.

The prompt cutoff (``--prompt_age``) is a numeric age in years, or
``"recruitment"`` (the default) for each participant's recruitment age. Batches
are left-padded, so ``logits[:, -1, :]`` is the genuine last prompt token and
no per-row gather is needed.

Output is a compressed ``.npz`` holding row-aligned arrays:
``participant_ids`` (N,), ``intensities`` (N, V), ``prompt_age`` (N, days), plus
``token_ids``/``token_names`` describing the vocab axis. Reload the pid->vector
mapping with ``dict(zip(d["participant_ids"], d["intensities"]))``.
"""

import math
import pprint
from dataclasses import dataclass

import numpy as np
import torch
from cloudpathlib import AnyPath
from tqdm import tqdm

from delphi.data import MultimodalDataset
from delphi.data.auto import multimodal_reader_cls
from delphi.data.transform import BiomarkerTransform, MultimodalPrompt, TokenTransform
from delphi.env import DELPHI_CKPT_READ, DELPHI_CKPT_WRITE
from delphi.experiment import GenerateConfig, eval_iter, load_ckpt, move_batch_to_device


@dataclass(kw_only=True)
class TaskConfig(GenerateConfig):
    fold: str = "val"
    fname: None | str = None

    def __post_init__(self):
        if not self.fname:
            self.fname = "forward"


args = TaskConfig.from_cli()
args.print()


# +
ckpt = AnyPath(DELPHI_CKPT_READ) / args.ckpt
model, ckpt_dict = load_ckpt(ckpt)
device = "cuda" if torch.cuda.is_available() else "cpu"

reader_args = ckpt_dict["reader_args"]
pprint.pp(
    {
        "reader_args": reader_args,
        "token_transform_args": ckpt_dict["token_transform_args"],
        "biomarker_transform_args": ckpt_dict.get("biomarker_transform_args"),
    }
)

ReaderCls = multimodal_reader_cls()
pids = ReaderCls.participants(args.fold)

# Always inherit the checkpoint's training set. Pass biomarker2idx as a dict (not
# a list) so the reader uses the checkpoint's index assignments instead of
# re-deriving them from sorted order.
reader = ReaderCls(
    biomarkers=model.config.biomarker2idx or None,
    expansion_packs=reader_args["expansion_packs"],
)
reader.describe()
# -

# +
# Resolve the prompt cutoff in days. A numeric --prompt_age is an age in years;
# None or "recruitment" cuts at each participant's recruitment age (dropping
# participants with no recorded recruitment time).
prompt_age_arg = args.prompt_age if args.prompt_age is not None else "recruitment"
if prompt_age_arg == "recruitment":
    rec = reader.recruitment_times(pids)
    has_rec = ~np.isnan(rec)
    pids = pids[has_rec]
    prompt_age = {int(p): float(a) for p, a in zip(pids, rec[has_rec])}
    print(f"{pids.size} participants with recruitment times")
else:
    prompt_age = float(prompt_age_arg) * 365.25
    print(f"prompt cutoff: {prompt_age_arg} years ({prompt_age:.0f} days)")

if args.subsample:
    pids = pids[: args.subsample]
    if isinstance(prompt_age, dict):
        prompt_age = {int(p): prompt_age[int(p)] for p in pids}
    print(f"subsampled to {pids.size} participants")
# -

# +
token_transform = TokenTransform.from_ckpt(ckpt_dict)
biomarker_transform = BiomarkerTransform.from_ckpt(ckpt_dict)
if biomarker_transform is not None:
    biomarker_transform = biomarker_transform.replace(dropout=None)
    biomarker_transform.describe()

prompt_transform = MultimodalPrompt(
    prompt_age=prompt_age,
    biomarker2idx=reader.biomarker2idx,
    append_no_event=args.prompt_no_event,
)

ds = MultimodalDataset(
    reader=reader,
    pids=pids,
    token_transform=token_transform,
    biomarker_transform=biomarker_transform,
    prompt_transform=prompt_transform,
)

# Longest-first packing minimizes padding and surfaces any OOM on the first
# batch. The reorder is in place and returns the new order; rebind pids so the
# saved rows stay aligned with the intensity matrix.
pids = ds.sort_by_length(descending=True)
# -

# +
intensities = []
it = tqdm(
    eval_iter(total_size=len(ds), batch_size=args.batch_size),
    total=math.ceil(len(ds) / args.batch_size),
)
with torch.no_grad():
    for batch_idx in it:
        pmt_idx, pmt_age, pmt_bio_x_dict, pmt_bio_t, pmt_bio_m, X1, T1 = (
            move_batch_to_device(ds.get_batch(batch_idx), device=device)
        )
        outputs, _, _ = model(
            pmt_idx,
            pmt_age,
            biomarker=pmt_bio_x_dict,
            mod_age=pmt_bio_t,
            mod_idx=pmt_bio_m,
        )
        # left-padded batches -> [:, -1] is the true last prompt token
        last = torch.exp(outputs["logits"][:, -1, :])
        intensities.append(last.cpu().numpy())

intensities = np.concatenate(intensities, axis=0)  # (N, V)
# -

# +
pids = np.asarray(pids, dtype=np.int64)
if isinstance(prompt_age, dict):
    prompt_age = np.array([prompt_age[int(p)] for p in pids], dtype=np.float32)
else:
    prompt_age = np.full(pids.size, prompt_age, dtype=np.float32)

token_ids = np.arange(intensities.shape[1], dtype=np.int64)
token_names = np.array([reader.detokenizer.get(int(t), str(t)) for t in token_ids])

ckpt_write = AnyPath(str(ckpt).replace(DELPHI_CKPT_READ, DELPHI_CKPT_WRITE))
ckpt_write.parent.mkdir(parents=True, exist_ok=True)
out_path = ckpt_write.parent / f"{args.fname}.npz"
with out_path.open("wb") as f:
    np.savez_compressed(
        f,
        participant_ids=pids,
        intensities=intensities,
        prompt_age=prompt_age,
        token_ids=token_ids,
        token_names=token_names,
    )
print(f"Saved {intensities.shape} intensities to {out_path}")
# -
