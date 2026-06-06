"""Age-stratified Mann-Whitney AUC over a fold (single-pass, "fast").

The discrimination sibling of ``apps/c-index-m4.py``: instead of concordance
over case/control time pairs, this scores per-disease AUC within age bins and
sex strata. One forward pass per participant evaluates the model's intensity at
the input position just before each target event (frozen history), then for
every disease token it asks: within an age bin and sex, can the model rank the
people who will develop the disease above those who will not?

Mirrors c-index-m4's data API: inherits the checkpoint's biomarker/expansion
set (overridable, clamped to the trained set via ``--panel``/``--biomarkers``),
builds a ``MultimodalDataset`` through the reader + transform stack, and routes
the model output through ``tpp.intensity`` so it is loss-type agnostic.

Output is the nested JSON logbook consumed by ``plot/compare_auc.py``:
``logbook[icd][sex][age_bin] = {"auc", "ctl_count", "dis_count"}``.
"""

# +
import json
import math
import pprint
from dataclasses import dataclass

import numpy as np
import torch
from cloudpathlib import AnyPath
from tqdm import tqdm

from delphi.data import MultimodalDataset
from delphi.data.auto import multimodal_reader_cls
from delphi.data.transform import BiomarkerTransform, TokenTransform
from delphi.env import DELPHI_CKPT_READ, DELPHI_CKPT_WRITE
from delphi.eval import (
    AgeStratRatesCollator,
    DiseaseRatesCollator,
    batched_mann_whitney_auc,
)
from delphi.experiment import CliConfig, eval_iter, load_ckpt, move_batch_to_device
from delphi.model.tpp import tpp_dispatch
from delphi.multimodal import parse_panel


@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    ckpt: str = "delphi-m4/delphi-m4/ckpt.pt"
    batch_size: int = 64
    min_time_gap: float = 0
    panel: None | str = None
    biomarkers: None | list = None
    expansion_packs: None | list[str] = None
    age_start: int = 40
    age_end: int = 85
    age_gap: int = 5
    fname: None | str = None
    panel_name: None | str = None
    fold: str = "val"

    def __post_init__(self):
        if self.panel:
            self.biomarkers, self.expansion_packs, self.panel_name = parse_panel(
                self.panel
            )
        if self.fname is None:
            self.fname = "auc"
            if self.panel_name is not None:
                self.fname += f"_{self.panel_name}"
            else:
                if self.biomarkers is not None:
                    self.fname += f"-{'-'.join(self.biomarkers)}"
                if self.expansion_packs is not None:
                    self.fname += f"-{'-'.join(self.expansion_packs)}"


args = TaskConfig.from_cli()
print("args:")
pprint.pp(args)


# +
ckpt = AnyPath(DELPHI_CKPT_READ) / args.ckpt
model, ckpt_dict = load_ckpt(ckpt)
device = "cuda" if torch.cuda.is_available() else "cpu"

reader_args = ckpt_dict["reader_args"]

ReaderCls = multimodal_reader_cls()
val_pids = ReaderCls.participants(args.fold)
# -

ckpt_biomarkers = list(reader_args["biomarkers"] or [])
ckpt_expansion_packs = list(reader_args["expansion_packs"] or [])

# Default: inherit the ckpt's training set. Overrides (panel / biomarkers /
# expansion_packs flags) are clamped to that set so typos or panels the ckpt
# doesn't know about don't silently sneak through.
if args.biomarkers is None:
    biomarkers = ckpt_biomarkers
else:
    biomarkers = sorted(set(ckpt_biomarkers).intersection(args.biomarkers))
    if not biomarkers:
        print(
            f"WARNING: biomarkers override {args.biomarkers} has no overlap "
            f"with ckpt biomarkers {ckpt_biomarkers}; using empty set"
        )

if args.expansion_packs is None:
    expansion_packs = ckpt_expansion_packs
else:
    expansion_packs = sorted(
        set(ckpt_expansion_packs).intersection(args.expansion_packs)
    )
    if not expansion_packs:
        print(
            f"WARNING: expansion_packs override {args.expansion_packs} has no "
            f"overlap with ckpt expansion_packs {ckpt_expansion_packs}; "
            "using empty set"
        )

print(f"biomarkers: {biomarkers}")
print(f"expansion_packs: {expansion_packs}")
# pass dict (not list) so reader uses the checkpoint's index assignments
# instead of re-deriving them from sorted order
biomarker2idx = {name: model.config.biomarker2idx[name] for name in biomarkers}
reader = ReaderCls(biomarkers=biomarker2idx, expansion_packs=expansion_packs)
reader.describe()

token_transform = TokenTransform.from_ckpt(ckpt_dict)
token_transform.describe()

biomarker_transform = BiomarkerTransform.from_ckpt(ckpt_dict) if biomarkers else None
if biomarker_transform is not None:
    biomarker_transform = biomarker_transform.replace(dropout=None)
    biomarker_transform.describe()

ds = MultimodalDataset(
    reader=reader,
    pids=val_pids,
    token_transform=token_transform,
    biomarker_transform=biomarker_transform,
)

# Longest-first packing minimizes padding and surfaces any OOM on the first
# batch. The reorder is in place and returns the new order; rebind val_pids so
# the per-participant is_female array stays aligned to the rate-matrix rows.
val_pids = ds.sort_by_length(descending=True)

# +
offset_days = args.min_time_gap * 365.25
model_targets = model.targets.to(device)
model_targets = model_targets[model_targets != 1]

# Age-bin edges in days; AgeStratRatesCollator makes len(edges) - 1 bins.
age_group_edges = (
    np.arange(args.age_start, args.age_end + args.age_gap, args.age_gap) * 365.25
)
n_bins = len(age_group_edges) - 1

ctl_collator = AgeStratRatesCollator(
    age_groups=torch.from_numpy(age_group_edges).float().to(device)
)
dis_collator = DiseaseRatesCollator(targets=model_targets)

it = tqdm(
    eval_iter(total_size=len(ds), batch_size=args.batch_size),
    total=math.ceil(len(ds) / args.batch_size),
    leave=False,
)
with torch.no_grad():
    for batch_idx in it:
        batch_input = ds.get_batch(batch_idx)
        batch_input = move_batch_to_device(batch_input, device=device)
        x0, t0, _, _, _, x1, t1 = batch_input

        out_dict, _, _ = model(*batch_input[:5])
        tpp = tpp_dispatch(model, out_dict)

        # Intensity at the input position strictly before each target's
        # (t1 - offset); nearest_t0 is that position's age. Same alignment the
        # old correct_time_offset produced, but exp'd, extinguishment-masked and
        # loss-type agnostic. exp() is monotonic so rank-based AUC is unchanged.
        intensity, nearest_t0 = tpp.intensity(t1 - offset_days)
        intensity = intensity.half()

        ctl_collator.step(timesteps=nearest_t0, logits=intensity)
        dis_collator.step(tokens=x1, timesteps=nearest_t0, logits=intensity)

ctl_rates, _ = ctl_collator.finalize()  # (N, n_bins, V)
dis_rates, dis_times = dis_collator.finalize()  # (N, V), (N, V)

ctl_rates = ctl_rates.numpy()
dis_rates = dis_rates.numpy()
dis_times = dis_times.numpy()
is_female = reader.is_female(val_pids)  # (N,) bool
# -

# +
# Bin each case by the age of its prediction position (dis_times == nearest_t0),
# matching the control rates' binning. Non-case entries are NaN -> bin out of
# range, but they are excluded by is_case below regardless.
dis_time_bin = np.searchsorted(age_group_edges, dis_times, side="right") - 1  # (N, V)
is_case = ~np.isnan(dis_rates)  # (N, V): participant developed this token

# For a fixed (sex, age bin) the AUC for every disease is one column-wise pass:
#   controls = same-sex participants who never develop the token, scored by
#     their control rate in this bin;
#   cases    = same-sex participants whose onset falls in this bin, scored by
#     their pre-onset rate.
# ctl_valid and case_valid are disjoint per token (case_valid implies is_case,
# ctl_valid implies ~is_case), so the merged score matrix is unambiguous.
results = {}  # (sex_label, bin_idx) -> (ctl_counts, case_counts, aucs), each (V,)
for sex_label, is_g in [("female", is_female), ("male", ~is_female)]:
    for i in range(n_bins):
        ctl_score = ctl_rates[:, i, :]  # (N, V)
        ctl_valid = (~is_case) & ~np.isnan(ctl_score) & is_g[:, None]
        case_valid = is_case & (dis_time_bin == i) & is_g[:, None]
        scores = np.where(case_valid, dis_rates, np.where(ctl_valid, ctl_score, np.nan))
        results[(sex_label, i)] = batched_mann_whitney_auc(
            scores, ctl=ctl_valid, case=case_valid
        )
# -

# +
age_group_keys = [
    f"{int(start / 365.25)}-{int(end / 365.25)}"
    for start, end in zip(age_group_edges[:-1], age_group_edges[1:])
]

logbook = {}
for d in model_targets.cpu().numpy().tolist():
    icd = reader.detokenizer.get(int(d), str(d))
    logbook[icd] = {"female": {}, "male": {}}
    for sex_label in ("female", "male"):
        for i, age_grp in enumerate(age_group_keys):
            ctl_counts, case_counts, aucs = results[(sex_label, i)]
            auc = aucs[d]
            logbook[icd][sex_label][age_grp] = {
                "auc": round(float(auc), 4) if not np.isnan(auc) else None,
                "ctl_count": int(ctl_counts[d]),
                "dis_count": int(case_counts[d]),
            }

pprint.pp(logbook.get("death", logbook.get(next(iter(logbook)), {})))
# -

ckpt_write = AnyPath(str(ckpt).replace(DELPHI_CKPT_READ, DELPHI_CKPT_WRITE))
ckpt_write.parent.mkdir(parents=True, exist_ok=True)
out_path = ckpt_write.parent / f"{args.fname}.json"
with out_path.open("w") as f:
    json.dump(logbook, f, indent=4)
print(f"Saved to {out_path}")
