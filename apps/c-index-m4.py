# +
import math
import pprint
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from cloudpathlib import AnyPath
from tqdm import tqdm

from delphi.data import MultimodalDataset
from delphi.data.auto import multimodal_reader_cls
from delphi.data.transform import BiomarkerTransform, TokenTransform
from delphi.env import DELPHI_CKPT_READ, DELPHI_CKPT_WRITE
from delphi.eval import (
    ConcordanceCollator,
    DiseaseRatesCollator,
    EventTimeCollator,
)
from delphi.experiment import CliConfig, eval_iter, load_ckpt, move_batch_to_device
from delphi.model.tpp import tpp_dispatch
from delphi.multimodal import parse_panel


@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    ckpt: str = "delphi-m4/delphi-m4/ckpt.pt"
    batch_size: int = 64
    chunk_size: int = 8192
    min_time_gap: float = 0
    panel: None | str = None
    biomarkers: None | list = None
    expansion_packs: None | list[str] = None
    max_gap: float = 5
    fname: None | str = None
    panel_name: None | str = None
    fold: str = "val"
    same_sex: bool = True

    def __post_init__(self):
        if self.panel:
            self.biomarkers, self.expansion_packs, self.panel_name = parse_panel(
                self.panel
            )
        if self.fname is None:
            self.fname = "cindex"
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

# Token packing: order participants by sequence length, longest first, so each
# batch pads to a similar width and the forward pass wastes less compute on
# padding. Longest-first puts the peak-memory batch first, so an OOM from too
# large a batch_size surfaces immediately instead of mid-run (and the allocator
# reserves its high-water-mark pool up front). The method reorders the dataset in
# place and returns the new order; rebind val_pids so the downstream
# per-participant arrays (is_female, pids_np) stay aligned to the rows.
val_pids = ds.sort_by_length(descending=True)

# +
offset_days = args.min_time_gap * 365.25
model_targets = model.targets.to(device)
model_targets = model_targets[model_targets != 1]

dis_collator = DiseaseRatesCollator(targets=model_targets)
onset_collator = EventTimeCollator(vocab_size=int(model.config.vocab_size))

it = tqdm(
    eval_iter(total_size=len(ds), batch_size=args.batch_size),
    total=math.ceil(len(ds) / args.batch_size),
    desc="Phase 1",
    leave=False,
)
with torch.no_grad():
    for batch_idx in it:
        batch_input = ds.get_batch(batch_idx)
        batch_input = move_batch_to_device(batch_input, device=device)
        x0, t0, _, _, _, x1, t1 = batch_input

        out_dict, _, _ = model(*batch_input[:5])
        tpp = tpp_dispatch(model, out_dict)

        intensity, nearest_t0 = tpp.intensity(t1 - offset_days)

        dis_collator.step(tokens=x1, timesteps=nearest_t0, logits=intensity)
        onset_collator.step(tokens=x1.cpu(), timestep=t1.cpu())

dis_rates, _ = dis_collator.finalize()  # (N, V)
is_female = torch.from_numpy(reader.is_female(val_pids))  # (N,)
onset_times, _ = onset_collator.finalize()
onset_times = torch.from_numpy(onset_times)  # (N, V)

# Move tensors to device for Phase 2
dis_rates = dis_rates.to(device)
onset_times = onset_times.to(device)
is_female = is_female.to(device)

# +
concordance_collator = ConcordanceCollator(
    dis_rates=dis_rates,
    case_times=onset_times - offset_days,
    is_female=is_female,
    max_gap_days=args.max_gap * 365.25,
    chunk_size=args.chunk_size,
    same_sex_only=args.same_sex,
)

# dis_rates and onset_times are fully consumed above (the collator extracts its
# case events and keeps its own case_times_mat copy); free both (N, V) matrices
# and return Phase 1's reserved pool before the Phase 2 forward passes.
del dis_rates, onset_times
if device == "cuda":
    torch.cuda.empty_cache()

it2 = tqdm(
    eval_iter(total_size=len(ds), batch_size=args.batch_size),
    total=math.ceil(len(ds) / args.batch_size),
    desc="Phase 2",
    leave=False,
)
with torch.no_grad():
    for batch_idx in it2:
        batch_input = ds.get_batch(batch_idx)
        batch_input = move_batch_to_device(batch_input, device=device)
        out_dict, _, _ = model(*batch_input[:5])
        tpp = tpp_dispatch(model, out_dict)
        concordance_collator.step(tpp=tpp)

case_sex, case_tokens, total_pairs, concordant = concordance_collator.finalize()
case_times = concordance_collator.case_times.cpu().numpy()
case_participants = concordance_collator.case_participants.cpu().numpy()
# -

ckpt_write = AnyPath(str(ckpt).replace(DELPHI_CKPT_READ, DELPHI_CKPT_WRITE))
ckpt_write.parent.mkdir(parents=True, exist_ok=True)

pids_np = np.array(val_pids)
ts_df = pd.DataFrame(
    {
        "icd": pd.Categorical(
            [reader.detokenizer.get(int(d), str(d)) for d in case_tokens]
        ),
        "sex": pd.Categorical(np.where(case_sex, "female", "male")),
        "participant_id": pids_np[case_participants].astype(np.int64),
        "case_time": case_times.astype(np.float32),
        "concordant": concordant.astype(np.float32),
        "total_pairs": total_pairs.astype(np.int32),
    }
)
ts_path = ckpt_write.parent / f"{args.fname}_timeseries.parquet"
with ts_path.open("wb") as f:
    ts_df.to_parquet(f, engine="pyarrow", compression="snappy", index=False)
print(f"Saved time series to {ts_path}")
