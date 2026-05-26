# +
import json
import math
import pprint
from dataclasses import asdict, dataclass

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
    after_recruit: bool = False
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
            if self.after_recruit:
                self.fname += f"-recruit"


args = TaskConfig.from_cli()
print("args:")
pprint.pp(args)


# +
ckpt = AnyPath(DELPHI_CKPT_READ) / args.ckpt
model, ckpt_dict = load_ckpt(ckpt)
device = "cuda" if torch.cuda.is_available() else "cpu"

reader_args = ckpt_dict["reader_args"]

ReaderCls = multimodal_reader_cls()
if args.after_recruit and not hasattr(ReaderCls, "recruitment_times"):
    raise ValueError(
        "--after_recruit requires recruitment_times on the reader; "
        f"{ReaderCls.__name__} doesn't support it"
    )
val_pids = ReaderCls.participants(args.fold)

# -
biomarkers = list(
    set(reader_args["biomarkers"] or []).intersection(set(args.biomarkers or []))
)
expansion_packs = list(
    set(reader_args["expansion_packs"] or []).intersection(
        set(args.expansion_packs or [])
    )
)
# pass dict (not list) so reader uses the checkpoint's index assignments
# instead of re-deriving them from sorted order
biomarker2idx = {name: model.config.biomarker2idx[name] for name in biomarkers}
reader = ReaderCls(biomarkers=biomarker2idx, expansion_packs=expansion_packs)
reader.describe()

token_transform = TokenTransform.from_ckpt(ckpt_dict)
token_transform.describe()

biomarker_transform = BiomarkerTransform.from_ckpt(ckpt_dict) if biomarkers else None
if biomarker_transform is not None:
    biomarker_transform.describe()

ds = MultimodalDataset(
    reader=reader,
    pids=val_pids,
    token_transform=token_transform,
    biomarker_transform=biomarker_transform,
)

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

dis_rates, nearest_t0 = dis_collator.finalize()  # (N, V)
is_female = torch.from_numpy(reader.is_female(val_pids))  # (N,)
onset_times, _ = onset_collator.finalize()
onset_times = torch.from_numpy(onset_times)  # (N, V)

if args.after_recruit:
    cutoff = reader.recruitment_times(val_pids)
    reject = nearest_t0.numpy() < cutoff[:, None]
    reject = reject | np.isnan(cutoff[:, None])
    dis_rates[torch.from_numpy(reject)] = torch.nan
    cutoff = torch.from_numpy(cutoff).to(device)

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

# Aggregate C-index per disease per sex
result = {"config": asdict(args)}
for d_int in np.unique(case_tokens):
    d_mask = case_tokens == d_int
    icd = reader.detokenizer.get(int(d_int), str(d_int))
    result[icd] = {}
    for sex_label, sex_mask in [
        ("female", case_sex),
        ("male", ~case_sex),
    ]:
        mask = d_mask & sex_mask
        n_events = int(mask.sum())
        n_pairs = int(total_pairs[mask].sum())
        conc = concordant[mask].sum()
        c_index = round(float(conc / n_pairs), 4) if n_pairs > 0 else None
        result[icd][sex_label] = {
            "c_index": c_index,
            "n_events": n_events,
            "n_pairs": n_pairs,
        }


pprint.pp(result.get("death", result.get(next(iter(result)), {})))


ckpt_write = AnyPath(str(ckpt).replace(DELPHI_CKPT_READ, DELPHI_CKPT_WRITE))
ckpt_write.parent.mkdir(parents=True, exist_ok=True)
out_path = ckpt_write.parent / f"{args.fname}.json"
with out_path.open("w") as f:
    json.dump(result, f, indent=4)
print(f"Saved to {out_path}")

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
