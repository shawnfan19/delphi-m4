# +
import json
import math
import os
import pprint
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm import tqdm

from delphi.data import MultimodalDataset
from delphi.data.transform import BiomarkerTransform, TokenTransform
from delphi.data.ukb import (
    Biomarker,
    ExpansionPack,
    MultimodalUKBReader,
    filter_participants_with_biomarkers,
    filter_participants_with_expansion_packs,
)
from delphi.env import DELPHI_CKPT_READ, DELPHI_CKPT_WRITE
from delphi.eval import (
    ConcordanceCollator,
    DiseaseRatesCollator,
    EventTimeCollator,
    correct_time_offset,
)
from delphi.experiment import CliConfig, eval_iter, load_ckpt, move_batch_to_device
from delphi.multimodal import Modality


def parse_biomarkers(biomarkers):
    if biomarkers is None:
        return None, None
    if isinstance(biomarkers, str):
        if biomarkers.endswith(".yaml"):
            path = Path(biomarkers)
            with open(path) as f:
                return yaml.safe_load(f), path.stem
        return [biomarkers], None
    return list(biomarkers), None


@dataclass
class TaskConfig(CliConfig):
    ckpt: str = "delphi-m4/delphi-m4/ckpt.pt"
    batch_size: int = 64
    min_time_gap: float = 0
    biomarkers: Any = None
    expansion_packs: None | list[str] = None
    max_gap: float = 5
    after_only: bool = False
    fname: None | str = None
    panel_name: None | str = None

    def __post_init__(self):
        self.biomarkers, self.panel_name = parse_biomarkers(self.biomarkers)
        if self.fname is None:
            self.fname = "cindex"
            if self.panel_name is not None:
                self.fname += f"_{self.panel_name}"
            elif self.biomarkers is not None:
                self.fname += f"-{'-'.join(self.biomarkers)}"
            if self.expansion_packs is not None:
                self.fname += f"-{'-'.join(self.expansion_packs)}"


args = TaskConfig.from_cli()
print("args:")
pprint.pp(args)


# +
ckpt = Path(DELPHI_CKPT_READ) / args.ckpt
model, ckpt_dict = load_ckpt(ckpt)
device = "cuda" if torch.cuda.is_available() else "cpu"

reader_args = ckpt_dict["reader_args"]
token_transform_args = ckpt_dict["token_transform_args"]
biomarker_transform_args = ckpt_dict.get("biomarker_transform_args")
biomarker_stats = ckpt_dict.get("biomarker_stats")

val_pids = MultimodalUKBReader.participants("val")
if args.biomarkers is not None:
    total_val = val_pids.size
    val_pids = filter_participants_with_biomarkers(
        val_pids, biomarkers=args.biomarkers, any=True
    )
    print(f"{val_pids.size} / {total_val} val pids (biomarker filter)")
if args.expansion_packs is not None:
    total_val = val_pids.size
    val_pids = filter_participants_with_expansion_packs(
        val_pids, expansion_packs=args.expansion_packs, any=True
    )
    print(f"{val_pids.size} / {total_val} val pids (expansion pack filter)")

pprint.pp(
    {
        "reader_args": reader_args,
        "token_transform_args": token_transform_args,
    }
)
# -

reader = MultimodalUKBReader(**reader_args)
token_transform = TokenTransform(**token_transform_args)
if biomarker_transform_args is not None:
    mean = biomarker_stats["mean"] if biomarker_stats else None
    std = biomarker_stats["std"] if biomarker_stats else None
    if mean is not None:
        mean = {Modality[k.upper()]: v for k, v in mean.items()}
    if std is not None:
        std = {Modality[k.upper()]: v for k, v in std.items()}
    biomarker_transform = BiomarkerTransform(
        **biomarker_transform_args, mean=mean, std=std
    )
    biomarker_transform.describe()
else:
    biomarker_transform = None

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
        logits = out_dict["logits"].half()
        t0 = out_dict["age"]

        t0_off, logits_off = correct_time_offset(t0, t1, logits, offset=offset_days)
        dis_collator.step(tokens=x1, timesteps=t0_off, logits=logits_off)
        onset_collator.step(tokens=x1.cpu(), timestep=t1.cpu())

dis_rates, dis_times = dis_collator.finalize()  # (N, V)
is_female = torch.from_numpy(reader.is_female(val_pids))  # (N,)
onset_times, _ = onset_collator.finalize()
onset_times = torch.from_numpy(onset_times)  # (N, V)

# Restrict to time points after first occurrence of any specified biomarker
# or expansion-pack token
if args.after_only and (args.biomarkers or args.expansion_packs):
    cutoff = np.full(len(ds), np.inf, dtype=np.float32)
    for mod_name in args.biomarkers or []:
        first = Biomarker.first_occurrence_times(mod_name, val_pids)
        cutoff = np.fmin(cutoff, first)
    for pack_name in args.expansion_packs or []:
        first = ExpansionPack.first_occurrence_times(pack_name, val_pids)
        cutoff = np.fmin(cutoff, first)
    # mask case events before the cutoff
    before_cutoff = dis_times.numpy() < cutoff[:, None]
    dis_rates[torch.from_numpy(before_cutoff)] = torch.nan
    cutoff = torch.from_numpy(cutoff).to(device)
else:
    cutoff = None

# Move tensors to device for Phase 2
dis_rates = dis_rates.to(device)
onset_times = onset_times.to(device)
is_female = is_female.to(device)

# +
concordance_collator = ConcordanceCollator(
    dis_rates=dis_rates,
    onset_times=onset_times,
    is_female=is_female,
    offset=offset_days,
    max_gap_days=args.max_gap * 365.25,
    cutoff=cutoff,
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
        scores = out_dict["logits"].half()
        age = out_dict["age"]
        concordance_collator.step(age=age, scores=scores)

case_sex, case_tokens, total_pairs, concordant = concordance_collator.finalize()
# -

# Aggregate C-index per disease per sex
result = {}
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


ckpt_write = Path(str(ckpt).replace(DELPHI_CKPT_READ, DELPHI_CKPT_WRITE))
os.makedirs(ckpt_write.parent, exist_ok=True)
out_path = ckpt_write.parent / f"{args.fname}.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=4)
print(f"Saved to {out_path}")
