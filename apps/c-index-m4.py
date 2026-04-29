# +
import json
import math
import os
import pprint
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
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
)
from delphi.experiment import CliConfig, eval_iter, load_ckpt, move_batch_to_device
from delphi.model.tpp import HomoPoissonTPP
from delphi.multimodal import Modality, parse_panel


def filter_participants(pids, biomarkers, expansion_packs):
    if biomarkers is not None:
        total = pids.size
        pids = filter_participants_with_biomarkers(
            pids, biomarkers=biomarkers, any=True
        )
        print(f"{pids.size} / {total} val pids (biomarker filter)")
    if expansion_packs is not None:
        total = pids.size
        pids = filter_participants_with_expansion_packs(
            pids, expansion_packs=expansion_packs, any=True
        )
        print(f"{pids.size} / {total} val pids (expansion pack filter)")

    return pids


def first_modality_timestep(pids, biomarkers, expansion_packs):

    cutoff = np.full(len(pids), np.nan, dtype=np.float32)
    for mod_name in biomarkers or []:
        first = Biomarker.first_occurrence_times(mod_name, pids)
        cutoff = np.fmin(cutoff, first)
    for pack_name in expansion_packs or []:
        first = ExpansionPack.first_occurrence_times(pack_name, pids)
        cutoff = np.fmin(cutoff, first)

    return cutoff


@dataclass
class TaskConfig(CliConfig):
    ckpt: str = "delphi-m4/delphi-m4/ckpt.pt"
    batch_size: int = 64
    chunk_size: int = 8192
    min_time_gap: float = 0
    panel: None | str = None
    biomarkers: None | list = None
    expansion_packs: None | list[str] = None
    max_gap: float = 5
    after_only: bool = True
    fname: None | str = None
    panel_name: None | str = None

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
ckpt = Path(DELPHI_CKPT_READ) / args.ckpt
model, ckpt_dict = load_ckpt(ckpt)
device = "cuda" if torch.cuda.is_available() else "cpu"

reader_args = ckpt_dict["reader_args"]
token_transform_args = ckpt_dict["token_transform_args"]
biomarker_transform_args = ckpt_dict.get("biomarker_transform_args")
biomarker_stats = ckpt_dict.get("biomarker_stats")

val_pids = MultimodalUKBReader.participants("val")
val_pids = filter_participants(val_pids, args.biomarkers, args.expansion_packs)

if args.after_only:
    cutoff = first_modality_timestep(val_pids, args.biomarkers, args.expansion_packs)
else:
    cutoff = None

# -
biomarkers = list(
    set(reader_args["biomarkers"] or []).intersection(set(args.biomarkers or []))
)
expansion_packs = list(
    set(reader_args["expansion_packs"] or []).intersection(
        set(args.expansion_packs or [])
    )
)
reader = MultimodalUKBReader(
    biomarkers=biomarkers, expansion_packs=expansion_packs, memmap=False
)
reader.describe()

token_transform = TokenTransform(**token_transform_args)
token_transform.describe()

if biomarker_transform_args and biomarkers:
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
        tpp = HomoPoissonTPP(
            logits=out_dict["logits"],
            tokens=out_dict["idx"],
            timesteps=out_dict["age"],
            terminate_except=torch.tensor(
                model.config.self_terminate_except, device=device
            ),
        )

        intensity, nearest_t0 = tpp.intensity(t1 - offset_days)

        dis_collator.step(tokens=x1, timesteps=nearest_t0, logits=intensity)
        onset_collator.step(tokens=x1.cpu(), timestep=t1.cpu())

dis_rates, nearest_t0 = dis_collator.finalize()  # (N, V)
is_female = torch.from_numpy(reader.is_female(val_pids))  # (N,)
onset_times, _ = onset_collator.finalize()
onset_times = torch.from_numpy(onset_times)  # (N, V)

if cutoff is not None:
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
        tpp = HomoPoissonTPP(
            logits=out_dict["logits"],
            tokens=out_dict["idx"],
            timesteps=out_dict["age"],
            terminate_except=torch.tensor(
                model.config.self_terminate_except, device=device
            ),
        )
        concordance_collator.step(tpp=tpp)

case_sex, case_tokens, total_pairs, concordant = concordance_collator.finalize()
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


ckpt_write = Path(str(ckpt).replace(DELPHI_CKPT_READ, DELPHI_CKPT_WRITE))
os.makedirs(ckpt_write.parent, exist_ok=True)
out_path = ckpt_write.parent / f"{args.fname}.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=4)
print(f"Saved to {out_path}")
