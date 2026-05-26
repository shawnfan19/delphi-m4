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

from delphi.data import Dataset
from delphi.data.transform import TokenTransform
from delphi.data.ukb import UKBReader
from delphi.env import DELPHI_CKPT_READ, DELPHI_CKPT_WRITE
from delphi.eval import (
    ConcordanceCollator,
    DiseaseRatesCollator,
    EventTimeCollator,
)
from delphi.experiment import CliConfig, eval_iter, load_ckpt, move_batch_to_device
from delphi.model.tpp import tpp_dispatch


@dataclass
class TaskConfig(CliConfig):
    ckpt: str = "delphi-2m/baseline/ckpt.pt"
    batch_size: int = 64
    chunk_size: int = 8192
    min_time_gap: float = 0
    max_gap: float = 5
    fname: str = "cindex"


args = TaskConfig.from_cli()
print("args:")
pprint.pp(args)


# +
ckpt = Path(DELPHI_CKPT_READ) / args.ckpt
model, ckpt_dict = load_ckpt(ckpt)
device = "cuda" if torch.cuda.is_available() else "cpu"

token_transform_args = ckpt_dict["token_transform_args"]
val_pids = UKBReader.participants("val")
# -

reader = UKBReader()
token_transform = TokenTransform(**token_transform_args)
token_transform.describe()

ds = Dataset(
    reader=reader,
    pids=val_pids,
    token_transform=token_transform,
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
        x0, t0, x1, t1 = batch_input

        out_dict, _, _ = model(x0, t0)
        tpp = tpp_dispatch(model, out_dict)

        intensity, nearest_t0 = tpp.intensity(t1 - offset_days)

        dis_collator.step(tokens=x1, timesteps=nearest_t0, logits=intensity)
        onset_collator.step(tokens=x1.cpu(), timestep=t1.cpu())

dis_rates, nearest_t0 = dis_collator.finalize()  # (N, V)
is_female = torch.from_numpy(reader.is_female(val_pids))  # (N,)
onset_times, _ = onset_collator.finalize()
onset_times = torch.from_numpy(onset_times)  # (N, V)

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
    same_sex_only=False,
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
        x0, t0, _, _ = batch_input

        out_dict, _, _ = model(x0, t0)
        tpp = tpp_dispatch(model, out_dict)
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
