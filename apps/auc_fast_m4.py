# +
import argparse
import json
import math
import os
import pprint
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm

from delphi import DAYS_PER_YEAR
from delphi.data.ukb import MultimodalUKBDataset
from delphi.env import DELPHI_CKPT_DIR
from delphi.eval import AgeStratRatesCollator as ControlRatesCollator
from delphi.eval import (
    DiseaseRatesCollator,
    SexCollator,
    correct_time_offset,
    mann_whitney_auc,
    sample_boolean_mask,
)
from delphi.experiment import eval_iter, move_batch_to_device
from delphi.model.multimodal import DelphiM4, DelphiM4Config

# -


# +
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="delphi-m4/delphi-m4/ckpt.pt")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--min_time_gap", type=float, default=0.01)
parser.add_argument("--age_start", type=int, default=40)
parser.add_argument("--age_end", type=int, default=85)
parser.add_argument("--age_gap", type=int, default=5)
parser.add_argument("--fname", type=str)

if "ipykernel" in sys.modules:
    print(f"running in jupyter notebook")
    args = parser.parse_args([])
    args.ckpt = "fusion/m4-lite/ckpt.pt"
    # args.fname = "test"
else:
    args = parser.parse_args()

print("args:")
pprint.pp(vars(args))


# -


def get_init_defaults(cls):
    """Get default __init__ arguments for a class as a dictionary."""
    import inspect

    sig = inspect.signature(cls.__init__)
    return {
        name: param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect.Parameter.empty and name != "self"
    }


# +
ckpt = Path(DELPHI_CKPT_DIR) / args.ckpt

ckpt_dict = torch.load(
    ckpt, map_location=torch.device("cpu") if not torch.cuda.is_available() else None
)
model_cfg = DelphiM4Config(**ckpt_dict["model_args"])
model = DelphiM4(model_cfg)
model.load_state_dict(ckpt_dict["model"])

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print(f"model: {ckpt} [iter: {ckpt_dict['iter_num']}]")

if "data_args" in ckpt_dict:
    data_args = ckpt_dict["data_args"].copy()
    data_args["subject_list"] = "participants/val_fold.bin"
    data_args["stats_subject_list"] = ckpt_dict["data_args"]["subject_list"]
else:
    data_args = get_init_defaults(MultimodalUKBDataset)
    data_args["subject_list"] = "participants/val_fold.bin"
    biomarkers = ckpt_dict["config"]["biomarkers"]
    if biomarkers is not None:
        biomarkers = list(biomarkers.keys())
    data_args["biomarkers"] = biomarkers
    expansion_packs = ckpt_dict["config"]["expansion_packs"]
    data_args["expansion_packs"] = expansion_packs
pprint.pp(data_args)
# -
ds = MultimodalUKBDataset(**data_args)

age_start = args.age_start
age_end = args.age_end
age_gap = args.age_gap
age_group_edges = np.arange(age_start, age_end + age_gap, age_gap) * 365.25
age_groups = [(i, j) for i, j in zip(age_group_edges[:-1], age_group_edges[1:])]

# +
model_targets = model.targets.to(device)
model_targets = model_targets[model_targets != 1]

ctl_collator = ControlRatesCollator(
    age_groups=torch.from_numpy(age_group_edges).to(device)
)
dis_collator = DiseaseRatesCollator(targets=model_targets)
sex_collator = SexCollator()

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

        out_dict, _, _ = model(*batch_input)
        logits = out_dict["logits"].half()

        t0, logits = correct_time_offset(
            t0, t1, logits, offset=args.min_time_gap * 365.25
        )
        ctl_collator.step(timesteps=t0, logits=logits)
        dis_collator.step(tokens=x1, timesteps=t0, logits=logits)
        sex_collator.step(tokens=x0)

ctl_rates, ctl_times = ctl_collator.finalize()
dis_rates, dis_times = dis_collator.finalize()
sex = sex_collator.finalize()
# +
ctl_rates, ctl_times = ctl_rates.numpy(), ctl_times.numpy()
dis_rates, dis_times = dis_rates.numpy(), dis_times.numpy()
sex = sex.numpy()

is_female = sex
is_male = ~sex
either = np.logical_or(is_female, is_male)


# +
dis_time_bin = np.searchsorted(age_group_edges, dis_times, side="right")
dis_time_bin -= 1

logbook = defaultdict(dict)

for i in tqdm(range(len(age_groups))):

    dis_in_range = dis_time_bin == i

    for dis_token in range(0, 1270):

        n_ctl = list()
        n_dis = list()
        aucs = list()

        for is_gender in [is_female, is_male, either]:

            ctl = ctl_rates[:, i, dis_token].copy()
            dis = dis_rates[:, dis_token].copy()
            ctl[~np.isnan(dis)] = np.nan
            dis[~dis_in_range[:, dis_token]] = np.nan
            ctl[~is_gender] = np.nan
            dis[~is_gender] = np.nan

            auc = mann_whitney_auc(ctl, dis)
            aucs.append(auc)
            n_ctl.append((~np.isnan(ctl)).sum())
            n_dis.append((~np.isnan(dis)).sum())

        logbook[dis_token][i] = {"auc": aucs, "ctl_count": n_ctl, "dis_count": n_dis}
# -


# compute_total
total_dis = defaultdict(dict)
total_ctl = defaultdict(dict)
for dis_token in range(0, 1270):
    for j, is_gender in enumerate([is_female, is_male, either]):
        total_dis[dis_token][j] = (~np.isnan(dis_rates[is_gender, dis_token])).sum()
        total_ctl[dis_token][j] = is_gender.sum() - total_dis[dis_token][j]
# compute mean
mean_auc = defaultdict(dict)
for dis_token in range(0, 1270):
    for j, is_gender in enumerate([is_female, is_male, either]):
        auc = list()
        for i in range(len(age_groups)):
            auc.append(logbook[dis_token][i]["auc"][j])
        mean_auc[dis_token][j] = np.nanmean(auc)

# +
reverse_tokenizer = {v: k for k, v in ds.tokenizer.items()}
age_group_keys = [
    f"{int(start / 365.25)}-{int(end / 365.25)}" for start, end in age_groups
]

fmt_logbook = dict()
for token in logbook.keys():
    icd = reverse_tokenizer[token]
    fmt_logbook[icd] = defaultdict(dict)
    for i in range(len(age_group_keys) + 1):
        if i == len(age_group_keys):
            aucs = mean_auc[token]
            ctl_count = total_ctl[token]
            dis_count = total_dis[token]
            age_grp = "total"
        else:
            aucs = logbook[token][i]["auc"]
            ctl_count = logbook[token][i]["ctl_count"]
            dis_count = logbook[token][i]["dis_count"]
            age_grp = age_group_keys[i]
        for j, sex in enumerate(["female", "male", "either"]):
            fmt_logbook[icd][sex][age_grp] = {
                "auc": round(aucs[j], 4) if not np.isnan(aucs[j]) else None,
                "ctl_count": int(ctl_count[j]),
                "dis_count": int(dis_count[j]),
            }
# -

pprint.pp(fmt_logbook["death"])

if args.fname is None:
    args.fname = f"auc-min_time_gap-{args.min_time_gap}-ckpt-{ckpt.stem}"
logbook_path = ckpt.parent / f"{args.fname}.json"
with open(logbook_path, "w") as f:
    json.dump(fmt_logbook, f, indent=4)
