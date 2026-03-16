# +
import argparse
import json
import math
import pprint
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from delphi.data.ukb import MultimodalUKBDataset
from delphi.env import DELPHI_CKPT_DIR
from delphi.eval import AgeStratRatesCollator as ControlRatesCollator
from delphi.eval import (
    DiseaseRatesCollator,
    SexCollator,
    correct_time_offset,
    mann_whitney_auc,
)
from delphi.experiment import eval_iter, load_ckpt, move_batch_to_device

# -


# +
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="delphi-m4/delphi-m4/ckpt.pt")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--min_time_gap", type=float, default=0.01)
parser.add_argument("--age_start", type=int, default=40)
parser.add_argument("--age_end", type=int, default=85)
parser.add_argument("--age_gap", type=int, default=5)
parser.add_argument(
    "--after_biomarker",
    type=str,
    default=None,
    help="Only compute AUC at time points after first occurrence of this biomarker modality (e.g. 'nmr')",
)
parser.add_argument(
    "--modalities",
    type=str,
    nargs="+",
    default=None,
    help="Only evaluate participants with ALL of these modalities (e.g. '--modalities nmr lipid')",
)
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


# +
ckpt = Path(DELPHI_CKPT_DIR) / args.ckpt
model, ckpt_dict = load_ckpt(ckpt)
device = "cuda" if torch.cuda.is_available() else "cpu"

data_args = ckpt_dict["data_args"].copy()
data_args["subject_list"] = "participants/val_fold.bin"
data_args["stats_subject_list"] = ckpt_dict["data_args"]["subject_list"]
# data_args["must_have_biomarkers"] = data_args["biomarkers"]
if args.modalities is not None:
    data_args["must_have_biomarkers"] = args.modalities

pprint.pp(data_args)
# -
ds = MultimodalUKBDataset(**data_args)

age_group_edges = (
    np.arange(args.age_start, args.age_end + args.age_gap, args.age_gap) * 365.25
)
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
        t0 = out_dict["age"]

        t0, logits = correct_time_offset(
            t0, t1, logits, offset=args.min_time_gap * 365.25
        )
        ctl_collator.step(timesteps=t0, logits=logits)
        dis_collator.step(tokens=x1, timesteps=t0, logits=logits)
        sex_collator.step(tokens=x0)

ctl_rates, ctl_times = ctl_collator.finalize()
dis_rates, dis_times = dis_collator.finalize()
is_female = sex_collator.finalize()
# +
ctl_rates, ctl_times = ctl_rates.numpy(), ctl_times.numpy()
dis_rates, dis_times = dis_rates.numpy(), dis_times.numpy()
is_female = is_female.numpy()

is_male = ~is_female

if args.after_biomarker is not None:
    bio_first = ds.first_occurrence_times(args.after_biomarker)
    # mask out participants without the biomarker
    no_bio = np.isnan(bio_first)
    ctl_rates[no_bio] = np.nan
    dis_rates[no_bio] = np.nan
    # mask control scores at time points before the biomarker
    before_bio_ctl = ctl_times < bio_first[:, None]
    ctl_rates[before_bio_ctl] = np.nan
    # mask disease scores at time points before the biomarker
    before_bio_dis = dis_times < bio_first[:, None]
    dis_rates[before_bio_dis] = np.nan

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

        for is_gender in [is_female, is_male]:

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

# +
reverse_tokenizer = {v: k for k, v in ds.tokenizer.items()}
age_group_keys = [
    f"{int(start / 365.25)}-{int(end / 365.25)}" for start, end in age_groups
]

fmt_logbook = dict()
for token in logbook.keys():
    icd = reverse_tokenizer[token]
    fmt_logbook[icd] = defaultdict(dict)
    for i, age_grp in enumerate(age_group_keys):
        aucs = logbook[token][i]["auc"]
        ctl_count = logbook[token][i]["ctl_count"]
        dis_count = logbook[token][i]["dis_count"]
        for j, sex in enumerate(["female", "male"]):
            fmt_logbook[icd][sex][age_grp] = {
                "auc": round(aucs[j], 4) if not np.isnan(aucs[j]) else None,
                "ctl_count": int(ctl_count[j]),
                "dis_count": int(dis_count[j]),
            }
# -

pprint.pp(fmt_logbook["death"])

if args.fname is None:
    args.fname = f"auc-min_time_gap-{args.min_time_gap}-ckpt-{ckpt.stem}"
    if args.after_biomarker is not None:
        args.fname += f"-after_{args.after_biomarker}"
    if args.modalities is not None:
        args.fname += f"-modalities_{'_'.join(args.modalities)}"
logbook_path = ckpt.parent / f"{args.fname}.json"
with open(logbook_path, "w") as f:
    json.dump(fmt_logbook, f, indent=4)
