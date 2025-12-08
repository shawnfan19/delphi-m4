# +
import argparse
import json
import math
import os
import pprint
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from tqdm.autonotebook import tqdm

from delphi import DAYS_PER_YEAR
from delphi.data.ukb import UKBDataset
from delphi.data.utils import collate_batches
from delphi.env import DELPHI_CKPT_DIR
from delphi.eval import mann_whitney_auc
from delphi.experiment import eval_iter, move_batch_to_device
from delphi.model.transformer import Delphi2M, Delphi2MConfig

# -


def rates_by_age_bin(
    input_time: np.ndarray,
    predicted_rates: np.ndarray,
    disease_free: np.ndarray,
    targets: np.ndarray,
    sub_idx: np.ndarray,
    dis_token: int,
    age_groups: list[tuple[int, int]],
):

    ctl_subjects = []
    dis_subjects = []
    ctl_rates = []
    dis_rates = []

    have_disease = targets == dis_token

    for age_start, age_end in age_groups:

        in_time_range = input_time >= age_start * DAYS_PER_YEAR
        in_time_range &= input_time < age_end * DAYS_PER_YEAR

        is_ctl = disease_free & in_time_range
        is_dis = have_disease & in_time_range

        ctl_subjects.append(sub_idx[is_ctl])
        dis_subjects.append(sub_idx[is_dis])

        ctl_rates.append(predicted_rates[is_ctl])
        dis_rates.append(predicted_rates[is_dis])

    return ctl_subjects, dis_subjects, ctl_rates, dis_rates


def sample_one_per_participant(subjects, rates):

    assert len(subjects) == len(rates), "subjects and rates must have the same length"

    perm = np.random.permutation(len(subjects))
    subjects = subjects[perm]
    rates = rates[perm]
    _, uniq_idx = np.unique(subjects, return_index=True)

    return rates[uniq_idx]


def corrective_indices(T0: np.ndarray, T1: np.ndarray, offset: float):

    m, _ = T0.shape
    _, p = T1.shape

    C = np.zeros((m, p), dtype=int)

    for i in range(m):
        t0_row = T0[i]
        t1_row = T1[i]

        c_idx = (
            np.broadcast_to(t0_row, (t1_row.size, t0_row.size))
            <= (t1_row - offset).reshape(-1, 1)
        ).sum(axis=1) - 1

        C[i] = c_idx

    return C


# +
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="time/baseline/ckpt.pt")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--min_time_gap", type=float, default=0.01)
parser.add_argument("--age_start", type=int, default=40)
parser.add_argument("--age_end", type=int, default=85)
parser.add_argument("--age_gap", type=int, default=5)
parser.add_argument("--fname", type=str)

if "ipykernel" in sys.modules:
    print(f"running in jupyter notebook")
    args = parser.parse_args([])
    args.ckpt = "time/baseline/ckpt.pt"
else:
    args = parser.parse_args()

print("args:")
pprint.pp(vars(args))

# +
ckpt = Path(DELPHI_CKPT_DIR) / args.ckpt

ckpt_dict = torch.load(
    ckpt, map_location=torch.device("cpu") if not torch.cuda.is_available() else None
)
model_cfg = Delphi2MConfig(**ckpt_dict["model_args"])
model = Delphi2M(model_cfg)
model.load_state_dict(ckpt_dict["model"])

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print(f"model: {ckpt} [iter: {ckpt_dict['iter_num']}]")


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


data_args = ckpt_dict.get("data_args", dict())
if len(data_args) > 0:
    data_args = ckpt_dict["data_args"].copy()
    data_args["stats_subject_list"] = ckpt_dict["data_args"]["subject_list"]
else:
    print(f"no data args found in checkpoint")
    data_args = get_init_defaults(UKBDataset)
data_args["subject_list"] = "participants/val_fold.bin"
data_args["perturb"] = False
pprint.pp(data_args)

ds = UKBDataset(**data_args)

batch_size = args.batch_size
total_size = len(ds)
it = tqdm(
    eval_iter(total_size=total_size, batch_size=batch_size),
    total=math.ceil(total_size / batch_size),
    leave=False,
)
logits_lst = list()
x0_lst = list()
t0_lst = list()
x1_lst = list()
t1_lst = list()
with torch.no_grad():
    for batch_idx in it:
        batch_input = ds.get_batch(batch_idx)
        batch_input = move_batch_to_device(batch_input, device=device)
        x0, t0, x1, t1 = batch_input
        x0 = x0.detach().cpu().numpy()
        t0 = t0.detach().cpu().numpy()
        x1 = x1.detach().cpu().numpy()
        t1 = t1.detach().cpu().numpy()

        out_dict, _, _ = model(*batch_input)
        batch_logits = out_dict["logits"].detach().cpu().numpy()
        batch_logits = batch_logits.astype(np.float16)
        _sub_idx, _pos_idx = np.nonzero(np.logical_and(x1 > 0, t0 > -1e4))
        logits_lst.append(batch_logits[_sub_idx, _pos_idx])
        x0_lst.append(x0)
        t0_lst.append(t0)
        x1_lst.append(x1)
        t1_lst.append(t1)
logits = np.vstack(logits_lst)

X_t0 = collate_batches(x0_lst)
T_t0 = collate_batches(t0_lst, fill_value=-1e4)
X_t1 = collate_batches(x1_lst)
T_t1 = collate_batches(t1_lst, fill_value=-1e4)

# +
min_time_gap = args.min_time_gap
sub_idx, pos_idx = np.nonzero(np.logical_and(X_t1 > 0, T_t0 > -1e4))
C = corrective_indices(
    T0=T_t0,
    T1=T_t1,
    offset=min_time_gap * 365.25,
)
offset_pos_idx = C[sub_idx, pos_idx]
logits_idx = np.arange(logits.shape[0])

has_input = offset_pos_idx >= 0
sub_idx, pos_idx = sub_idx[has_input], pos_idx[has_input]
offset_pos_idx = offset_pos_idx[has_input]
logits_idx = logits_idx[has_input]
logits_idx = logits_idx[np.arange(logits_idx.shape[0]) + offset_pos_idx - pos_idx]
t_t0 = T_t0[sub_idx, offset_pos_idx]
targets = X_t1[sub_idx, pos_idx]
# -
logits.shape, sub_idx.shape

# +
age_start = args.age_start
age_end = args.age_end
age_gap = args.age_gap
age_group_edges = np.arange(age_start, age_end + age_gap, age_gap)
age_groups = [(i, j) for i, j in zip(age_group_edges[:-1], age_group_edges[1:])]
age_group_keys = [f"{start}-{end}" for start, end in age_groups]

tokenizer = ds.tokenizer
is_female = (X_t0 == tokenizer["female"]).any(axis=1)[sub_idx]
is_male = (X_t0 == tokenizer["male"]).any(axis=1)[sub_idx]
is_gender_dict = {
    "female": is_female,
    "male": is_male,
    "either": is_female | is_male,
}
# -

all_keys = list(tokenizer.keys())
icd_keys = all_keys[:1270]

logbook = {}
for disease in tqdm(icd_keys):
    logbook[disease] = {}
    dis_token = tokenizer[disease]

    disease_free = (~(X_t1 == dis_token).any(axis=1))[sub_idx]
    y_t1 = logits[logits_idx, dis_token]

    for gender, is_gender in is_gender_dict.items():

        ctl_subjects, dis_subjects, ctl_rates, dis_rates = rates_by_age_bin(
            input_time=t_t0[is_gender],
            sub_idx=sub_idx[is_gender],
            predicted_rates=y_t1[is_gender],
            disease_free=disease_free[is_gender],
            targets=targets[is_gender],
            dis_token=dis_token,
            age_groups=age_groups,
        )

        ctl_rates = [
            sample_one_per_participant(subj, rate)
            for subj, rate in zip(ctl_subjects, ctl_rates)
        ]
        auc = [
            mann_whitney_auc(ctl_rate, dis_rate)
            for ctl_rate, dis_rate in zip(ctl_rates, dis_rates)
        ]

        n_ctl = [len(np.unique(s)) for s in ctl_subjects]
        n_dis = [len(s) for s in dis_subjects]
        logbook[disease][gender] = {
            age_group_keys[i]: {
                "auc": round(float(auc[i]), 2) if not np.isnan(auc[i]) else None,
                "ctl_count": int(n_ctl[i]),
                "dis_count": int(n_dis[i]),
            }
            for i in range(len(age_groups))
        }
        mean_auc = float(np.nanmean(auc))
        all_ctl = np.concatenate(ctl_subjects)
        all_dis = np.concatenate(dis_subjects)
        logbook[disease][gender]["total"] = {
            "auc": round(mean_auc, 2) if not np.isnan(mean_auc) else None,
            "ctl_count": len(np.unique(all_ctl)),
            "dis_count": len(all_dis),
        }

pprint.pp(logbook["death"])

if args.fname is None:
    args.fname = f"auc-min_time_gap-{args.min_time_gap}-ckpt-{ckpt.stem}"
logbook_path = ckpt.parent / f"{args.fname}.json"
with open(logbook_path, "w") as f:
    json.dump(logbook, f, indent=4)
