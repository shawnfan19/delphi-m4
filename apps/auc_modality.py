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
    ModalityCollator,
    SexCollator,
    correct_time_offset,
    mann_whitney_auc,
)
from delphi.experiment import eval_iter, move_batch_to_device
from delphi.model.multimodal import DelphiM4, DelphiM4Config
from delphi.multimodal import Modality

# -


def remove_modality(mod_idx, biomarker, mod_age, remove: Modality):

    mod_idx[mod_idx == remove.value] = 0
    mod_age[mod_idx == remove.value] = -1e4
    age_sort = torch.argsort(mod_age, dim=1)
    mod_idx = torch.take_along_dim(input=mod_idx, indices=age_sort, dim=1)
    mod_age = torch.take_along_dim(input=mod_age, indices=age_sort, dim=1)
    del biomarker[remove]

    return mod_idx, biomarker, mod_age


# +
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="delphi-m4/delphi-m4/ckpt.pt")
parser.add_argument("--modality", type=str)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--min_time_gap", type=float, default=0.01)
parser.add_argument("--age_start", type=int, default=40)
parser.add_argument("--age_end", type=int, default=85)
parser.add_argument("--age_gap", type=int, default=5)
parser.add_argument("--fname", type=str)

if "ipykernel" in sys.modules:
    print(f"running in jupyter notebook")
    args = parser.parse_args([])
    args.ckpt = "ablate_blood_biomarker/token/ckpt.pt"
    args.modality = "wbc"
else:
    args = parser.parse_args()

print("args:")
pprint.pp(vars(args))

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
# -
data_args = ckpt_dict["data_args"].copy()
data_args["subject_list"] = "participants/val_fold.bin"
data_args["stats_subject_list"] = ckpt_dict["data_args"]["subject_list"]
pprint.pp(data_args)

biomarkers = data_args["biomarkers"]
if biomarkers is not None:
    if not args.modality in biomarkers:
        biomarkers.append(args.modality)
        to_remove = True
    else:
        to_remove = False
else:
    biomarkers = [args.modality]
    data_args["biomarkers"] = biomarkers
    to_remove = True

ds = MultimodalUKBDataset(**data_args)

age_group_edges = (
    np.arange(args.age_start, args.age_end + args.age_gap, args.age_gap) * 365.25
)
n_age_groups = len(age_group_edges) - 1
age_groups = [
    f"{int(i/365.25)}–{int(j/365.25)}"
    for i, j in zip(age_group_edges[:-1], age_group_edges[1:])
]

# +
model_targets = model.targets.to(device)
model_targets = model_targets[model_targets > 1]

ctl_collator = ControlRatesCollator(
    age_groups=torch.from_numpy(age_group_edges).to(device)
)
dis_collator = DiseaseRatesCollator(targets=model_targets)
sex_collator = SexCollator()
mod_collator = ModalityCollator(modalities=biomarkers)
moi = Modality[args.modality.upper()]

it = tqdm(
    eval_iter(total_size=len(ds), batch_size=args.batch_size),
    total=math.ceil(len(ds) / args.batch_size),
    leave=False,
)
with torch.no_grad():
    for batch_idx in it:
        batch_input = ds.get_batch(batch_idx)
        batch_input = move_batch_to_device(batch_input, device=device)
        x0, t0, biomarker, mod_age, mod_idx, x1, t1 = batch_input
        raw_x0 = x0.clone()

        mod_timesteps = mod_collator.step(mod_tokens=mod_idx, timesteps=mod_age)
        mod_timesteps = mod_timesteps[:, [moi.value]]

        if to_remove:
            mod_idx, biomarker, mod_age = remove_modality(
                mod_idx, biomarker, mod_age, remove=moi
            )

        out_dict, _, _ = model(x0, t0, biomarker, mod_age, mod_idx, x1, t1)
        logits = out_dict["logits"].half()

        t0, logits = correct_time_offset(
            t0, t1, logits, offset=args.min_time_gap * 365.25
        )

        timesteps = t0.clone()
        timesteps[x1 == 0] = -1e4
        timesteps[timesteps < mod_timesteps] = -1e4
        ctl_collator.step(timesteps=timesteps, logits=logits)
        dis_collator.step(tokens=x1, timesteps=timesteps, logits=logits)
        sex_collator.step(tokens=raw_x0)

ctl_rates, ctl_times = ctl_collator.finalize()
dis_rates, dis_times = dis_collator.finalize()
is_female = sex_collator.finalize()
mod_times = mod_collator.finalize()


# +
ctl_rates, ctl_times = ctl_rates.numpy(), ctl_times.numpy()
dis_rates, dis_times = dis_rates.numpy(), dis_times.numpy()
mod_times = mod_times.numpy()

is_female = is_female.numpy()
is_male = ~is_female
either = np.logical_or(is_female, is_male)
# -


mod_time = mod_times[:, moi.value]
dis_times[dis_times < mod_time[:, None]] = np.nan
ctl_times[ctl_times < mod_time[:, None]] = np.nan

# +
dis_time_bin = np.searchsorted(age_group_edges, dis_times, side="right") - 1
mod_time_bin = np.searchsorted(age_group_edges, mod_time, side="right") - 1

auc_grids, ctl_grids, dis_grids = list(), list(), list()
for is_gender in [is_female, is_male]:
    auc_grid = np.zeros((n_age_groups, n_age_groups, model.config.vocab_size))
    ctl_grid, dis_grid = auc_grid.copy(), auc_grid.copy()
    for j in tqdm(range(n_age_groups), leave=False):
        mod_mask = np.logical_and(mod_time_bin == j, ~np.isnan(mod_time))
        for i in range(n_age_groups):
            aucs, ctl_cts, dis_cts = list(), list(), list()
            for dis_token in range(0, 1270):
                ctl = ctl_rates[:, i, dis_token].copy()
                dis = dis_rates[:, dis_token].copy()

                ctl[~np.isnan(dis)] = np.nan
                dis_in_range = np.logical_and(
                    dis_time_bin[:, dis_token] == i, ~np.isnan(dis_times[:, dis_token])
                )
                dis[~dis_in_range] = np.nan

                ctl[~is_gender] = np.nan
                dis[~is_gender] = np.nan

                ctl[~mod_mask] = np.nan
                dis[~mod_mask] = np.nan

                aucs.append(mann_whitney_auc(ctl, dis))
                ctl_cts.append((~np.isnan(ctl)).sum())
                dis_cts.append((~np.isnan(dis)).sum())

            auc_grid[j, i] = np.array(aucs)
            ctl_grid[j, i] = np.array(ctl_cts)
            dis_grid[j, i] = np.array(dis_cts)

    auc_grids.append(auc_grid)
    ctl_grids.append(ctl_grid)
    dis_grids.append(dis_grid)

f_auc_grid, m_auc_grid = auc_grids
f_ctl_grid, m_ctl_grid = ctl_grids
f_dis_grid, m_dis_grid = dis_grids


# -


def grid_to_json_dict(
    auc_grid: np.ndarray,
    ctl_grid: np.ndarray,
    dis_grid: np.ndarray,
    keys1: list,
    keys2: list,
    detokenizer: dict,
):
    assert len(auc_grid.shape) == 3
    assert auc_grid.shape == ctl_grid.shape == dis_grid.shape
    assert auc_grid.shape[0] == len(keys1)
    assert auc_grid.shape[1] == len(keys2)
    grid_dict = dict()
    for j, k1 in enumerate(keys1):
        grid_dict[k1] = defaultdict(dict)
        for i, k2 in enumerate(keys2):
            aucs, ctl_cts, dis_cts = auc_grid[j, i], ctl_grid[j, i], dis_grid[j, i]
            for i, (auc, ctl_ct, dis_ct) in enumerate(zip(aucs, ctl_cts, dis_cts)):
                if not np.isnan(auc):
                    grid_dict[k1][k2][detokenizer[i]] = {
                        "auc": float(auc),
                        "ctl_count": int(ctl_ct),
                        "dis_count": int(dis_ct),
                    }
    return grid_dict


# +
f_auc_dict = grid_to_json_dict(
    f_auc_grid,
    f_ctl_grid,
    f_dis_grid,
    age_groups,
    age_groups,
    detokenizer=ds.detokenizer,
)
m_auc_dict = grid_to_json_dict(
    m_auc_grid,
    m_ctl_grid,
    m_dis_grid,
    age_groups,
    age_groups,
    detokenizer=ds.detokenizer,
)

f_mod_time_cnt = {
    age_groups[i]: int(np.logical_and(is_female, mod_time == i).sum())
    for i in range(n_age_groups)
}
m_mod_time_cnt = {
    age_groups[i]: int(np.logical_and(is_male, mod_time == i).sum())
    for i in range(n_age_groups)
}
# -

logbook = {
    "male": m_auc_dict,
    "female": f_auc_dict,
    "male_count": m_mod_time_cnt,
    "female_count": f_mod_time_cnt,
}
with open(ckpt.parent / f"{args.modality.lower()}_auc.json", "w") as f:
    json.dump(logbook, f)
