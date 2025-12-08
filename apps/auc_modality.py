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

import matplotlib.pyplot as plt
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
    ModalityCollator,
    SexCollator,
    correct_time_offset,
    corrective_indices,
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
    args.ckpt = "fusion/blood-early/ckpt.pt"
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
# +
# biomarkers = ckpt_dict["config"]["biomarkers"]
# if biomarkers is not None:
#     biomarkers = list(biomarkers.keys())
# print(f"biomarkers: {biomarkers}")
# expansion_packs = ckpt_dict["config"]["expansion_packs"]
# print(f"expansion packs: {expansion_packs}")

data_args = ckpt_dict["data_args"].copy()
data_args["subject_list"] = "participants/val_fold.bin"
data_args["stats_subject_list"] = ckpt_dict["data_args"]["subject_list"]
pprint.pp(data_args)
# -

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
data_args["biomarkers"], to_remove

ds = MultimodalUKBDataset(**data_args)

age_start = args.age_start
age_end = args.age_end
age_gap = args.age_gap
age_group_edges = np.arange(age_start, age_end + age_gap, age_gap) * 365.25
n_age_groups = len(age_group_edges) - 1
age_groups = [
    f"{int(i/365.25)}–{int(j/365.25)}"
    for i, j in zip(age_group_edges[:-1], age_group_edges[1:])
]

# +
ctl_collator = ControlRatesCollator(
    age_groups=torch.from_numpy(age_group_edges).to(device)
)
dis_collator = DiseaseRatesCollator(vocab_size=ds.vocab_size)
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
        x0, t0, mod_idx, biomarker, mod_age, x1, t1 = batch_input

        mod_timesteps = mod_collator.step(mod_tokens=mod_idx, timesteps=mod_age)
        mod_timesteps = mod_timesteps[:, [moi.value]]

        if to_remove:
            mod_idx, biomarker, mod_age = remove_modality(
                mod_idx, biomarker, mod_age, remove=moi
            )

        out_dict, _, _ = model(x0, t0, mod_idx, biomarker, mod_age, x1, t1)
        logits = out_dict["logits"].half()

        t0, logits = correct_time_offset(
            t0, t1, logits, offset=args.min_time_gap * 365.25
        )

        timesteps = t0.clone()
        timesteps[x1 == 0] = -1e4
        timesteps[timesteps < mod_timesteps] = -1e4
        ctl_collator.step(timesteps=timesteps, logits=logits)
        dis_collator.step(tokens=x1, timesteps=timesteps, logits=logits)
        sex_collator.step(tokens=x0)

ctl_rates, ctl_times = ctl_collator.finalize()
dis_rates, dis_times = dis_collator.finalize()
sex = sex_collator.finalize()
mod_times = mod_collator.finalize()


# +
ctl_rates, ctl_times = ctl_rates.numpy(), ctl_times.numpy()
dis_rates, dis_times = dis_rates.numpy(), dis_times.numpy()
mod_times = mod_times.numpy()

sex = sex.numpy()
is_female = sex
is_male = ~sex
either = np.logical_or(is_female, is_male)
# -


mod_time = mod_times[:, moi.value]
dis_times[dis_times < mod_time[:, None]] = np.nan
ctl_times[ctl_times < mod_time[:, None]] = np.nan

# +
dis_time_bin = np.searchsorted(age_group_edges, dis_times, side="right") - 1
mod_time = np.searchsorted(age_group_edges, mod_time, side="right") - 1

auc_grids = list()
for is_gender in [is_female, is_male]:
    auc_grid = np.zeros((n_age_groups, n_age_groups))
    for j in tqdm(range(n_age_groups), leave=False):
        mod_mask = mod_time == j
        for i in range(n_age_groups):
            dis_in_range = dis_time_bin == i
            aucs = list()
            for dis_token in range(0, 1270):
                ctl = ctl_rates[:, i, dis_token].copy()
                dis = dis_rates[:, dis_token].copy()
                ctl[~np.isnan(dis)] = np.nan
                dis[~dis_in_range[:, dis_token]] = np.nan
                ctl[~is_gender] = np.nan
                dis[~is_gender] = np.nan
                auc = mann_whitney_auc(ctl[mod_mask], dis[mod_mask])
                aucs.append(auc)
            auc_grid[j, i] = np.nanmean(np.array(aucs))
    auc_grids.append(auc_grid)
f_auc_grid, m_auc_grid = auc_grids


# -
def grid_to_json_dict(grid: np.ndarray, keys1: list, keys2: list):
    assert len(grid.shape) == 2
    assert grid.shape[0] == len(keys1)
    assert grid.shape[1] == len(keys2)
    grid_dict = dict()
    for j, k1 in enumerate(keys1):
        grid_dict[k1] = dict()
        for i, k2 in enumerate(keys2):
            val = grid[j, i]
            if np.isnan(val):
                val = None
            else:
                val = float(val)
            grid_dict[k1][k2] = val
    return grid_dict


# +
f_auc_dict = grid_to_json_dict(f_auc_grid, age_groups, age_groups)
m_auc_dict = grid_to_json_dict(m_auc_grid, age_groups, age_groups)

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

# +
# np.save(ckpt.parent / f"m_{args.modality}_time_cnt.npy", m_mod_time_cnt)
# np.save(ckpt.parent / "f_auc_grid.npy", f_auc_grid)
# np.save(ckpt.parent / "m_auc_grid.npy", m_auc_grid)

# +
auc_grid = m_auc_grid
mod_time_cnt = m_mod_time_cnt
fig, axs = plt.subplots(1, 2, figsize=(16, 8))
axs = axs.ravel()
im = axs[0].imshow(auc_grid, cmap="viridis")
im.set_clim(vmin=0.5, vmax=1)
cbar = plt.colorbar(im, ax=axs[0])
for i in range(auc_grid.shape[0]):
    for j in range(auc_grid.shape[1]):
        auc_val = auc_grid[i, j]
        if not np.isnan(auc_val):
            axs[0].text(
                j,
                i,
                f"{auc_val:.2f}",
                ha="center",
                va="center",
                color="white" if auc_val < 0.5 else "black",
            )
axs[0].set_xticks(np.arange(auc_grid.shape[1]), age_group_edges[:-1] / 365.25)
axs[0].set_yticks(np.arange(auc_grid.shape[1]), age_group_edges[:-1] / 365.25)
axs[0].set_xlabel("disease age bin")
axs[0].set_ylabel("modality age bin")

axs[1].bar(np.arange(len(mod_time_cnt)), mod_time_cnt)
axs[1].set_yscale("log")
axs[1].set_xticks(np.arange(auc_grid.shape[1]), age_group_edges[:-1] / 365.25)
# -


age_groups
