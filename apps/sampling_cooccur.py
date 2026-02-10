# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os

os.chdir("/hps/nobackup/birney/users/sfan/Delphi")

import argparse
import math
import pprint
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from delphi.data.ukb import UKBDataset, cut_prompt
from delphi.data.utils import pack_clusters
from delphi.env import DELPHI_CKPT_DIR
from delphi.eval import ClusterStatsTracker, CooccurrenceTracker
from delphi.experiment import eval_iter
from delphi.model.transformer import Delphi2M, Delphi2MConfig, generate

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="delphi-2m-og/ckpt.pt")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--age", type=int, default=60)
parser.add_argument("--subset", type=int)
parser.add_argument("--stop_at_block_size", type=bool, default=True)
parser.add_argument("--max_new_tokens", type=int, default=128)
parser.add_argument("--prompt_no_event", type=bool, default=False)

if "ipykernel" in sys.modules:
    print(f"running in jupyter notebook")
    args = parser.parse_args([])
    args.ckpt = "cluster/dx_token/ckpt.pt"
    args.batch_size = 512
    args.age = None
else:
    args = parser.parse_args()
print("args:")
pprint.pp(vars(args))

# %%
ckpt = Path(DELPHI_CKPT_DIR) / args.ckpt
ckpt_dict = torch.load(
    ckpt,
    map_location=torch.device("cpu") if not torch.cuda.is_available() else None,
)

# %%
model = Delphi2M(Delphi2MConfig(**ckpt_dict["model_args"]))
pprint.pp(ckpt_dict["model_args"])
missing, unexpected = model.load_state_dict(ckpt_dict["model"], strict=False)
print("missing:", missing)
print("unexpected:", unexpected)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# %%
data_args = ckpt_dict["data_args"].copy()
data_args["subject_list"] = "participants/val_fold.bin"
data_args["perturb"] = False
data_args["deterministic"] = True
data_args["additional_dx_token"] = ckpt_dict["data_args"].get(
    "additional_dx_token", False
)
pprint.pp(data_args)

# %%
ds = UKBDataset(**data_args)
if args.age is not None:
    ds.subset_participants_for_prompt(prompt_age=args.age * 365.25)
else:
    ds.subset_by_tokens(tokens=ds.lifestyle_tokens)

# %%
ds.dx_token


# %%
whitelist = np.concatenate(
    (np.array([0, 1]), np.array([ds.dx_token]), ds.sex_tokens, ds.lifestyle_tokens)
)
whitelist


# %%

# %%
if args.subset is None:
    total = len(ds)
else:
    total = args.subset
it = eval_iter(total_size=total, batch_size=args.batch_size)
pbar = tqdm(it, total=math.ceil(total / args.batch_size), leave=False)
gt_tracker = CooccurrenceTracker(vocab_size=model.config.vocab_size)
gt_stats = ClusterStatsTracker()
tracker = CooccurrenceTracker(vocab_size=model.config.vocab_size)
stats = ClusterStatsTracker()

break_clusters = data_args.get("break_clusters", False)

torch.manual_seed(42)

for batch_idx in pbar:
    X0, T0, X1, T1 = ds.get_batch(batch_idx)
    pmt_idx, pmt_age, cutoff = cut_prompt(
        X0,
        T0,
        prompt_age=args.age * 365.25 if args.age is not None else None,
        prompt_token=torch.Tensor(ds.lifestyle_tokens),
        append_no_event=args.prompt_no_event,
    )
    cutoff = cutoff.detach().cpu().numpy()

    X1_np = X1.detach().cpu().numpy().copy()
    T1_np = T1.detach().cpu().numpy().copy()
    X1_np[T1_np <= cutoff] = 0
    T1_np[T1_np <= cutoff] = -1e4
    if break_clusters:
        X1_np, T1_np = pack_clusters(X1_np, T1_np, whitelist, dx_token=ds.dx_token)
    gt_tracker.step(tokens=X1_np, timesteps=T1_np)
    gt_stats.step(tokens=X1_np, timesteps=T1_np)

    pmt_idx, pmt_age = pmt_idx.to(device), pmt_age.to(device)
    # cutoff = cutoff.to(device)
    tokens, timesteps, logits, gen_stats = generate(
        model=model,
        idx=pmt_idx,
        age=pmt_age,
        max_age=T1.max(dim=1)[0].to(device),
        no_repeat=True,
        no_repeat_except=torch.Tensor([1, ds.dx_token]),
        max_new_tokens=args.max_new_tokens,
        termination_tokens=[1269],
        stop_at_block_size=args.stop_at_block_size,
        exclude_pad=True,
    )

    tokens = tokens.detach().cpu().numpy()
    timesteps = timesteps.detach().cpu().numpy()
    tokens[timesteps <= cutoff] = 0
    timesteps[timesteps <= cutoff] = -1e4
    if break_clusters:
        tokens, timesteps = pack_clusters(
            tokens, timesteps, whitelist, dx_token=ds.dx_token
        )
    tracker.step(tokens=tokens, timesteps=timesteps)
    stats.step(tokens=tokens, timesteps=timesteps)

    n_gen = gen_stats["n_gen"].mean() - gen_stats["n_prompt"].mean()
    assert n_gen > 0, n_gen
    pbar.set_postfix({"n_gen": n_gen})

# %%
gt_heatmap = gt_tracker.finalize()
gt_n_clusters, gt_cluster_sizes = gt_stats.finalize()

heatmap = tracker.finalize()
n_clusters, cluster_sizes = stats.finalize()

# %%
cmap = "inferno"
fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
axs = axs.ravel()
# calculate the 99.5th percentile to ignore extreme outliers
vmax = np.percentile(gt_heatmap, 99.5)
axs[0].imshow(np.log1p(gt_heatmap), cmap=cmap, vmin=0, vmax=np.log1p(vmax))
axs[0].set_xlabel("token index")
axs[0].set_ylabel("token index")
axs[0].set_title("ground truth")
# vmax = np.percentile(heatmap, 99.5)
axs[1].imshow(np.log1p(heatmap), cmap=cmap, vmax=np.log1p(vmax))
axs[1].set_xlabel("token index")
axs[1].set_ylabel("token index")
axs[1].set_title("model")

# %%
(gt_heatmap == heatmap).all()

# %%

# %%
k = 15
print(f"top {k} diseases that show up in clusters")
diseases = np.argsort(gt_heatmap.sum(axis=1))[::-1]
for i in range(k):
    print(ds.detokenizer[diseases[i]])

# %%

# %%
alpha = 0.3
fig, axs = plt.subplots(1, 2, figsize=(16, 8))
axs = axs.ravel()

n_bins = max(gt_n_clusters.max() + 1, n_clusters.max() + 1)
bins = np.arange(1, n_bins)
axs[0].hist(
    gt_n_clusters,
    bins=bins,
    alpha=alpha,
    label="ground truth",
)
axs[0].hist(n_clusters, bins=bins, alpha=alpha, label="model")
axs[0].set_xlabel("# disease clusters per participant")
axs[0].set_xticks(bins, bins)
axs[0].legend()

# n_bins = max(gt_cluster_sizes.max() + 1, cluster_sizes.max() + 1)
n_bins = 15
bins = np.arange(1, n_bins)
axs[1].hist(
    gt_cluster_sizes,
    bins=bins,
    alpha=alpha,
    label="ground truth",
)
axs[1].hist(
    cluster_sizes,
    bins=bins,
    alpha=alpha,
    label="model",
)
axs[1].set_xticks(bins, bins)
axs[1].legend()
axs[1].set_xlabel("size of disease clusters")

# %%

# %%
