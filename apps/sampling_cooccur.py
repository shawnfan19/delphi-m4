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
import math
import pprint
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from delphi.data import MultimodalDataset
from delphi.data.auto import multimodal_reader_cls
from delphi.data.transform import MultimodalPrompt, TokenTransform
from delphi.data.utils import pack_clusters
from delphi.env import DELPHI_CKPT_DIR, DELPHI_CKPT_WRITE
from delphi.eval import ClusterStatsTracker, TiedEventTracker
from delphi.experiment import GenerateConfig, eval_iter, load_ckpt
from delphi.model.transformer import generate


# %%
@dataclass
class TaskConfig(GenerateConfig):
    # if set, save the figures under DELPHI_CKPT_WRITE/<ckpt dir>/<write>/
    write: None | str = None


args = TaskConfig.from_cli()
print("args:")
pprint.pp(args)

# %%
model, ckpt_dict = load_ckpt(Path(DELPHI_CKPT_DIR) / args.ckpt)
device = "cuda" if torch.cuda.is_available() else "cpu"

out_dir = None
if args.write is not None:
    out_dir = Path(DELPHI_CKPT_WRITE) / Path(args.ckpt).parent / args.write
    out_dir.mkdir(parents=True, exist_ok=True)

# %%
# dataset-aware: UKB on the cluster, AoU on the workbench. Honors
# DELPHI_DATASET (set in the dsub env), else auto-detects from the data dir.
ReaderCls = multimodal_reader_cls()
reader = ReaderCls(biomarkers=None)
pids = ReaderCls.participants("val")

token_transform_args = ckpt_dict["token_transform_args"]
token_transform = TokenTransform(**token_transform_args)
token_transform.describe()

lifestyle_tokens = [reader.tokenizer[k] for k in reader.lifestyle_keys]
sex_tokens = [reader.tokenizer[k] for k in reader.sex_keys]
dx_token = token_transform_args.get("dx_token") or 1
break_clusters = token_transform_args.get("break_clusters", False)
# reuse the exact whitelist training stored so dissolve/pack stay symmetric;
# fall back to reconstructing it for checkpoints that predate whitelist_tokens.
whitelist = np.array(
    token_transform_args.get("whitelist_tokens")
    or [0, 1] + sex_tokens + lifestyle_tokens
)

if dx_token != 1:
    model.config.self_terminate_except = list(
        set(model.config.self_terminate_except).union({dx_token})
    )

# %%
prompt_age_arg = args.prompt_age if args.prompt_age is not None else "recruitment"
if prompt_age_arg == "recruitment":
    rec = reader.times_at(pids, "recruitment")
    has_rec = ~np.isnan(rec)
    pids = pids[has_rec]
    prompt_age = {int(p): float(a) for p, a in zip(pids, rec[has_rec])}
else:
    prompt_age = float(prompt_age_arg) * 365.25
prompt_transform = MultimodalPrompt(
    prompt_age=prompt_age, biomarker2idx={}, append_no_event=args.prompt_no_event
)
ds = MultimodalDataset(
    reader=reader,
    pids=pids,
    token_transform=token_transform,
    prompt_transform=prompt_transform,
)

# %%
if args.subsample is None:
    total = len(ds)
else:
    total = args.subsample
it = eval_iter(total_size=total, batch_size=args.batch_size)
pbar = tqdm(it, total=math.ceil(total / args.batch_size), leave=False)
gt_tracker = TiedEventTracker(vocab_size=model.config.vocab_size)
gt_stats = ClusterStatsTracker()
tracker = TiedEventTracker(vocab_size=model.config.vocab_size)
stats = ClusterStatsTracker()

torch.manual_seed(42)

for batch_idx in pbar:
    pmt_idx, pmt_age, _, _, _, X1, T1 = ds.get_batch(batch_idx)
    if isinstance(prompt_age, dict):
        batch_pids = ds.participants[batch_idx]
        cutoff = np.array([prompt_age[int(p)] for p in batch_pids])[:, None]
    else:
        cutoff = prompt_age

    X1_np = X1.detach().cpu().numpy().copy()
    T1_np = T1.detach().cpu().numpy().copy()
    X1_np[T1_np <= cutoff] = 0
    T1_np[T1_np <= cutoff] = -1e4
    if break_clusters:
        X1_np, T1_np = pack_clusters(X1_np, T1_np, whitelist, dx_token=dx_token)
    gt_tracker.step(tokens=X1_np, timesteps=T1_np)
    gt_stats.step(tokens=X1_np, timesteps=T1_np)

    pmt_idx, pmt_age = pmt_idx.to(device), pmt_age.to(device)
    tokens, timesteps, gen_stats = generate(
        model=model,
        idx=pmt_idx,
        age=pmt_age,
        max_age=T1.max(dim=1)[0].to(device),
        max_new_tokens=args.max_new_tokens,
        termination_tokens=[1269],
        exclude_pad=True,
        cached=True,
    )

    tokens = tokens.detach().cpu().numpy()
    timesteps = timesteps.detach().cpu().numpy()
    tokens[timesteps <= cutoff] = 0
    timesteps[timesteps <= cutoff] = -1e4
    if break_clusters:
        tokens, timesteps = pack_clusters(
            tokens, timesteps, whitelist, dx_token=dx_token
        )
    tracker.step(tokens=tokens, timesteps=timesteps)
    stats.step(tokens=tokens, timesteps=timesteps)

    pbar.set_postfix(
        {"n_gen": gen_stats["n_gen"].mean(), "n_pmt": gen_stats["n_prompt"].mean()}
    )

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

if out_dir is not None:
    with (out_dir / "cooccur_heatmap.png").open("wb") as f:
        fig.savefig(f, format="png", dpi=300, bbox_inches="tight")
plt.show()

# %%
(gt_heatmap == heatmap).all()

# %%

# %%
k = 15
print(f"top {k} diseases that show up in clusters")
diseases = np.argsort(gt_heatmap.sum(axis=1))[::-1]
for i in range(k):
    print(reader.detokenizer[diseases[i]])

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

if out_dir is not None:
    with (out_dir / "cluster_hist.png").open("wb") as f:
        fig.savefig(f, format="png", dpi=300, bbox_inches="tight")
plt.show()

# %%

# %%
