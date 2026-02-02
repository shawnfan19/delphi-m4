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
from scipy import sparse
from tqdm import tqdm

from delphi.data.ukb import UKBDataset, cut_prompt
from delphi.env import DELPHI_CKPT_DIR
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
    args.ckpt = "cluster/homo_cluster_poisson/ckpt.pt"
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
data_args = ckpt_dict["data_args"]
data_args["subject_list"] = "participants/val_fold.bin"
data_args["perturb"] = False
data_args["deterministic"] = True
pprint.pp(data_args)

# %%
ds = UKBDataset(**data_args)
if args.age is not None:
    ds.subset_participants_for_prompt(prompt_age=args.age * 365.25)
else:
    ds.subset_by_tokens(tokens=ds.lifestyle_tokens)


# %%
class ClusterStatsTracker:
    def __init__(self):
        self.n_clusters_per_sub = list()
        self.cluster_size = list()

    def step(self, tokens: np.ndarray, timesteps: np.ndarray):

        # 1. Mask Padding
        # Filter out 0 (padding) tokens
        mask = tokens != 0
        if not np.any(mask):
            return  # Skip empty batches
        flat_tokens = tokens[mask]
        flat_times = timesteps[mask]

        # Get row indices (0 to N-1) for every valid token
        N, K = tokens.shape
        row_indices = np.arange(N).repeat(K).reshape(N, K)
        flat_rows = row_indices[mask]
        # 2. Identify Unique Events
        # An event is a unique combination of (Batch_Row_Index, Timestep)
        # We stack them to create unique keys for grouping
        event_keys = np.column_stack((flat_rows, flat_times))

        # np.unique maps every (row, time) pair to a unique integer ID (0 to Num_Events-1)
        _, event_ids, cluster_size = np.unique(
            event_keys, axis=0, return_index=True, return_counts=True
        )

        event_subs = flat_rows[event_ids]
        cluster_subs = event_subs[cluster_size > 1]
        _, n_clusters_per_sub = np.unique(cluster_subs, return_counts=True)

        self.n_clusters_per_sub.append(n_clusters_per_sub)
        self.cluster_size.append(cluster_size)

    def finalize(self):
        return np.concatenate(self.n_clusters_per_sub), np.concatenate(
            self.cluster_size
        )


class CooccurrenceTracker:
    def __init__(self, vocab_size):
        """
        Initializes the tracker.

        Parameters:
        - vocab_size: int, the dimension V of the vocabulary (max token id + 1).
        """
        self.vocab_size = vocab_size

        # We use a sparse matrix for the running sum to save memory during accumulation.
        # CSR format is efficient for arithmetic operations.
        self.global_cooccurrence = sparse.csr_matrix(
            (vocab_size, vocab_size), dtype=np.int32
        )

    def step(self, tokens, timesteps):
        """
        Updates the co-occurrence counts with a new batch of data.

        Parameters:
        - tokens: (N, K) numpy array of integers.
        - timesteps: (N, K) numpy array of discrete days.
        """
        # 1. Mask Padding
        # Filter out 0 (padding) tokens
        mask = tokens != 0
        if not np.any(mask):
            return  # Skip empty batches
        flat_tokens = tokens[mask]
        flat_times = timesteps[mask]

        # Get row indices (0 to N-1) for every valid token
        N, K = tokens.shape
        row_indices = np.arange(N).repeat(K).reshape(N, K)
        flat_rows = row_indices[mask]
        # 2. Identify Unique Events
        # An event is a unique combination of (Batch_Row_Index, Timestep)
        # We stack them to create unique keys for grouping
        event_keys = np.column_stack((flat_rows, flat_times))

        # np.unique maps every (row, time) pair to a unique integer ID (0 to Num_Events-1)
        _, event_ids = np.unique(event_keys, axis=0, return_inverse=True)
        num_events = event_ids.max() + 1
        # 3. Create Incidence Matrix for this Batch (Events x Vocab)
        # Rows = Events, Cols = Token IDs
        # Values = 1 (presence).
        # Note: If a token appears twice in one event, the values sum up.
        ones = np.ones(len(flat_tokens), dtype=int)

        X_batch = sparse.csr_matrix(
            (ones, (event_ids, flat_tokens)), shape=(num_events, self.vocab_size)
        )
        # 4. Compute Batch Co-occurrence via Dot Product
        # (V x Events) @ (Events x V) -> (V x V)
        batch_cooccurrence = X_batch.T @ X_batch
        # 5. Update Global State
        self.global_cooccurrence += batch_cooccurrence

    def finalize(self, as_dense=True):
        """
        Finalizes the calculation, removes self-occurrences, and returns the heatmap.

        Parameters:
        - as_dense: bool. If True, returns a numpy array. If False, returns sparse matrix.

        Returns:
        - heatmap: (V, V) matrix.
        """
        # Work on a copy to avoid corrupting the running state if called multiple times
        result_matrix = self.global_cooccurrence.copy()

        # The prompt asks for co-occurrence with "any OTHER token".
        # We set the diagonal to 0 to remove self-occurrences (Token A with Token A).
        result_matrix.setdiag(0)

        # Eliminate any zeros created by setdiag from the sparse structure
        result_matrix.eliminate_zeros()

        if as_dense:
            return result_matrix.toarray()
        else:
            return result_matrix


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
for batch_idx in pbar:
    X0, T0, X1, T1 = ds.get_batch(batch_idx)
    pmt_idx, pmt_age = cut_prompt(
        X0,
        T0,
        prompt_age=args.age * 365.25 if args.age is not None else None,
        prompt_token=torch.Tensor(ds.lifestyle_tokens),
        append_no_event=args.prompt_no_event,
    )

    # X1[T1 <= prompt_age] = 0
    # T1[T1 <= prompt_age] = -1e4
    X1_np = X1.detach().cpu().numpy()
    T1_np = T1.detach().cpu().numpy()
    gt_tracker.step(tokens=X1_np, timesteps=T1_np)
    gt_stats.step(tokens=X1_np, timesteps=T1_np)

    pmt_idx, pmt_age = pmt_idx.to(device), pmt_age.to(device)
    tokens, timesteps, logits, gen_stats = generate(
        model=model,
        idx=pmt_idx,
        age=pmt_age,
        max_age=T1.max(dim=1)[0].to(device),
        no_repeat=True,
        max_new_tokens=args.max_new_tokens,
        termination_tokens=[1269],
        stop_at_block_size=args.stop_at_block_size,
        exclude_pad=True,
    )

    # tokens[timesteps <= prompt_age.to(device)] = 0
    # timesteps[timesteps <= prompt_age.to(device)] = -1e4
    tokens = tokens.detach().cpu().numpy()
    timesteps = timesteps.detach().cpu().numpy()
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

axs[0].hist(
    gt_n_clusters,
    bins=np.arange(gt_n_clusters.max() + 1),
    alpha=alpha,
    label="ground truth",
)
axs[0].hist(
    n_clusters, bins=np.arange(n_clusters.max() + 1), alpha=alpha, label="model"
)
axs[0].set_xlabel("# disease clusters per participant")
axs[0].legend()
axs[1].hist(
    gt_cluster_sizes,
    bins=np.arange(2, gt_cluster_sizes.max() + 1),
    alpha=alpha,
    label="ground truth",
)
axs[1].hist(
    cluster_sizes,
    bins=np.arange(2, cluster_sizes.max() + 1),
    alpha=alpha,
    label="model",
)
axs[1].legend()
axs[1].set_xlabel("size of disease clusters")

# %%
