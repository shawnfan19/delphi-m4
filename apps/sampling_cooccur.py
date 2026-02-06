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
    args.ckpt = "cluster/chain/ckpt.pt"
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
# whitelist = torch.from_numpy(
#     np.concatenate((np.array([0, 1]), ds.sex_tokens, ds.lifestyle_tokens))
# )
# def pack_clusters(tokens, timesteps, dx_token = 1):
#     batch_size = tokens.shape[0]
#     is_dx_token = tokens == dx_token
#     if dx_token == 1:
#         prev_token = torch.cat(
#             (torch.full((batch_size, 1), fill_value=0), tokens[:, :-1]),
#             dim=1
#         )
#         is_dx_token = torch.logical_and(is_dx_token, ~torch.isin(prev_token, whitelist))
#     to_pack = ~torch.isin(tokens, whitelist)
#     timesteps = backward_fill(timesteps, to_pack, dim=1)

#     tokens[is_dx_token] = 0
#     timesteps[is_dx_token] = -1e4

#     sort_by_age = torch.argsort(timesteps, dim=1)
#     timesteps = torch.take_along_dim(input=timesteps, indices=sort_by_age, dim=1)
#     tokens = torch.take_along_dim(input=tokens, indices=sort_by_age, dim=1)

#     return tokens, timesteps


whitelist = np.concatenate((np.array([0, 1]), ds.sex_tokens, ds.lifestyle_tokens))


def pack_clusters(tokens, timesteps, dx_token=1):
    batch_size = tokens.shape[0]
    is_dx_token = tokens == dx_token
    if dx_token == 1:
        prev_token = np.concatenate(
            (np.full((batch_size, 1), fill_value=0), tokens[:, :-1]), axis=1
        )
        is_dx_token = np.logical_and(is_dx_token, ~np.isin(prev_token, whitelist))
    to_pack = ~np.isin(tokens, whitelist)
    timesteps = backward_fill(timesteps, to_pack, axis=1)

    tokens[is_dx_token] = 0
    timesteps[is_dx_token] = -1e4

    sort_by_age = np.argsort(timesteps, axis=1)
    timesteps = np.take_along_axis(timesteps, sort_by_age, axis=1)
    tokens = np.take_along_axis(tokens, sort_by_age, axis=1)

    return tokens, timesteps


def backward_fill(t, mask, axis=-1):
    """
    Args:
        t: Data array (e.g. shape [Batch, Time])
        mask: Boolean mask, True indicates missing value
        axis: The axis to fill along (default -1)
    """
    idx_len = t.shape[axis]

    # 1. Create indices [0, 1, ... L-1]
    # We reshape it so it broadcasts against t (e.g. shape [1, L] for 2D)
    idx = np.arange(idx_len)
    shape_view = [1] * t.ndim
    shape_view[axis] = idx_len
    idx = idx.reshape(shape_view)

    # 2. Fill masked areas with the LAST index (L-1)
    # This prepares the array for minimum accumulation from right-to-left
    val_idx = np.where(~mask, idx, idx_len - 1)

    # 3. Propagate indices backwards
    # NumPy accumulate works left-to-right, so we:
    # Flip -> Accumulate Minimum -> Flip Back
    val_idx_flipped = np.flip(val_idx, axis=axis)
    bfill_idx_flipped = np.minimum.accumulate(val_idx_flipped, axis=axis)
    bfill_idx = np.flip(bfill_idx_flipped, axis=axis)

    # 4. Use take_along_axis to fetch values
    # t[bfill_idx] would not work correctly in 2D+
    return np.take_along_axis(t, bfill_idx, axis=axis)


# def backward_fill(timestep, mask, dim=-1):
#     """
#     Args:
#         timestep: The data tensor (e.g., shape [Batch, Time])
#         mask: Boolean tensor, True indicates missing value
#         dim: The dimension to fill along (default last dim)
#     """
#     l = timestep.shape[dim]

#     # 1. Create indices [0, 1, ... L-1]
#     # We unsqueeze/view to make it broadcastable (e.g. [1, L] for 2D)
#     idx = torch.arange(l, device=timestep.device)
#     shape_view = [1] * timestep.ndim
#     shape_view[dim] = l
#     idx = idx.view(shape_view)

#     # 2. Fill masked areas with the LAST index (L-1)
#     # This prepares the tensor for a minimum accumulation
#     val_idx = torch.where(~mask, idx, torch.tensor(l - 1, device=timestep.device))

#     # 3. Propagate indices backwards
#     # PyTorch doesn't have a "reverse accumulate", so we flip, cummin, then flip back.
#     val_idx_flipped = torch.flip(val_idx, dims=[dim])
#     bfill_idx_flipped = torch.cummin(val_idx_flipped, dim=dim).values
#     bfill_idx = torch.flip(bfill_idx_flipped, dims=[dim])

#     # 4. Use gather to fetch values based on the calculated indices
#     return torch.gather(timestep, dim, bfill_idx)


# %%
def cut_prompt(
    idx: torch.Tensor,
    age: torch.Tensor,
    prompt_age: None | float | torch.Tensor,
    prompt_token: None | torch.Tensor,
    append_no_event: bool,
):

    idx = idx.clone()
    age = age.clone()

    if prompt_age is None:
        assert prompt_token is not None
        is_prompt = torch.isin(idx, prompt_token)
        assert is_prompt.any(dim=1).all(), "found sequences with no prompt_token(s)"
        prompt_age = age.clone()
        prompt_age[~is_prompt] = -10000
        prompt_age = prompt_age.max(dim=1, keepdim=True)[0]

    idx[age > prompt_age] = 0
    age[age > prompt_age] = -10000.0

    if append_no_event:
        idx = torch.nn.functional.pad(idx, (0, 1), "constant", 1)
        age = torch.cat((age, age.max(dim=1, keepdim=True)[0]), dim=1)

    age_sort = age.argsort(1)
    idx = idx.gather(1, age_sort)
    age = age.gather(1, age_sort)

    trim_margin = torch.min(torch.sum(idx == 0, dim=1)).item()
    idx, age = idx[:, trim_margin:], age[:, trim_margin:]

    return idx, age, prompt_age


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
    if data_args["break_clusters"]:
        X1_np, T1_np = pack_clusters(X1_np, T1_np, dx_token=1)
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
        max_new_tokens=args.max_new_tokens,
        termination_tokens=[1269],
        stop_at_block_size=args.stop_at_block_size,
        exclude_pad=True,
    )

    tokens = tokens.detach().cpu().numpy()
    timesteps = timesteps.detach().cpu().numpy()
    tokens[timesteps <= cutoff] = 0
    timesteps[timesteps <= cutoff] = -1e4
    if data_args["break_clusters"]:
        tokens, timesteps = pack_clusters(tokens, timesteps, dx_token=1)
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
