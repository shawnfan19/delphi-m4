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

import matplotlib.pyplot as plt
import numpy as np
from cloudpathlib import AnyPath
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

from delphi.data.ukb import Biomarker
from delphi.env import DELPHI_CKPT_DIR, DELPHI_DATA_DIR
from delphi.experiment import load_ckpt


# %%
def load_ig(dirpath: str | os.PathLike):
    dirpath = AnyPath(dirpath)
    attributions = np.load(dirpath / "attributions.npy", mmap_mode="r")
    logits = np.load(dirpath / "logits.npy", mmap_mode="r")
    pids = np.load(dirpath / "pids.npy")
    features = np.load(dirpath / "features.npy")
    targets = np.load(dirpath / "targets.npy")
    return attributions, logits, pids, features, targets


def load_ckpt_meta(ckpt_path):
    model, ckpt_dict = load_ckpt(ckpt_path)
    tokenizer = ckpt_dict["tokenizer"]
    targets = model.targets
    targets = targets[targets != 1].cpu().numpy()
    data_args = ckpt_dict["data_args"]
    return tokenizer, targets, data_args


def load_biomarker(modality: str, data_args: dict) -> Biomarker:
    return Biomarker(
        path=AnyPath(DELPHI_DATA_DIR)
        / "ukb_real_data"
        / "biomarkers"
        / modality.lower(),
        stats_subjects=np.fromfile(
            AnyPath(DELPHI_DATA_DIR) / "ukb_real_data" / data_args["subject_list"],
            dtype=np.uint32,
        ),
        first_time_only=True,
        z_score=data_args.get("z_score_biomarkers", False),
    )


# %%
ckpt_path = AnyPath(DELPHI_CKPT_DIR) / "interpret/blood/ckpt.pt"
ig_dir = ckpt_path.parent / "ig-biomarker"

attrs, logits, pids, feature_names, target_ids = load_ig(ig_dir)
tokenizer, all_targets, data_args = load_ckpt_meta(ckpt_path)
detokenizer = {v: k for k, v in tokenizer.items()}
target_names = [detokenizer[int(t)] for t in target_ids]

print(
    f"attributions: {attrs.shape}, features: {len(feature_names)}, targets: {len(target_names)}"
)

# %% [markdown]
# ## Load biomarker values

# %%
biomarker_values = []
for mod_name in data_args["biomarkers"]:
    bio = load_biomarker(mod_name, data_args)
    vals, _ = bio.to_array(pids)
    biomarker_values.append(vals)

biomarker_values = np.concatenate(biomarker_values, axis=1)  # (N, n_features)
print(f"biomarker_values: {biomarker_values.shape}")

# %% [markdown]
# ## Attribution vs feature value

# %%
target_name = "e78_(disorders_of_lipoprotein_metabolism_and_other_lipidaemias)"


# %%
def plot_ig_vs_value(
    attributions,
    biomarker_values,
    feature_names,
    target_idx,
    target_name=None,
    n_cols=3,
    figsize_per_subplot=(4, 3.5),
    lowess_frac=None,
):
    import textwrap

    k = len(feature_names)
    n_rows = int(np.ceil(k / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_subplot[0] * n_cols, figsize_per_subplot[1] * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    display_name = target_name if target_name else f"target {target_idx}"

    for idx, feat in enumerate(feature_names):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        x = biomarker_values[:, idx]
        y = attributions[:, target_idx, idx].astype(np.float32)

        ax.scatter(x, y, alpha=0.2, s=10, c="steelblue", rasterized=True)

        if lowess_frac is not None:
            sort_idx = np.argsort(x)
            smoothed = sm_lowess(y[sort_idx], x[sort_idx], frac=lowess_frac)
            ax.plot(smoothed[:, 0], smoothed[:, 1], "r-", linewidth=2)

        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.set_xlabel(feat)
        ax.set_ylabel("attribution")
        ax.set_title(textwrap.fill(display_name, width=25), wrap=True, pad=15)

    for idx in range(k, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    plt.show()


# %%
tidx = target_names.index(target_name)

plot_ig_vs_value(
    attributions=attrs,
    biomarker_values=biomarker_values,
    feature_names=feature_names,
    target_name=target_name,
    target_idx=tidx,
    lowess_frac=0.3,
)

# %% [markdown]
# ## Heatmap: mean attribution (features x targets)

# %%
mean_attr = np.nanmean(np.abs(attrs), axis=0)  # (n_targets, n_features)

fig, ax = plt.subplots(
    figsize=(max(6, len(feature_names) * 0.8), max(4, len(target_names) * 0.4))
)
im = ax.imshow(mean_attr, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(len(feature_names)))
ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(target_names)))
ax.set_yticklabels(target_names, fontsize=8)
ax.set_xlabel("Feature")
ax.set_ylabel("Disease")
fig.colorbar(im, ax=ax, label="Mean |attribution|")
fig.suptitle("Integrated gradients: biomarker attributions across diseases")
plt.tight_layout()
plt.show()

# %%
