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
import argparse
import gzip
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

from delphi.data.ukb import Biomarker
from delphi.env import DELPHI_CKPT_DIR, DELPHI_DATA_DIR

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--saliency_path", type=str, help="Path to saliency .pkl.gz file")

if "ipykernel" in sys.modules:
    args = parser.parse_args([])
    args.saliency_path = "interpret/blood_0.1/saliency-LIPID-ckpt-ckpt.pkl.gz"  # fill in when running interactively
else:
    args = parser.parse_args()


# %%
def load_saliency(filepath: str) -> dict:
    with gzip.open(filepath, "rb") as f:
        data = pickle.load(f)
    targets = data.pop("targets")
    tokenizer = data.pop("tokenizer")
    modality = data.pop("modality")
    return data, targets, tokenizer, modality


sal_data, targets, tokenizer, modality_name = load_saliency(
    Path(DELPHI_CKPT_DIR) / args.saliency_path
)
detokenizer = {v: k for k, v in tokenizer.items()}

pids = np.array(list(sal_data.keys()))


# %%
def load_biomarker_values(modality_name: str, pids: np.ndarray):
    """Load raw (un-z-scored) biomarker values for the given participants."""
    bio = Biomarker(
        path=str(
            Path(DELPHI_DATA_DIR) / f"ukb_real_data/biomarkers/{modality_name.lower()}"
        ),
        z_score=False,
    )
    data, subs = bio.to_array(pids)
    meas_times = bio.first_occurrence_times(subs)
    return data, subs, bio.features, bio.feat2idx, bio.std, meas_times


bio_values, bio_pids, bio_features, bio_feat2idx, bio_std, bio_meas_times = (
    load_biomarker_values(modality_name, pids)
)
print(
    f"modality: {modality_name}, "
    f"n_participants: {len(pids)}, "
    f"n_features: {len(bio_features)}, "
    f"n_targets: {len(targets)}, "
    f"biomarker values loaded for {len(bio_pids)} participants"
)

# Compute time horizon: sal_timestamp (age at prediction target) - measurement time
sal_timestamps = np.array([sal_data[int(pid)]["timestamp"] for pid in bio_pids])
horizon = sal_timestamps - bio_meas_times
print(
    f"horizon (years): median={np.nanmedian(horizon):.1f}, range=[{np.nanmin(horizon):.1f}, {np.nanmax(horizon):.1f}]"
)


# %%
STRATA = [
    ("<5 yr", 0, 5, "tab:red"),
    ("5–10 yr", 5, 10, "tab:orange"),
    (">10 yr", 10, np.inf, "tab:blue"),
]


def _hazard_pct_formatter(val, pos):
    pct = (np.exp(val) - 1) * 100
    return f"{pct:+.0f}%"


def plot_saliency_vs_value(
    sal_data,
    bio_values,
    bio_pids,
    feature_names,
    bio_feat2idx,
    bio_std,
    target_idx,
    target_name=None,
    horizon=None,
    n_cols=3,
    figsize_per_subplot=(4, 3.5),
    sample_size=2000,
    lowess_frac=None,
):
    """
    For each feature in the modality, scatter biomarker value (x) vs
    saliency for a given target disease (y), with optional LOWESS trend.

    Saliency is rescaled from per-z-unit to per-raw-unit (dividing by σ_f),
    and y-axis ticks show the corresponding % change in hazard rate.

    If horizon is provided (array of years per participant), scatter points
    and LOWESS lines are stratified by time horizon using STRATA bins.
    """
    k = len(feature_names)
    n_rows = int(np.ceil(k / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_subplot[0] * n_cols, figsize_per_subplot[1] * n_rows),
        squeeze=False,
    )

    # Collect saliency values aligned with bio_pids
    n_features = len(feature_names)
    sal_list = []
    for pid in bio_pids:
        jac = sal_data[int(pid)]["jacobian"]  # (n_meas * n_features, n_targets)
        jac = jac.reshape(
            -1, n_features, jac.shape[-1]
        )  # (n_meas, n_features, n_targets)
        jac = jac.mean(axis=0)  # (n_features, n_targets) — average across measurements
        sal_list.append(jac[:, target_idx])  # (n_features,)
    sal_matrix = np.array(sal_list)  # (n_pids, n_features)

    # Build stratum masks
    if horizon is not None:
        strata_masks = []
        for label, lo, hi, color in STRATA:
            mask = (horizon >= lo) & (horizon < hi)
            strata_masks.append((label, mask, color))
    else:
        strata_masks = None

    # Subsample indices for scatter (per stratum if stratified)
    rng = np.random.default_rng(42)

    display_name = target_name if target_name else f"target {target_idx}"

    for idx, feat in enumerate(feature_names):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        feat_col = bio_feat2idx[feat]
        x = bio_values[:, feat_col]
        # Rescale from ∂logλ/∂z to ∂logλ/∂x by dividing by σ
        y = sal_matrix[:, idx].astype(np.float32) / bio_std[feat_col]

        if strata_masks is not None:
            for label, mask, color in strata_masks:
                xs, ys = x[mask], y[mask]
                n_s = int(mask.sum())
                if n_s == 0:
                    continue
                # Subsample scatter
                if sample_size and n_s > sample_size:
                    si = rng.choice(n_s, size=sample_size, replace=False)
                else:
                    si = np.arange(n_s)
                leg_label = f"{label} (n={n_s:,})" if idx == 0 else None
                ax.scatter(
                    xs[si],
                    ys[si],
                    alpha=0.1,
                    s=8,
                    c=color,
                    rasterized=True,
                    label=leg_label,
                )
                if lowess_frac is not None and n_s > 10:
                    sort_idx = np.argsort(xs)
                    smoothed = sm_lowess(ys[sort_idx], xs[sort_idx], frac=lowess_frac)
                    ax.plot(smoothed[:, 0], smoothed[:, 1], color=color, linewidth=2)
        else:
            if sample_size and len(bio_pids) > sample_size:
                scatter_idx = rng.choice(len(bio_pids), size=sample_size, replace=False)
            else:
                scatter_idx = np.arange(len(bio_pids))
            ax.scatter(
                x[scatter_idx],
                y[scatter_idx],
                alpha=0.3,
                s=10,
                c="steelblue",
                rasterized=True,
            )
            if lowess_frac is not None:
                sort_idx = np.argsort(x)
                smoothed = sm_lowess(y[sort_idx], x[sort_idx], frac=lowess_frac)
                ax.plot(smoothed[:, 0], smoothed[:, 1], "r-", linewidth=2)

        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_hazard_pct_formatter))
        ax.set_xlabel(feat)
        if col == 0:
            ax.set_ylabel("Δ hazard per unit")

    for idx in range(k, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    if strata_masks is not None:
        fig.legend(
            *axes[0, 0].get_legend_handles_labels(), loc="upper right", fontsize=9
        )

    fig.suptitle(
        f'{modality_name} → "{display_name}": Saliency vs Value (n={len(bio_pids):,})',
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()


# %%
# Example usage: pick a target disease by name or index
target_name = "i21_(acute_myocardial_infarction)"
# target_name = "i20_(angina_pectoris)"
target_idx = targets.index(target_name)

# target_idx =
target_name = targets[target_idx]
print(f"target: {target_name} (idx={target_idx})")

plot_saliency_vs_value(
    sal_data,
    bio_values,
    bio_pids,
    bio_features,
    bio_feat2idx,
    bio_std,
    target_idx=target_idx,
    target_name=target_name,
    horizon=horizon,
    lowess_frac=0.3,
)

# %%
len(sal_data)

# %%
