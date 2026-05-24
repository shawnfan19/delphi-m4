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
import gzip
import pickle
import warnings
from collections import defaultdict
from typing import Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cloudpathlib import AnyPath
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

from delphi.data.ukb import Biomarker, MultimodalUKBDataset
from delphi.env import DELPHI_CKPT_DIR, DELPHI_DATA_DIR
from delphi.multimodal import Modality


def load_shap_data(filepath) -> dict:
    """Load the gzip-compressed pickle file containing SHAP results."""
    with AnyPath(filepath).open("rb") as raw, gzip.open(raw, "rb") as f:
        shap_data = pickle.load(f)
    tokenizer = shap_data["tokenizer"].copy()
    del shap_data["tokenizer"]

    return shap_data, tokenizer


# %%
def compute_feature_counts(shap_data: dict) -> Dict[str, int]:
    """
    Compute the total occurrence count for each feature across all participants.

    Args:
        shap_data: Loaded SHAP data dictionary

    Returns:
        Dictionary mapping feature names to their total occurrence counts
    """
    feature_counts = defaultdict(int)

    for participant_id, data in shap_data.items():
        for feat in data["features"]:
            feature_counts[feat] += 1

    return dict(feature_counts)


def suggest_min_samples_threshold(
    shap_data: dict, percentiles: List[int] = [5, 10, 25, 50]
) -> Dict[str, int]:
    """
    Suggest reasonable min_samples thresholds based on feature occurrence distribution.

    Args:
        shap_data: Loaded SHAP data dictionary
        percentiles: List of percentiles to compute

    Returns:
        Dictionary with percentile labels as keys and threshold values
    """
    feature_counts = compute_feature_counts(shap_data)
    counts = np.array(list(feature_counts.values()))

    suggestions = {}
    for p in percentiles:
        suggestions[f"p{p}"] = int(np.percentile(counts, p))

    suggestions["mean"] = int(np.mean(counts))
    suggestions["median"] = int(np.median(counts))
    suggestions["min"] = int(np.min(counts))
    suggestions["max"] = int(np.max(counts))

    return suggestions


def get_data_summary(shap_data: dict) -> dict:
    """Get summary statistics about the SHAP data."""
    vocab_size = None
    all_features = set()
    n_features_per_participant = []

    for participant_id, data in shap_data.items():
        if vocab_size is None:
            vocab_size = data["shap"].shape[1]
        all_features.update(data["features"])
        n_features_per_participant.append(len(data["features"]))

    feature_counts = compute_feature_counts(shap_data)

    return {
        "n_participants": len(shap_data),
        "vocab_size": vocab_size,
        "n_unique_features": len(all_features),
        "all_features": sorted(all_features),
        "avg_features_per_participant": np.mean(n_features_per_participant),
        "min_features": min(n_features_per_participant),
        "max_features": max(n_features_per_participant),
        "feature_counts": feature_counts,
        "min_samples_suggestions": suggest_min_samples_threshold(shap_data),
    }


# =============================================================================
# Task 1: For a given feature, which disease does it contribute to the most?
# =============================================================================


def compute_feature_disease_contributions(
    shap_data: dict, feature_name: str
) -> pd.DataFrame:
    """
    Aggregate SHAP values for a given feature across all participants and diseases.

    Note: No min_samples filtering applied here - user can visualize any feature.

    Returns a DataFrame with one row per disease, containing aggregated SHAP statistics.
    """
    disease_shap_values = defaultdict(list)

    for participant_id, data in shap_data.items():
        shap_values = data["shap"]  # [n_features, vocab_size]
        features = data["features"]

        for i, feat in enumerate(features):
            if feat == feature_name:
                for disease_idx in range(shap_values.shape[1]):
                    disease_shap_values[disease_idx].append(shap_values[i, disease_idx])

    if not disease_shap_values:
        raise ValueError(f"Feature '{feature_name}' not found in the data.")

    records = []
    for disease_idx, shap_vals in disease_shap_values.items():
        shap_arr = np.array(shap_vals)
        records.append(
            {
                "disease_idx": disease_idx,
                "mean_shap": np.mean(shap_arr),
                "median_shap": np.median(shap_arr),
                "std_shap": np.std(shap_arr),
                "mean_abs_shap": np.mean(np.abs(shap_arr)),
                "q25": np.percentile(shap_arr, 25),
                "q75": np.percentile(shap_arr, 75),
                "n_samples": len(shap_arr),
                "mean_odds_ratio": np.mean(np.exp(shap_arr)),
                "median_odds_ratio": np.median(np.exp(shap_arr)),
            }
        )

    return pd.DataFrame(records).sort_values("mean_abs_shap", ascending=False)


def plot_feature_to_diseases(
    shap_data: dict,
    feature_name: str,
    disease_names: Optional[Dict[int, str]] = None,
    top_k: int = 15,
    contribution_direction: Literal["all", "positive", "negative"] = "all",
    figsize: Tuple[int, int] = (12, 8),
) -> pd.DataFrame:
    """
    Visualize which diseases a given feature contributes to the most.

    Shows a single plot with median (circle) and mean (diamond) markers in
    exponentiated space (rate multiplier), with IQR bars.

    Args:
        shap_data: Loaded SHAP data dictionary
        feature_name: Feature to analyze (e.g., 'diabetes', 'WBC.rbc')
        disease_names: Optional mapping from disease index to name
        top_k: Number of top diseases to display
        contribution_direction: Filter by contribution direction
            - 'all': show all contributions (default)
            - 'positive': show only risk-increasing contributions (median_shap > 0)
            - 'negative': show only risk-decreasing contributions (median_shap < 0)
        figsize: Figure size

    Returns:
        DataFrame with disease contributions (filtered and sorted)
    """
    df = compute_feature_disease_contributions(shap_data, feature_name)

    # Apply direction filter
    if contribution_direction == "positive":
        df = df[df["median_shap"] > 0].copy()
        direction_label = " (Risk-Increasing)"
        default_color = "#d62728"  # red
    elif contribution_direction == "negative":
        df = df[df["median_shap"] < 0].copy()
        direction_label = " (Risk-Decreasing)"
        default_color = "#1f77b4"  # blue
    else:
        direction_label = ""
        default_color = None

    if df.empty:
        print(
            f"No diseases found with {contribution_direction} contributions for feature '{feature_name}'"
        )
        return pd.DataFrame()

    # Re-sort after filtering and take top_k
    df = df.sort_values("mean_abs_shap", ascending=False)
    df_top = df.head(top_k).copy()

    if len(df_top) < top_k:
        print(
            f"Note: Only {len(df_top)} diseases found with {contribution_direction} contributions"
        )

    # Add disease names
    if disease_names:
        df_top["disease_name"] = df_top["disease_idx"].map(
            lambda x: f"{disease_names.get(x, f'Disease {x}')}[{x}]"
        )
    else:
        df_top["disease_name"] = df_top["disease_idx"].apply(lambda x: f"Disease {x}")

    fig, ax = plt.subplots(figsize=figsize)

    # Determine colors
    if default_color:
        colors = [default_color] * len(df_top)
    else:
        colors = ["#d62728" if x > 0 else "#1f77b4" for x in df_top["median_shap"]]

    # Single plot: median & mean markers with IQR in exponentiated space.
    # We use exp(mean_shap) rather than mean(exp(shap)) for the mean marker.
    # Reason: exp is a convex function, so by Jensen's inequality
    # mean(exp(shap)) >= exp(mean(shap)), and mean(exp(shap)) is more sensitive
    # to outliers in the heavy-tailed exp space. exp(mean_shap) gives the
    # fold-change at the average logit, which is more robust and directly
    # comparable to exp(median_shap).
    for i, (_, row) in enumerate(df_top.iterrows()):
        color = colors[i]
        exp_median = np.exp(row["median_shap"])
        exp_mean = np.exp(row["mean_shap"])
        exp_q25 = np.exp(row["q25"])
        exp_q75 = np.exp(row["q75"])
        ax.hlines(i, exp_q25, exp_q75, colors="gray", linewidth=2, zorder=2)
        ax.plot(
            exp_median,
            i,
            "o",
            color=color,
            markersize=8,
            zorder=3,
            label="Median" if i == 0 else None,
        )
        ax.plot(
            exp_mean,
            i,
            "D",
            color=color,
            markersize=5,
            zorder=4,
            label="Mean" if i == 0 else None,
        )
    ax.axvline(x=1, color="black", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_yticks(range(len(df_top)))
    ax.set_yticklabels(df_top["disease_name"])
    ax.set_xlabel("Rate Multiplier (exp(SHAP))")
    ax.set_title(
        f'Feature: "{feature_name}" → Which diseases?{direction_label}',
        fontsize=14,
    )
    ax.legend(loc="lower right")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.show()

    return df_top


# =============================================================================
# SHAP Dependence Plot: feature value vs modality SHAP
# =============================================================================


def plot_shap_dependence(
    shap_data: dict,
    modality: str,
    disease_idx: int,
    disease_name: Optional[str] = None,
    n_cols: int = 3,
    figsize_per_subplot: Tuple[float, float] = (4, 3.5),
    sample_size: Optional[int] = 2000,
    lowess_frac: Optional[float] = None,
) -> pd.DataFrame:
    """
    SHAP dependence plot for a modality's sub-features.

    For each sub-feature in the modality's biomarker panel, plots the raw
    feature value (x) against the modality-level SHAP value for the given
    disease (y), with an optional LOWESS trend curve.

    Args:
        shap_data: Loaded SHAP data dictionary
        modality: Modality name (e.g., "LFT", "LIPID")
        disease_idx: Target disease index
        disease_name: Optional display name for the disease
        n_cols: Number of columns in subplot grid
        figsize_per_subplot: (width, height) per subplot panel
        sample_size: Max scatter points (None for all); LOWESS uses full data
        lowess_frac: LOWESS smoothing fraction (0-1), or None to disable

    Returns:
        DataFrame with pid, shap_value, and one column per sub-feature
    """
    # 1. Extract per-participant SHAP values for this modality → disease
    pid_shap = {}
    for pid, data in shap_data.items():
        features = data["features"]
        shap_values = data["shap"]
        for i, feat in enumerate(features):
            if feat == modality:
                pid_shap[pid] = float(shap_values[i, disease_idx])
                break

    if not pid_shap:
        print(f"Modality '{modality}' not found in SHAP data.")
        return pd.DataFrame()

    # 2. Load raw biomarker values
    biomarker = Biomarker(
        path=AnyPath(DELPHI_DATA_DIR) / f"ukb_real_data/biomarkers/{modality.lower()}",
        z_score=False,
    )
    shap_pids = np.array(list(pid_shap.keys()))
    bio_data, bio_subs = biomarker.to_array(shap_pids)

    # 3. Align: keep only participants present in both
    bio_shap_values = np.array([pid_shap[pid] for pid in bio_subs])

    # 4. Build DataFrame
    df = pd.DataFrame({"pid": bio_subs, "shap_value": bio_shap_values})
    for feat_name in biomarker.features:
        col_idx = biomarker.feat2idx[feat_name]
        df[feat_name] = bio_data[:, col_idx]

    if df.empty:
        print(
            f"No overlapping participants between SHAP data and {modality} biomarker."
        )
        return pd.DataFrame()

    # 5. Plot
    k = len(biomarker.features)
    n_rows = int(np.ceil(k / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_subplot[0] * n_cols, figsize_per_subplot[1] * n_rows),
        squeeze=False,
    )

    display_name = disease_name if disease_name else f"Disease {disease_idx}"

    # Subsample for scatter
    if sample_size and len(df) > sample_size:
        df_scatter = df.sample(n=sample_size, random_state=42)
    else:
        df_scatter = df

    for idx, feat_name in enumerate(biomarker.features):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        x_scatter = df_scatter[feat_name].values
        y_scatter = np.exp(df_scatter["shap_value"].values)
        ax.scatter(
            x_scatter, y_scatter, alpha=0.3, s=10, c="steelblue", rasterized=True
        )

        x_full = df[feat_name].values
        y_full = np.exp(df["shap_value"].values)

        # LOWESS on full data (optional)
        if lowess_frac is not None:
            sort_idx = np.argsort(x_full)
            smoothed = sm_lowess(y_full[sort_idx], x_full[sort_idx], frac=lowess_frac)
            ax.plot(smoothed[:, 0], smoothed[:, 1], "r-", linewidth=2)

        ax.axvline(x=np.mean(x_full), color="gray", linestyle="--", linewidth=0.8)
        ax.axhline(y=1, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel(feat_name)
        ax.set_yscale("log")
        if col == 0:
            ax.set_ylabel("Odds ratio")

    # Hide unused subplots
    for idx in range(k, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        f'{modality} → "{display_name}": SHAP Dependence (n={len(df):,})',
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()

    return df


def plot_shap_dependence_interactive(
    df: pd.DataFrame,
    modality: str,
    disease_name: Optional[str] = None,
    n_cols: int = 3,
    sample_size: Optional[int] = 2000,
    height_per_row: int = 350,
    width_per_col: int = 400,
):
    """
    Interactive (plotly) version of the SHAP dependence plot with hover PIDs.

    Args:
        df: DataFrame returned by plot_shap_dependence (pid, shap_value, features)
        modality: Modality name for the title
        disease_name: Optional display name for the disease
        n_cols: Number of columns in subplot grid
        sample_size: Max scatter points (None for all)
        height_per_row: Pixel height per subplot row
        width_per_col: Pixel width per subplot column
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    feat_cols = [c for c in df.columns if c not in ("pid", "shap_value")]
    k = len(feat_cols)
    n_rows = int(np.ceil(k / n_cols))

    if sample_size and len(df) > sample_size:
        df_plot = df.sample(n=sample_size, random_state=42)
    else:
        df_plot = df

    # Exponentiate SHAP values to odds ratios
    y_values = np.exp(df_plot["shap_value"])

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=feat_cols)

    for idx, feat_name in enumerate(feat_cols):
        row, col = divmod(idx, n_cols)
        fig.add_trace(
            go.Scatter(
                x=df_plot[feat_name],
                y=y_values,
                mode="markers",
                marker=dict(size=4, color="steelblue", opacity=0.3),
                customdata=df_plot["pid"],
                hovertemplate=(
                    f"PID: %{{customdata}}<br>"
                    f"{feat_name}: %{{x:.2f}}<br>"
                    f"OR: %{{y:.4f}}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row + 1,
            col=col + 1,
        )
        # Mean vertical line
        feat_mean = df[feat_name].mean()
        fig.add_vline(
            x=feat_mean,
            line_dash="dash",
            line_color="gray",
            line_width=0.8,
            row=row + 1,
            col=col + 1,
        )
        # y=1 reference (odds ratio = 1 means no effect)
        fig.add_hline(
            y=1,
            line_dash="dash",
            line_color="black",
            line_width=0.8,
            row=row + 1,
            col=col + 1,
        )

    display_name = disease_name if disease_name else modality
    fig.update_layout(
        title=f'{modality} → "{display_name}": SHAP Dependence (n={len(df):,})',
        height=height_per_row * n_rows,
        width=width_per_col * n_cols,
    )
    fig.show(renderer="iframe")


# =============================================================================
# Participant timeline viewer
# =============================================================================


def visualize_participant(pid: int, ds: MultimodalUKBDataset):
    """
    Print a readable chronological timeline for a single participant.

    Shows all token events (diagnoses, lifestyle, etc.) and biomarker
    measurements with their values, sorted by age.

    Args:
        pid: Participant ID
        ds: MultimodalUKBDataset instance (e.g. with val_fold subjects)
    """
    matches = np.where(ds.participants == pid)[0]
    if len(matches) == 0:
        print(f"PID {pid} not found in dataset ({len(ds.participants)} participants)")
        return
    idx = matches[0]

    x0, t0, bio_x_dict, bio_t, bio_m, x1, t1 = ds[idx]
    detok = ds.detokenizer

    events = []

    # Token events
    for token_id, timestamp in zip(x0, t0):
        name = detok.get(int(token_id), f"token_{token_id}")
        events.append(
            {
                "time_days": float(timestamp),
                "age_years": float(timestamp) / 365.25,
                "type": "token",
                "name": name,
                "details": "",
            }
        )

    # Biomarker events — track position per modality to index into bio_x_dict
    mod_counters = {}
    for timestamp, mod_val in zip(bio_t, bio_m):
        mod = Modality(int(mod_val))
        bio_arrays = bio_x_dict.get(mod)
        if bio_arrays is None:
            continue
        count = mod_counters.get(mod, 0)
        if count >= len(bio_arrays):
            continue
        feat_vec = bio_arrays[count]
        mod_counters[mod] = count + 1

        # Format feature values
        feat_names = ds.mod_ds[mod].features
        parts = [f"{fn}={feat_vec[i]:.2f}" for i, fn in enumerate(feat_names)]
        details = ", ".join(parts)

        events.append(
            {
                "time_days": float(timestamp),
                "age_years": float(timestamp) / 365.25,
                "type": "biomarker",
                "name": mod.name,
                "details": details,
            }
        )

    # Sort by time
    events.sort(key=lambda e: e["time_days"])

    # Print
    print(
        f"Timeline for PID {pid}  ({len(x0)} tokens, {len(bio_t)} biomarker measurements)"
    )
    print("-" * 100)
    print(f"{'Age (yrs)':>10}  {'Type':<10}  {'Name':<25}  Details")
    print("-" * 100)
    for e in events:
        print(
            f"{e['age_years']:>10.1f}  {e['type']:<10}  {e['name']:<25}  {e['details']}"
        )


# =============================================================================
# Task 3: Feature-disease contribution over time
# =============================================================================


def compute_feature_disease_temporal(
    shap_data: dict, feature_name: str, disease_idx: int
) -> pd.DataFrame:
    """
    Collect SHAP values for a (feature, disease) pair along with timestamps.

    Note: No min_samples filtering applied here - user can visualize any feature.
    """
    records = []

    for participant_id, data in shap_data.items():
        shap_values = data["shap"]
        features = data["features"]
        timesteps = data["timesteps"]

        timesteps = timesteps.max() - timesteps
        # assert min(timesteps) >= 0, f"{min(timesteps)}"

        for i, feat in enumerate(features):
            if feat == feature_name:
                # Convert to float64 to avoid float16 issues with pandas
                time_days = float(timesteps[i])
                shap_val = float(shap_values[i, disease_idx])

                records.append(
                    {
                        "participant_id": participant_id,
                        "time_days": time_days,
                        "time_years": time_days / 365.25,
                        "shap_value": shap_val,
                        "odds_ratio": np.exp(shap_val),
                    }
                )

    return pd.DataFrame(records)


def _compute_binned_statistics(
    df: pd.DataFrame, x_col: str, y_col: str, n_bins: int = 10
):
    """Compute binned statistics for trend visualization."""
    df_sorted = df.sort_values(x_col).copy()

    # Convert to float64 to avoid float16 issues with pandas
    x_values = df_sorted[x_col].astype(np.float64)

    df_sorted["bin"] = pd.cut(x_values, bins=n_bins, labels=False)

    binned = (
        df_sorted.groupby("bin", observed=True)
        .agg(
            {
                x_col: "mean",
                y_col: [
                    "mean",
                    "median",
                    "std",
                    lambda x: np.percentile(x, 25),
                    lambda x: np.percentile(x, 75),
                ],
            }
        )
        .dropna()
    )

    binned.columns = ["x_center", "y_mean", "y_median", "y_std", "y_q25", "y_q75"]
    return binned.reset_index(drop=True)


def plot_feature_disease_temporal(
    shap_data: dict,
    feature_name: str,
    disease_idx: int,
    disease_name: Optional[str] = None,
    time_unit: str = "years",
    n_bins: int = 10,
    figsize: Tuple[int, int] = (14, 5),
    xlim: Optional[Tuple[float, float]] = None,
    sample_size: Optional[int] = 2000,
) -> pd.DataFrame:
    """
    Visualize how a feature's contribution to a disease changes over time.

    Note: No min_samples filtering applied here - user can visualize any feature.

    Args:
        shap_data: Loaded SHAP data dictionary
        feature_name: Feature to analyze
        disease_idx: Disease index to analyze
        disease_name: Optional display name for disease
        time_unit: 'years' or 'days'
        n_bins: Number of bins for trend line
        figsize: Figure size
        xlim: Optional x-axis limits
        sample_size: Max points to plot (for performance)

    Returns:
        DataFrame with temporal data
    """
    df = compute_feature_disease_temporal(shap_data, feature_name, disease_idx)

    if df.empty:
        print(
            f"No data found for feature '{feature_name}' and disease index {disease_idx}"
        )
        return pd.DataFrame()

    display_name = disease_name if disease_name else f"Disease {disease_idx}"
    time_col = "time_years" if time_unit == "years" else "time_days"
    time_label = "Time (years)" if time_unit == "years" else "Time (days)"

    # Subsample for plotting if needed
    df_plot = (
        df.sample(n=min(sample_size, len(df)), random_state=42)
        if sample_size and len(df) > sample_size
        else df
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot 1: Scatter plot (logit-space SHAP)
    ax1 = axes[0]
    ax1.scatter(
        df_plot[time_col],
        df_plot["shap_value"],
        alpha=0.3,
        s=10,
        c="steelblue",
        rasterized=True,
    )

    # Add trend line
    if len(df) > n_bins * 2:
        binned = _compute_binned_statistics(df, time_col, "shap_value", n_bins)
        ax1.plot(
            binned["x_center"],
            binned["y_mean"],
            "r-",
            linewidth=2.5,
            label="Binned mean",
        )
        ax1.fill_between(
            binned["x_center"],
            binned["y_mean"] - binned["y_std"],
            binned["y_mean"] + binned["y_std"],
            color="red",
            alpha=0.2,
        )
        ax1.legend()

    ax1.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    ax1.set_xlabel(time_label)
    ax1.set_ylabel("SHAP Value (logit-space)")
    ax1.set_title("SHAP over Time")
    if xlim:
        ax1.set_xlim(xlim)

    # Plot 2: Scatter plot (odds ratio scale)
    ax2 = axes[1]
    ax2.scatter(
        df_plot[time_col],
        df_plot["odds_ratio"],
        alpha=0.3,
        s=10,
        c="steelblue",
        rasterized=True,
    )

    if len(df) > n_bins * 2:
        binned_or = _compute_binned_statistics(df, time_col, "odds_ratio", n_bins)
        ax2.plot(
            binned_or["x_center"],
            binned_or["y_median"],
            "r-",
            linewidth=2.5,
            label="Binned median",
        )
        ax2.legend()

    ax2.axhline(y=1, color="black", linestyle="--", linewidth=0.8)
    ax2.set_xlabel(time_label)
    ax2.set_ylabel("Odds Ratio (exp(SHAP))")
    ax2.set_yscale("log")
    ax2.set_title("Risk Multiplier over Time")
    if xlim:
        ax2.set_xlim(xlim)

    # Plot 3: Boxplot by time bins
    ax3 = axes[2]
    if len(df) > n_bins * 2:
        df_binned = df.copy()
        df_binned["time_bin"] = pd.cut(df_binned[time_col], bins=n_bins)

        bin_labels = [
            f"{interval.left:.1f}-{interval.right:.1f}"
            for interval in sorted(df_binned["time_bin"].dropna().unique())
        ]

        box_data = [
            group["odds_ratio"].values
            for name, group in df_binned.groupby("time_bin", observed=True)
        ]

        bp = ax3.boxplot(box_data, labels=bin_labels, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.6)

        ax3.axhline(y=1, color="black", linestyle="--", linewidth=0.8)
        ax3.set_yscale("log")
        ax3.set_xlabel(f"Time Bin ({time_unit})")
        ax3.set_ylabel("Odds Ratio")
        ax3.set_title("Distribution by Time Period")
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.suptitle(
        f'"{feature_name}" → "{display_name}": Temporal Dynamics\n(n={len(df):,})',
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()
    plt.show()

    return df


# =============================================================================
# Bonus: Heatmap for multiple features and diseases
# =============================================================================


def plot_feature_disease_heatmap(
    shap_data: dict,
    features: List[str],
    disease_indices: List[int],
    disease_names: Optional[Dict[int, str]] = None,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = "RdBu_r",
):
    """
    Create a heatmap showing mean SHAP values for multiple features and diseases.
    """
    matrix = np.zeros((len(features), len(disease_indices)))

    for i, feat in enumerate(features):
        try:
            df = compute_feature_disease_contributions(shap_data, feat)
            for j, disease_idx in enumerate(disease_indices):
                row = df[df["disease_idx"] == disease_idx]
                if not row.empty:
                    matrix[i, j] = row["mean_shap"].values[0]
        except ValueError:
            continue

    disease_labels = [
        disease_names.get(d, f"Disease {d}") if disease_names else f"Disease {d}"
        for d in disease_indices
    ]

    plt.figure(figsize=figsize)
    sns.heatmap(
        matrix,
        xticklabels=disease_labels,
        yticklabels=features,
        cmap=cmap,
        center=0,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Mean SHAP (logit-space)"},
    )
    plt.xlabel("Disease")
    plt.ylabel("Feature")
    plt.title("Feature → Disease SHAP Contributions")
    plt.tight_layout()
    plt.show()


# %%

# %%
shap_data, tokenizer = load_shap_data(
    AnyPath(DELPHI_CKPT_DIR) / "interpret/blood_0.1/shap_biomarkers.pickle.gz"
)

# Get summary
summary = get_data_summary(shap_data)
print("=" * 60)
print("SHAP Data Summary")
print("=" * 60)
print(f"  Participants:        {summary['n_participants']:,}")
print(f"  Vocabulary size:     {summary['vocab_size']}")
print(f"  Unique features:     {summary['n_unique_features']}")
print(f"  Avg features/person: {summary['avg_features_per_participant']:.1f}")
print(f"\nSample features: {summary['all_features'][:10]}")
print(f"\nSuggested min_samples thresholds:")
for key, value in summary["min_samples_suggestions"].items():
    print(f"  {key}: {value}")
print("=" * 60)

# %%
participants = np.array(list(shap_data.keys()))
participants.shape

# %%
detokenizer = {v: k for k, v in tokenizer.items()}


# %%
def moi_pids(modality: str, feature: str, greater: bool, cutoff: float):

    moi_ds = Biomarker(
        path=AnyPath(DELPHI_DATA_DIR) / f"ukb_real_data/biomarkers/{modality.lower()}",
        z_score=True,
    )
    moi_data, moi_subs = moi_ds.to_transformed_array(participants)
    feat_data = moi_data[:, moi_ds.feat2idx[feature]]

    if greater:
        mask = feat_data >= cutoff
    else:
        mask = feat_data < cutoff
    select_pids = np.unique(moi_subs[mask])

    return select_pids


# %%

# %%
# # Pick a feature from the data
# sample_feature = "LIPID.ldl_direct"
# print(f"analyzing feature: '{sample_feature}'")

# modality = "LIPID"
# feature = "cholesterol"
# modality = "LFT"
# feature = "total_bilirubin"
shap_feature = f"LIPID"
print(f"analyzing feature: '{shap_feature}'")
cutoff = 2
contribution_direction = "positive"

if cutoff is not None:
    greater = cutoff > 0
    select_pids = moi_pids(
        modality=shap_feature,
        feature="ldl_direct",
        greater=greater,
        cutoff=cutoff,
    )
    shap_subset = {pid: shap_data[pid] for pid in select_pids}
else:
    shap_subset = shap_data

df_feature = plot_feature_to_diseases(
    shap_subset,
    feature_name=shap_feature,
    disease_names=detokenizer,
    top_k=20,
    figsize=(12, 8),
    contribution_direction=contribution_direction,
)

# %%

# %%
disease_idx = 688

df_dep = plot_shap_dependence(
    shap_data,
    modality=shap_feature,
    disease_idx=disease_idx,
    disease_name=detokenizer[disease_idx],
)

plot_shap_dependence_interactive(
    df_dep,
    modality=shap_feature,
    disease_name=detokenizer[disease_idx],
)

# %%

# %%
val_ds = MultimodalUKBDataset(
    subject_list="participants/val_fold.bin",
    biomarkers=["lft"],
    no_event_interval=None,
    deterministic=True,
    block_size=None,
    z_score_biomarkers=False,
)

# %%
visualize_participant(pid=5076502, ds=val_ds)

# %%

# %%
