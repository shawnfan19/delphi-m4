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
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from delphi.env import DELPHI_CKPT_DIR


def load_shap_data(filepath: str) -> dict:
    """Load the gzip-compressed pickle file containing SHAP results."""
    with gzip.open(filepath, "rb") as f:
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
    figsize: Tuple[int, int] = (14, 6),
) -> pd.DataFrame:
    """
    Visualize which diseases a given feature contributes to the most.

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

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Determine colors
    if default_color:
        colors = [default_color] * len(df_top)
    else:
        colors = ["#d62728" if x > 0 else "#1f77b4" for x in df_top["median_shap"]]

    # Plot 1: Mean absolute SHAP (importance magnitude)
    ax1 = axes[0]
    ax1.barh(range(len(df_top)), df_top["mean_abs_shap"], color=colors, alpha=0.8)
    ax1.set_yticks(range(len(df_top)))
    ax1.set_yticklabels(df_top["disease_name"])
    ax1.set_xlabel("Mean |SHAP| (logit-space)")
    ax1.set_title("Contribution Magnitude")
    ax1.invert_yaxis()

    # Plot 2: Directional mean SHAP with error bars
    ax2 = axes[1]
    ax2.barh(
        range(len(df_top)),
        df_top["mean_shap"],
        xerr=df_top["std_shap"],
        color=colors,
        alpha=0.8,
        capsize=3,
    )
    ax2.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_yticks(range(len(df_top)))
    ax2.set_yticklabels([])
    ax2.set_xlabel("Mean SHAP ± SD (logit-space)")
    ax2.set_title("Directional Contribution")
    ax2.invert_yaxis()

    # Plot 3: Median with IQR (distribution view)
    ax3 = axes[2]
    for i, (_, row) in enumerate(df_top.iterrows()):
        color = colors[i]
        ax3.plot(row["median_shap"], i, "o", color=color, markersize=8, zorder=3)
        ax3.hlines(i, row["q25"], row["q75"], colors="gray", linewidth=2, zorder=2)
    ax3.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    ax3.set_yticks(range(len(df_top)))
    ax3.set_yticklabels([])
    ax3.set_xlabel("Median SHAP [IQR]")
    ax3.set_title("Median with IQR")
    ax3.invert_yaxis()

    plt.suptitle(
        f'Feature: "{feature_name}" → Which diseases?{direction_label}',
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()
    plt.show()

    return df_top


# =============================================================================
# Task 2: For a given disease, which features are the most predictive?
# =============================================================================


def compute_disease_feature_importance(
    shap_data: dict, disease_idx: int, min_samples: int = 1
) -> pd.DataFrame:
    """
    Aggregate SHAP values for a given disease across all features and participants.

    Args:
        shap_data: Loaded SHAP data dictionary
        disease_idx: Index of the disease to analyze
        min_samples: Minimum number of occurrences for a feature to be included

    Returns a DataFrame with one row per feature, containing aggregated SHAP statistics.
    """
    feature_shap_values = defaultdict(list)

    for participant_id, data in shap_data.items():
        shap_values = data["shap"]  # [n_features, vocab_size]
        features = data["features"]

        if disease_idx >= shap_values.shape[1]:
            raise ValueError(f"Disease index {disease_idx} out of range")

        disease_shap = shap_values[:, disease_idx]

        for i, feat in enumerate(features):
            feature_shap_values[feat].append(disease_shap[i])

    records = []
    features_excluded = 0

    for feat, shap_vals in feature_shap_values.items():
        if len(shap_vals) < min_samples:
            features_excluded += 1
            continue

        shap_arr = np.array(shap_vals)
        records.append(
            {
                "feature": feat,
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

    if features_excluded > 0:
        print(
            f"Excluded {features_excluded} features with fewer than {min_samples} samples"
        )

    return pd.DataFrame(records).sort_values("mean_abs_shap", ascending=False)


def plot_disease_predictive_features(
    shap_data: dict,
    disease_idx: int,
    disease_name: Optional[str] = None,
    top_k: int = 20,
    min_samples: int = 10,
    contribution_direction: Literal["all", "positive", "negative"] = "all",
    figsize: Tuple[int, int] = (14, 8),
) -> pd.DataFrame:
    """
    Visualize which features are most predictive of a given disease.

    Args:
        shap_data: Loaded SHAP data dictionary
        disease_idx: Index of the disease to analyze
        disease_name: Optional display name for the disease
        top_k: Number of top features to display
        min_samples: Minimum samples for feature to be included
        contribution_direction: Filter by contribution direction
            - 'all': show all contributions (default)
            - 'positive': show only risk-increasing contributions (median_shap > 0)
            - 'negative': show only risk-decreasing contributions (median_shap < 0)
        figsize: Figure size

    Returns:
        DataFrame with top feature importances (filtered and sorted)
    """
    df = compute_disease_feature_importance(
        shap_data, disease_idx, min_samples=min_samples
    )

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
            f"No features found with {contribution_direction} contributions for disease index {disease_idx}"
        )
        return pd.DataFrame()

    # Re-sort after filtering and take top_k
    df = df.sort_values("mean_abs_shap", ascending=False)
    df_top = df.head(top_k).copy()

    if len(df_top) < top_k:
        print(
            f"Note: Only {len(df_top)} features found with {contribution_direction} contributions (min_samples={min_samples})"
        )

    display_name = disease_name if disease_name else f"Disease {disease_idx}"

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Determine colors
    if default_color:
        colors = [default_color] * len(df_top)
    else:
        colors = ["#d62728" if x > 0 else "#1f77b4" for x in df_top["median_shap"]]

    # Plot 1: Mean absolute SHAP (importance magnitude)
    ax1 = axes[0]
    ax1.barh(range(len(df_top)), df_top["mean_abs_shap"], color=colors, alpha=0.8)
    ax1.set_yticks(range(len(df_top)))
    ax1.set_yticklabels(df_top["feature"])
    ax1.set_xlabel("Mean |SHAP| (logit-space)")
    ax1.set_title("Feature Importance")
    ax1.invert_yaxis()

    # Plot 2: Directional contribution
    ax2 = axes[1]
    ax2.barh(
        range(len(df_top)),
        df_top["mean_shap"],
        xerr=df_top["std_shap"],
        color=colors,
        alpha=0.8,
        capsize=3,
    )
    ax2.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_yticks(range(len(df_top)))
    ax2.set_yticklabels([])
    ax2.set_xlabel("Mean SHAP ± SD")
    ax2.set_title("Directional Effect")
    ax2.invert_yaxis()

    # Plot 3: Median odds ratio (interpretable scale)
    ax3 = axes[2]
    for i, (_, row) in enumerate(df_top.iterrows()):
        color = colors[i]
        ax3.plot(row["median_odds_ratio"], i, "o", color=color, markersize=8, zorder=3)
        q25_or = np.exp(row["q25"])
        q75_or = np.exp(row["q75"])
        ax3.hlines(i, q25_or, q75_or, colors="gray", linewidth=2, zorder=2)
    ax3.axvline(x=1, color="black", linestyle="--", linewidth=0.8)
    ax3.set_xscale("log")
    ax3.set_yticks(range(len(df_top)))
    ax3.set_yticklabels([])
    ax3.set_xlabel("Median Odds Ratio [IQR]")
    ax3.set_title("Risk Multiplier")
    ax3.invert_yaxis()

    plt.suptitle(
        f'Disease: "{display_name}" ← Which features predict it?{direction_label}',
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()
    plt.show()

    return df_top


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
    Path(DELPHI_CKPT_DIR) / "bug/blood/shap.pickle.gz"
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

# %%
detokenizer = {v: k for k, v in tokenizer.items()}

# %%
# Pick a feature from the data
sample_feature = "LIPID.ldl_direct"
# sample_feature = "c25_malignant_neoplasm_of_pancreas"
print(f"Analyzing feature: '{sample_feature}'")

df_feature = plot_feature_to_diseases(
    shap_data,
    feature_name=sample_feature,
    disease_names=detokenizer,
    top_k=20,
    contribution_direction="positive",
)

# %%

# %%
disease_idx = 1269  # Change to your disease of interest
print(f"Analyzing disease index: {disease_idx}")

df_disease = plot_disease_predictive_features(
    shap_data,
    disease_idx=disease_idx,
    disease_name=None,
    top_k=20,
    min_samples=10,
    contribution_direction="positive",
)

# %%
sample_feature = "LIPID.ldl_direct"
disease_idx = 266

df_temporal = plot_feature_disease_temporal(
    shap_data,
    feature_name=sample_feature,
    disease_idx=disease_idx,
    disease_name=detokenizer[disease_idx],
    time_unit="years",
    n_bins=10,
    sample_size=2000,
)
print(f"Temporal data points: {len(df_temporal):,}")

# %%

# %%
