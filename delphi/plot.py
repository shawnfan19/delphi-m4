"""Shared plotting utilities for Delphi."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from delphi.data.ukb import load_label_meta


def _icd_from_key(key: str) -> str:
    """Extract uppercase ICD code from JSON key like 'e11_(…)' → 'E11'."""
    return key.split("_")[0].upper()


def plot_diff_by_chapter(
    df,
    label_a: str,
    label_b: str,
    skip_chapters=("Technical", "Sex", "Smoking, Alcohol and BMI"),
    ylim=(-0.1, 0.1),
    figsize=(10, 4),
    title="Metric difference by disease",
):
    """Scatter of per-disease metric difference (B − A), grouped by ICD-10 chapter.

    Parameters
    ----------
    df : DataFrame
        Must have columns: ``key``, ``diff``, ``n_events``.
    label_a, label_b : str
        Human-readable labels for the two runs.
    skip_chapters : tuple of str
        Chapters to exclude from the plot.
    ylim : tuple or None
        Y-axis limits.
    figsize : tuple
        Figure size.
    title : str
        Plot title (mean diff is appended automatically).

    Returns
    -------
    fig, ax
    """
    # Join with label metadata to get chapter + color
    labels_df = load_label_meta()
    labels_df["icd"] = labels_df["name"].str.split().str[0].str.upper()
    icd_meta = (
        labels_df.drop_duplicates("icd")
        .set_index("icd")[["name", "ICD-10 Chapter (short)", "color"]]
        .rename(columns={"ICD-10 Chapter (short)": "chapter"})
    )

    df = df.copy()
    df["icd"] = df["key"].map(_icd_from_key)
    df = df.join(icd_meta, on="icd")
    df["chapter"] = df["chapter"].fillna("Unknown")
    df["color"] = df["color"].fillna("#888888")

    if skip_chapters:
        df = df[~df["chapter"].isin(skip_chapters)]

    # Sort by chapter (Death last), then by ICD code within each chapter
    chapter_order = sorted(c for c in df["chapter"].unique() if c != "Death") + [
        "Death"
    ]
    df["_chap_order"] = df["chapter"].map({c: i for i, c in enumerate(chapter_order)})
    df = df.sort_values(["_chap_order", "icd"]).reset_index(drop=True)
    df["x"] = np.arange(len(df))

    # size: log(n_events)^5 / 2k
    def _sz(n):
        return (np.log(np.clip(n, 1, None)) ** 5) / 2_000

    sizes = _sz(df["n_events"].values)

    fig, ax = plt.subplots(figsize=figsize)
    zorders = np.random.uniform(3, 3.5, size=len(df))
    for i in range(len(df)):
        ax.scatter(
            df["x"].iloc[i],
            df["diff"].iloc[i],
            c=[df["color"].iloc[i]],
            s=[sizes[i]],
            edgecolor="white",
            linewidth=0.3,
            zorder=zorders[i],
            clip_on=(i != len(df) - 1),
        )
    ax.axhline(0, ls="--", c="0.5", lw=0.75, zorder=5)

    # Per-chapter: mean line, alternating background
    min_half_width = 3
    chapter_mids, chapter_labels = [], []
    for num, (chapter, g) in enumerate(df.groupby("chapter", sort=False)):
        m = g["diff"].mean()
        x0, x1 = g["x"].min(), g["x"].max()
        xmid = (x0 + x1) / 2
        x0_vis = min(x0, xmid - min_half_width)
        x1_vis = max(x1, xmid + min_half_width)
        ax.hlines(m, x0_vis, x1_vis, colors="red", linewidths=1.5, zorder=5)
        chapter_mids.append(xmid)
        chapter_labels.append(chapter)

        if num % 2 == 0:
            ax.fill_between([x0_vis, x1_vis], -1, 1, color=(0.945, 0.945, 0.945))

    ax.set_xticks(chapter_mids)
    ax.set_xticklabels(chapter_labels, rotation=45, ha="right", fontsize=7)
    ax.set_xlim(-0.5, max(len(df) - 0.5, chapter_mids[-1] + min_half_width + 0.5))

    if ylim:
        ax.set_ylim(ylim)
    ax.set_ylabel(f"Δ ({label_b} − {label_a})")
    title = f"{title}\nmean diff {df['diff'].mean():.3f}"
    ax.set_title(title, y=1.15)

    # Size legend
    legend_tokens = np.array([100, 1000, 10000])
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="0.5",
            markeredgecolor="white",
            markersize=np.sqrt(_sz(t)),
            label=f"{t:,}",
        )
        for t in legend_tokens
    ]
    legend_handles.append(
        Line2D([0], [0], color="red", linewidth=1.5, label="Chapter mean")
    )
    ax.legend(
        handles=legend_handles,
        title="N events",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=7,
        title_fontsize=7,
        framealpha=0.8,
        labelspacing=1.2,
        frameon=False,
    )

    fig.tight_layout()
    ax.grid(axis="x", visible=False)
    return fig, ax
