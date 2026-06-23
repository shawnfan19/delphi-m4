"""Shared plotting utilities for Delphi.

Also registers a matplotlib backend that renders figures inline via the Kitty
graphics protocol. Enable with:

    export MPLBACKEND="module://delphi.plot"

Works in any terminal speaking the Kitty graphics protocol (Ghostty, Kitty,
WezTerm). Inside tmux, requires tmux >= 3.3 with `set -g allow-passthrough on`.
"""

import base64
import json
import sys
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import FigureManagerBase, _Backend
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D

from delphi.data.ukb import MultimodalUKBReader


def _emit_kitty(fig, dpi=100):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    data = base64.b64encode(buf.getvalue()).decode()
    chunk_size = 4096
    chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
    for i, chunk in enumerate(chunks):
        m = 1 if i < len(chunks) - 1 else 0
        if i == 0:
            sys.stdout.write(f"\x1b_Ga=T,f=100,m={m};{chunk}\x1b\\")
        else:
            sys.stdout.write(f"\x1b_Gm={m};{chunk}\x1b\\")
    sys.stdout.write("\n")
    sys.stdout.flush()


@_Backend.export
class _BackendKittyAgg(_Backend):
    FigureCanvas = FigureCanvasAgg
    FigureManager = FigureManagerBase

    @staticmethod
    def show(*args, **kwargs):
        for manager in Gcf.get_all_fig_managers():
            _emit_kitty(manager.canvas.figure)
        Gcf.destroy_all()


def _icd_from_key(key: str) -> str:
    """Extract uppercase ICD code from JSON key like 'e11_(…)' → 'E11'."""
    return key.split("_")[0].upper()


def label_diseases(df, key_col="key"):
    """Add ``disease_name`` and ``color`` columns from the UKB label table.

    Joins on the ICD code parsed from ``key_col`` (e.g. ``'e11_(…)' -> 'E11'``);
    unknown codes fall back to the raw key (name) and grey (color). Returns a
    copy with all original columns preserved.
    """
    labels_df = MultimodalUKBReader.labels()
    labels_df["icd"] = labels_df["name"].str.split().str[0].str.upper()
    icd_meta = (
        labels_df.drop_duplicates("icd")
        .set_index("icd")[["name", "color"]]
        .rename(columns={"name": "disease_name"})
    )
    df = df.copy()
    df["icd"] = df[key_col].map(_icd_from_key)
    df = df.join(icd_meta, on="icd")
    df["disease_name"] = df["disease_name"].fillna(df[key_col])
    df["color"] = df["color"].fillna("#888888")
    return df


def barh_diff(df, *, xlabel, title, value_col="diff", invert=False, figsize=(8, 5)):
    """Horizontal bar plot of a signed per-disease metric, colored by disease.

    ``df`` must have ``disease_name``, ``color`` and ``value_col`` columns (see
    :func:`label_diseases`) and already be in the intended row order. Bars are
    drawn bottom-up; pass ``invert=True`` to put the first row at the top. Each
    bar is annotated with its signed value. Returns ``(fig, ax)``.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(
        df["disease_name"],
        df[value_col],
        color=df["color"],
        edgecolor="white",
        linewidth=0.5,
    )
    if invert:
        ax.invert_yaxis()
    for y, val in enumerate(df[value_col]):
        ax.text(val + 0.001, y, f"{val:+.3f}", va="center", fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig, ax


def plot_by_chapter(
    df,
    value_col: str,
    ylabel: str,
    hline: float | None = 0,
    skip_chapters=("Technical", "Sex", "Smoking, Alcohol and BMI"),
    ylim=(-0.1, 0.1),
    round_to: float = 0.05,
    figsize=(10, 4),
    title="Metric by disease",
):
    """Scatter of a per-disease metric, grouped by ICD-10 chapter.

    Parameters
    ----------
    df : DataFrame
        Must have columns: ``key``, ``<value_col>``, ``n_events``.
    value_col : str
        Column in ``df`` to plot on the y-axis.
    ylabel : str
        Y-axis label.
    hline : float or None
        If not None, draws a dashed horizontal reference line at this y-value.
    skip_chapters : tuple of str
        Chapters to exclude from the plot.
    ylim : tuple or None
        Y-axis limits ``(low, high)``. ``None`` autoscales both axes; either
        element may be ``None`` to autoscale just that side, rounding the data
        extreme to a multiple of ``round_to`` (ceil on top, floor on bottom).
    round_to : float
        Granularity for autoscaled (``None``) ylim bounds.
    figsize : tuple
        Figure size.
    title : str
        Plot title.

    Returns
    -------
    fig, ax
    """
    # Join with label metadata to get chapter + color
    labels_df = MultimodalUKBReader.labels()
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
            df[value_col].iloc[i],
            c=[df["color"].iloc[i]],
            s=[sizes[i]],
            edgecolor="white",
            linewidth=0.3,
            zorder=zorders[i],
            clip_on=(i != len(df) - 1),
        )
    if hline is not None:
        ax.axhline(hline, ls="--", c="0.5", lw=0.75, zorder=5)

    # Per-chapter: mean line, alternating background
    min_half_width = 3
    chapter_mids, chapter_labels = [], []
    for num, (chapter, g) in enumerate(df.groupby("chapter", sort=False)):
        m = g[value_col].mean()
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

    if ylim is not None:
        low, high = ylim
        if low is None:
            low = np.floor(df[value_col].min() / round_to) * round_to
        if high is None:
            vmax = df[value_col].max()
            high = (
                np.ceil(vmax / round_to) * round_to if np.isfinite(vmax) else low + 0.2
            )
        high = max(high, low + round_to)  # guard against a degenerate/inverted axis
        ax.set_ylim(low, high)
    ax.set_ylabel(ylabel)
    ax.set_title(title, y=1.15)

    # Size legend
    legend_tokens = np.array([500, 2000, 10000])
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


def load_auc_json(path):
    """``(config, logbook)`` from an ``apps/auc-fast-m4.py`` JSON.

    The current format wraps output as ``{"config": ..., "logbook": ...}``; older
    files are the flat logbook dict (``icd -> sex -> age_bin -> stats``). For the
    flat format ``config`` is returned as ``{}``.
    """
    with path.open("r") as f:
        data = json.load(f)
    if "logbook" in data:
        return data.get("config", {}), data["logbook"]
    return {}, data


def load_logbook(path):
    """The bare AUC logbook (``icd -> sex -> age_bin -> {auc, ctl_count,
    dis_count}``) from an auc-fast-m4 JSON; see :func:`load_auc_json`."""
    return load_auc_json(path)[1]


def per_disease_auc(logbook, sex_key, aggregate="weighted"):
    """Per-disease ``(n_events, auc)`` collapsing the age-stratified logbook.

    For each disease, aggregate the per-age-bin AUCs into one number. "male" /
    "female" use that sex's per-bin AUC; "either" is the dis_count-weighted mean
    of the male and female AUCs within each bin (a sex-pooled AUC cannot be
    recomputed from the aggregated logbook). Bins are then combined either as a
    plain mean ("uniform") or weighted by case count ("weighted"). Diseases with
    no valid AUC under the sex grouping are dropped.
    """
    icds, n_events, aucs = [], [], []
    for disease, sexes in logbook.items():
        age_bins = [k for k in sexes["female"] if k != "total"]

        bin_aucs, bin_cnts = [], []
        for ag in age_bins:
            f_auc = sexes["female"][ag]["auc"]
            f_cnt = sexes["female"][ag]["dis_count"] or 0
            m_auc = sexes["male"][ag]["auc"]
            m_cnt = sexes["male"][ag]["dis_count"] or 0

            if sex_key == "female":
                auc, cnt = f_auc, f_cnt
            elif sex_key == "male":
                auc, cnt = m_auc, m_cnt
            else:  # "either": dis_count-weighted combine of the two sexes
                total = f_cnt + m_cnt
                if total == 0:
                    auc, cnt = None, 0
                elif f_auc is not None and m_auc is not None:
                    auc, cnt = (f_auc * f_cnt + m_auc * m_cnt) / total, total
                elif f_auc is not None:
                    auc, cnt = f_auc, f_cnt
                elif m_auc is not None:
                    auc, cnt = m_auc, m_cnt
                else:
                    auc, cnt = None, 0

            bin_aucs.append(np.nan if auc is None else auc)
            bin_cnts.append(cnt)

        bin_aucs = np.array(bin_aucs, dtype=float)
        bin_cnts = np.array(bin_cnts, dtype=float)
        valid = ~np.isnan(bin_aucs)
        if not valid.any():
            continue

        if aggregate == "uniform":
            auc = float(np.nanmean(bin_aucs))
        elif aggregate == "weighted":
            weights = bin_cnts[valid]
            auc = (
                float(np.average(bin_aucs[valid], weights=weights))
                if weights.sum() > 0
                else np.nan
            )
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate!r}")

        icds.append(disease)
        n_events.append(int(bin_cnts.sum()))
        aucs.append(auc)

    return pd.DataFrame(
        {"n_events": n_events, "auc": aucs}, index=pd.Index(icds, name="icd")
    )
