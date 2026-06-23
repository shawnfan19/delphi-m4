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

# %% [markdown]
# # C-index vs offset (prediction lead-time)
#
# One violin per offset, each the distribution of per-disease c-index across
# diseases. Files come from the SAME checkpoint scored at different `--offset`
# values in apps/c-index-m4.py. A larger offset moves the scoring time to
# `offset` years before each disease's onset, so the model must rank patients on
# earlier, less-informative history (a genuinely harder task) — each violin
# should therefore drift downward.
#
# Files are named explicitly: `ckpt_dir` + a list of `fnames` (parquet stems,
# no `.parquet`). Each file's offset is read from the run config embedded in its
# Parquet footer (apps/c-index-m4.py writes it under the `config` metadata key).
# Parquets must live under DELPHI_CKPT_READ (copy from the WRITE root if the two
# differ), matching compare_cindex.py.
#
# Reading caveats:
# - Each violin point is ONE disease's c-index, unweighted by event count — a
#   distribution over diseases, not a pooled/event-weighted c-index.
# - Diseases are the intersection across offsets (>= --min events at EVERY
#   offset): a paired comparison of the same diseases as they get harder. A
#   disease falling below --min only at a large offset is pruned from every
#   violin, which can under-state the drift; per-offset eligible counts are
#   printed so the attrition is visible.
# - "either" pools the within-female and within-male c-indices (pair-weighted),
#   not a cross-sex ranking (scoring restricts controls to the case's sex).

# %%
import json
from dataclasses import dataclass
from typing import Any

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from cloudpathlib import AnyPath

from delphi.data.auto import multimodal_reader_cls
from delphi.env import DELPHI_CKPT_READ as DELPHI_CKPT_DIR
from delphi.env import DELPHI_RESULTS_DIR
from delphi.experiment import CliConfig

mpl.rcParams["figure.dpi"] = 150


@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    # Checkpoint directory under DELPHI_CKPT_DIR (the dir holding the parquets).
    ckpt_dir: str = "delphi-m4/delphi-m4"
    # Parquet stems (filename without the `.parquet` suffix) to plot, one violin
    # each. They should be the same checkpoint scored at different offsets.
    fnames: None | list = None
    # Drop diseases with fewer than `min` case events in ANY listed offset.
    min: int = 50
    # If set, save the figure to results/<write>/cindex_offset.png (repo root).
    write: None | str = None
    # Restrict to cases occurring at/after a cutoff age (case_time >= cutoff).
    # Either a reader-method NAME whose `<name>_times(pid)` gives a per-participant
    # cutoff ("recruitment" on UKB, "first_biomarker" on AoU), OR a NUMBER read as a
    # fixed cutoff age in years applied to everyone. None disables.
    filter_after: Any = None


args = TaskConfig.from_cli()
fnames = args.fnames or ["cindex"]

# %%
ckpt_dir = AnyPath(DELPHI_CKPT_DIR) / args.ckpt_dir

OUT_DIR = None
if args.write is not None:
    OUT_DIR = AnyPath(DELPHI_RESULTS_DIR) / args.write
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name):
    if OUT_DIR is None:
        return
    out_path = OUT_DIR / name
    with out_path.open("wb") as f:
        fig.savefig(f, format="png", bbox_inches="tight")
    print(f"Saved {out_path}")


# %%
# A string filter_after needs the reader's `<name>_times` cutoff; build it once
# (this loads the full token dataset into memory). Numeric / None need no reader.
reader = multimodal_reader_cls()() if isinstance(args.filter_after, str) else None


def apply_filter(df, name, reader, label=""):
    if name is None:
        return df
    pids = df["participant_id"].unique()
    if isinstance(name, (int, float)):
        # numeric: a fixed cutoff age in years, applied to every participant
        cutoff_arr = np.full(len(pids), float(name) * 365.25, dtype=np.float32)
    else:
        # string: per-participant cutoff from the reader's `<name>_times` method
        method = getattr(reader, f"{name}_times", None)
        if method is None:
            raise ValueError(
                f"{type(reader).__name__} doesn't support filter_after={name!r}; "
                f"expected method `{name}_times`"
            )
        cutoff_arr = method(pids)
    cutoff = dict(zip(pids, cutoff_arr))
    n_before = len(df)
    df = df.assign(cutoff=df["participant_id"].map(cutoff).astype("float32"))
    df = df.dropna(subset=["cutoff"])
    df = df[df["case_time"] >= df["cutoff"]].drop(columns="cutoff")
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}filter_after={name!r}: {n_before} -> {len(df)} case rows")
    return df


# %%
def load_run(stem):
    """(offset, dataframe) for one parquet stem; offset from the embedded config."""
    path = ckpt_dir / f"{stem}.parquet"
    with path.open("rb") as f:
        table = pq.read_table(f)
    meta = table.schema.metadata or {}
    if b"config" not in meta:
        raise ValueError(
            f"{path} has no `config` metadata; was it written by c-index-m4.py?"
        )
    offset = float(json.loads(meta[b"config"])["offset"])
    return offset, apply_filter(
        table.to_pandas(), args.filter_after, reader, label=stem
    )


runs = sorted((load_run(s) for s in fnames), key=lambda r: r[0])
offsets = [o for o, _ in runs]
dfs = [df for _, df in runs]
print(f"offsets: {offsets}")


# %%
def per_disease_cindex(df, sex_key):
    """Per-disease (n_events, c_index) under a sex grouping."""
    sub = df if sex_key == "either" else df[df["sex"] == sex_key]
    g = sub.groupby("icd", observed=True).agg(
        n_events=("case_time", "size"),
        conc=("concordant", "sum"),
        tot=("total_pairs", "sum"),
    )
    g["c_index"] = g["conc"] / g["tot"]
    return g[["n_events", "c_index"]]


def violins_for_sex(sex_key):
    """One c_index array per offset over the diseases kept at EVERY offset:
    >= `min` events and a defined c_index in all runs (intersection). Prints the
    per-offset eligible count vs the kept count so cross-offset attrition shows."""
    per = [per_disease_cindex(df, sex_key) for df in dfs]
    eligible = [
        set(p.index[(p["n_events"] >= args.min) & p["c_index"].notna()]) for p in per
    ]
    keep = sorted(set.intersection(*eligible)) if eligible else []
    print(
        f"[{sex_key}] keeps {len(keep)} diseases (intersection); "
        f"per-offset eligible: {dict(zip(offsets, [len(e) for e in eligible]))}"
    )
    return [p.loc[keep, "c_index"].to_numpy() for p in per], keep


# %%
SEXES = [("either", "Either"), ("male", "Male"), ("female", "Female")]
positions = np.arange(len(offsets)) + 1  # categorical: even spacing
xticklabels = [f"{o:g}" for o in offsets]

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
for ax, (sex_key, title) in zip(axes, SEXES):
    arrays, keep = violins_for_sex(sex_key)
    if keep:
        ax.violinplot(arrays, positions=positions, showmedians=True)
        medians = [np.median(a) for a in arrays]
        ax.plot(positions, medians, color="C3", marker="o", lw=1, zorder=3)
    else:
        ax.text(0.5, 0.5, f"no diseases with >= {args.min} events", ha="center")
    ax.axhline(0.5, ls=":", c="gray", lw=1)
    ax.set_xticks(positions, xticklabels)
    ax.set_xlabel("offset / prediction lead-time (years)")
    ax.set_title(f"{title} (n={len(keep)} diseases)")
axes[0].set_ylabel("c-index")
fig.suptitle(f"C-index vs offset — {args.ckpt_dir} (>= {args.min} events)")
fig.tight_layout(rect=(0, 0, 1, 0.96))
save_fig(fig, "cindex_offset.png")
plt.show()
