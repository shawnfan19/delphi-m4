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
# # Compare dynamic I/D AUC across N checkpoints
#
# Smooths AUC(t) from each parquet in `parquets` (first entry is the baseline)
# using Saha-Chaudhuri & Heagerty (uniform kernel). Produces one figure per
# disease (three subplots — either / female / male) with N curves per subplot,
# plus a summary figure averaging across diseases. The disease set is auto-
# derived as the union of diseases where any non-baseline parquet improved by
# >= `cutoff` over the baseline; can be overridden via `diseases`.

# %%
from dataclasses import dataclass, field
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cloudpathlib import AnyPath

from delphi.data.ukb import UKBReader
from delphi.env import DELPHI_CKPT_READ, DELPHI_CKPT_WRITE
from delphi.experiment import CliConfig, flexi_list

mpl.rcParams["figure.dpi"] = 300


@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    # List of parquet paths. First entry is the baseline; remaining entries
    # are comparison checkpoints. Minimum length 2.
    parquets: list[str] = field(default_factory=list)
    out_dir: str
    # Optional manual override. If None, diseases are auto-derived from the
    # parquets via cutoff + min_events. Accepts a single string, an inline
    # list, or a path to a .yaml file (normalized via flexi_list).
    diseases: Any = None
    # Optional labels for each parquet; must equal len(parquets) if provided.
    # Defaults to the parquet file stems.
    labels: None | list[str] = None
    # Saha-Chaudhuri & Heagerty h_n: half-width of uniform kernel in days.
    bandwidth: float = 1000
    n_grid: int = 20
    age_since_recruit: bool = True
    # Auto-derivation: include disease if (c_index_i - c_index_baseline) >=
    # cutoff for any non-baseline parquet i, with n_events >= min_events in
    # both baseline and i.
    cutoff: float = 0.02
    min_events: int = 50

    def __post_init__(self):
        if len(self.parquets) < 2:
            raise ValueError(
                "parquets must contain at least 2 paths (first is baseline)"
            )
        if self.labels is not None and len(self.labels) != len(self.parquets):
            raise ValueError("labels must have the same length as parquets")
        if self.diseases is not None:
            self.diseases = flexi_list(self.diseases)


args = TaskConfig.from_cli()
reader = UKBReader()


# %%
def resolve_disease(d: str) -> str:
    try:
        return reader.detokenizer[int(d)]
    except (ValueError, KeyError):
        return d


def saha_chaudhuri_heagerty(case_times, concordant, total_pairs, query_times, h_days):
    """WMR with uniform rectangular kernel; see plot/dynamic_auc.py."""
    agg = (
        pd.DataFrame(
            {
                "case_time": case_times,
                "concordant": concordant,
                "total_pairs": total_pairs,
            }
        )
        .groupby("case_time", as_index=False)
        .agg({"concordant": "sum", "total_pairs": "sum"})
    )
    t_k = agg["case_time"].to_numpy()
    A_k = (agg["concordant"] / agg["total_pairs"]).to_numpy()
    out = np.full_like(query_times, np.nan, dtype=float)
    for i, q in enumerate(query_times):
        in_window = np.abs(t_k - q) < h_days
        if in_window.any():
            out[i] = A_k[in_window].mean()
    return out


def reexpress_since_recruit(df: pd.DataFrame) -> pd.DataFrame:
    pids = df["participant_id"].unique()
    recruit_days = reader.recruitment_times(pids)
    pid_to_recruit = dict(zip(pids, recruit_days))
    df = df.assign(
        recruit_time=df["participant_id"].map(pid_to_recruit).astype("float32"),
    )
    n_before = len(df)
    df = df.dropna(subset=["recruit_time"])
    df = df.assign(case_time=df["case_time"] - df["recruit_time"])
    df = df[df["case_time"] >= 0]
    print(f"age_since_recruit: kept {len(df)}/{n_before} rows")
    return df


def load_parquet(path_str: str):
    path = AnyPath(DELPHI_CKPT_READ) / path_str
    with path.open("rb") as f:
        df = pd.read_parquet(f, engine="pyarrow")
    return path, df


# %%
paths, dfs = [], []
for p in args.parquets:
    path, df = load_parquet(p)
    if args.age_since_recruit:
        df = reexpress_since_recruit(df)
    paths.append(path)
    dfs.append(df)

labels = args.labels or [p.stem for p in paths]
# Baseline = black (emphasized via thicker line); non-baseline parquets use tab10.
_tab10 = list(mpl.colormaps["tab10"].colors)
colors = ["black"] + _tab10[: len(dfs) - 1]
linewidths = [1.8] + [1.2] * (len(dfs) - 1)

out_dir = AnyPath(DELPHI_CKPT_WRITE) / args.out_dir
out_dir.mkdir(parents=True, exist_ok=True)

suffix = "_since_recruit" if args.age_since_recruit else ""
xlabel = "Years since recruitment" if args.age_since_recruit else "Age at event (years)"


# %%
# Resolve the disease list — either auto-derive from the parquets or use the
# user-supplied diseases flag verbatim.
def per_disease_cindex(df):
    g = df.groupby("icd", observed=True).agg(
        n_events=("case_time", "size"),
        conc=("concordant", "sum"),
        tot=("total_pairs", "sum"),
    )
    g["c_index"] = g["conc"] / g["tot"]
    return g


if args.diseases is None:
    ci_per = [per_disease_cindex(df) for df in dfs]
    baseline_ci = ci_per[0]
    delta_cols = []
    for i in range(1, len(dfs)):
        shared = baseline_ci.index.intersection(ci_per[i].index)
        enough = (baseline_ci.loc[shared, "n_events"] >= args.min_events) & (
            ci_per[i].loc[shared, "n_events"] >= args.min_events
        )
        d = ci_per[i].loc[shared, "c_index"] - baseline_ci.loc[shared, "c_index"]
        d = d.where(enough)  # NaN out diseases failing min_events
        delta_cols.append(d.rename(labels[i]))
    delta_df = pd.concat(delta_cols, axis=1)
    delta_max = delta_df.max(axis=1)
    icd_list = (
        delta_max[delta_max >= args.cutoff].sort_values(ascending=False).index.tolist()
    )
    print(
        f"Auto-derived {len(icd_list)} improved diseases "
        f"(cutoff={args.cutoff}, min_events={args.min_events}, "
        f"{len(dfs) - 1} comparison parquets)"
    )
else:
    icd_list = [resolve_disease(d) for d in args.diseases]

if not icd_list:
    raise ValueError(
        "No diseases to plot — lower cutoff, lower min_events, "
        "or pass diseases explicitly"
    )

# Global query grid spanning all selected diseases across all parquets
t_min = min(df.loc[df["icd"].isin(icd_list), "case_time"].min() for df in dfs)
t_max = max(df.loc[df["icd"].isin(icd_list), "case_time"].max() for df in dfs)
query_times = np.linspace(t_min, t_max, args.n_grid)
x = query_times / 365.25

panels = [("either", None), ("female", None), ("male", None)]
agg = {sex: [[] for _ in dfs] for sex, _ in panels}

# %%
for icd in icd_list:
    subs = [df[df["icd"] == icd] for df in dfs]
    if any(len(s) == 0 for s in subs):
        missing = [labels[i] for i, s in enumerate(subs) if len(s) == 0]
        raise ValueError(f"Disease '{icd}' missing from {missing}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, (sex, _) in zip(axes, panels):
        if sex == "either":
            sub_per_p = subs
        else:
            sub_per_p = [s[s["sex"] == sex] for s in subs]
        if any(len(s) == 0 for s in sub_per_p):
            ax.set_title(f"{sex} — no data")
            continue
        for i, s in enumerate(sub_per_p):
            curve = saha_chaudhuri_heagerty(
                s["case_time"].to_numpy(),
                s["concordant"].to_numpy(),
                s["total_pairs"].to_numpy(),
                query_times,
                args.bandwidth,
            )
            agg[sex][i].append(curve)
            ax.plot(
                x,
                curve,
                color=colors[i],
                linewidth=linewidths[i],
                label=f"{labels[i]} (n={len(s)})",
            )
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_title(sex)
        ax.set_ylim(0.4, 1.0)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("I/D AUC")
    fig.suptitle(f"{icd} — Dynamic AUC (h={args.bandwidth:.0f} days)")
    fig.tight_layout()

    out_path = out_dir / f"{icd}.png"
    with out_path.open("wb") as f:
        fig.savefig(f, format="png", bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)

# %%
# Summary figure: mean AUC(t) across all diseases, per sex, per parquet
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
n_diseases = 0
for ax, (sex, _) in zip(axes, panels):
    if not agg[sex][0]:
        ax.set_title(f"{sex} — no data")
        continue
    for i in range(len(dfs)):
        stack = np.stack(agg[sex][i])
        n_diseases = max(n_diseases, stack.shape[0])
        mean_curve = np.nanmean(stack, axis=0)
        ax.plot(
            x,
            mean_curve,
            color=colors[i],
            linewidth=linewidths[i],
            label=f"{labels[i]} (mean of {stack.shape[0]})",
        )
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(sex)
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=8)

axes[0].set_ylabel("I/D AUC")
fig.suptitle(f"Mean across {n_diseases} diseases (h={args.bandwidth:.0f} days)")
fig.tight_layout()

out_path = out_dir / "summary.png"
with out_path.open("wb") as f:
    fig.savefig(f, format="png", bbox_inches="tight")
print(f"Saved {out_path}")
plt.close(fig)
