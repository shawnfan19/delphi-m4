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
# # Compare dynamic I/D AUC across two checkpoints
#
# For each disease in `diseases`, smooth AUC(t) from `parquet_a` and `parquet_b`
# using Saha-Chaudhuri & Heagerty (uniform kernel), and plot the per-sex
# difference (curve_b - curve_a) as a single curve.

# %%
from dataclasses import dataclass
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
    parquet_a: str
    parquet_b: str
    out_dir: str
    # Accepts a single string, an inline list, or a path to a .yaml file
    # containing a list. Normalized via flexi_list in __post_init__.
    diseases: Any
    label_a: str = ""
    label_b: str = ""
    # Saha-Chaudhuri & Heagerty h_n: half-width of uniform kernel in days.
    bandwidth: float = 1000
    n_grid: int = 20
    age_since_recruit: bool = True

    def __post_init__(self):
        self.diseases = flexi_list(self.diseases)


args = TaskConfig.from_cli()
if not args.diseases:
    raise ValueError("diseases must be a non-empty list or yaml path")

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
parquet_a_path, df_a = load_parquet(args.parquet_a)
parquet_b_path, df_b = load_parquet(args.parquet_b)

if args.age_since_recruit:
    df_a = reexpress_since_recruit(df_a)
    df_b = reexpress_since_recruit(df_b)

label_a = args.label_a or parquet_a_path.stem
label_b = args.label_b or parquet_b_path.stem

out_dir = AnyPath(DELPHI_CKPT_WRITE) / args.out_dir
out_dir.mkdir(parents=True, exist_ok=True)

suffix = "_since_recruit" if args.age_since_recruit else ""
xlabel = "Years since recruitment" if args.age_since_recruit else "Age at event (years)"

# %%
for disease in args.diseases:
    icd = resolve_disease(disease)
    sub_a = df_a[df_a["icd"] == icd]
    sub_b = df_b[df_b["icd"] == icd]
    if len(sub_a) == 0 or len(sub_b) == 0:
        missing = "parquet_a" if len(sub_a) == 0 else "parquet_b"
        raise ValueError(f"Disease '{icd}' missing from {missing}")

    t_min = min(sub_a["case_time"].min(), sub_b["case_time"].min())
    t_max = max(sub_a["case_time"].max(), sub_b["case_time"].max())
    query_times = np.linspace(t_min, t_max, args.n_grid)
    x = query_times / 365.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for sex, color in [
        ("either", "black"),
        ("female", "tab:red"),
        ("male", "tab:blue"),
    ]:
        if sex == "either":
            ssA, ssB = sub_a, sub_b
        else:
            ssA = sub_a[sub_a["sex"] == sex]
            ssB = sub_b[sub_b["sex"] == sex]
        if len(ssA) == 0 or len(ssB) == 0:
            continue
        curveA = saha_chaudhuri_heagerty(
            ssA["case_time"].to_numpy(),
            ssA["concordant"].to_numpy(),
            ssA["total_pairs"].to_numpy(),
            query_times,
            args.bandwidth,
        )
        curveB = saha_chaudhuri_heagerty(
            ssB["case_time"].to_numpy(),
            ssB["concordant"].to_numpy(),
            ssB["total_pairs"].to_numpy(),
            query_times,
            args.bandwidth,
        )
        diff = curveB - curveA
        ax.plot(
            x,
            diff,
            color=color,
            label=f"{sex} (n_a={len(ssA)}, n_b={len(ssB)})",
        )

    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Δ I/D AUC ({label_b} − {label_a})")
    ax.set_title(f"{icd} — Δ AUC (h={args.bandwidth:.0f} days)")
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path = out_dir / f"{icd}.png"
    with out_path.open("wb") as f:
        fig.savefig(f, format="png", bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)
