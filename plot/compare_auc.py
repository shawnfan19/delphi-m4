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
# # Compare AUC per Disease Between Two Checkpoints
#
# Scatter plot: x-axis = checkpoint A, y-axis = checkpoint B.
# Each point is one disease. Size/alpha = log(n_events).
#
# Reads the age-stratified AUC logbooks produced by `apps/auc-fast-m4.py`
# (`logbook[icd][sex][age_bin] = {"auc", "ctl_count", "dis_count"}`). Each
# disease's per-bin AUCs are collapsed into one number per sex grouping
# (`--aggregate`); "either" is the dis_count-weighted mean of the male & female
# AUCs (a pooled AUC can't be recomputed from the aggregated logbook).

# %%
from dataclasses import dataclass

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from cloudpathlib import AnyPath

from delphi.env import DELPHI_CKPT_READ as DELPHI_CKPT_DIR
from delphi.experiment import CliConfig
from delphi.plot import (
    barh_diff,
    label_diseases,
    load_logbook,
    per_disease_auc,
    plot_by_chapter,
)

mpl.rcParams["figure.dpi"] = 300


@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    auc: str
    baseline_auc: str
    min: int = 50
    # How to collapse a disease's per-age-bin AUCs into one number:
    # "weighted" (by dis_count) or "uniform" (plain mean over bins).
    aggregate: str = "weighted"
    # If set, write the list of diseases whose AUC improved by >= 0.02
    # (using "either" sex grouping) to this YAML path, relative to this script's dir.
    improved_yaml: None | str = None


args = TaskConfig.from_cli()

# %%
ckpt_a_path = AnyPath(DELPHI_CKPT_DIR) / args.baseline_auc
ckpt_b_path = AnyPath(DELPHI_CKPT_DIR) / args.auc

label_a = str(ckpt_a_path.parent)
label_b = str(ckpt_b_path.parent)

min_events = args.min  # drop diseases with fewer events in either checkpoint


# %%
logbook_a = load_logbook(ckpt_a_path)
logbook_b = load_logbook(ckpt_b_path)


# %%
def build_df(logbook_a, logbook_b, sex_key, aggregate, min_events):
    a = per_disease_auc(logbook_a, sex_key, aggregate)
    b = per_disease_auc(logbook_b, sex_key, aggregate)
    shared = a.index.intersection(b.index)
    a = a.loc[shared]
    b = b.loc[shared]
    keep = (a["n_events"] >= min_events) & (b["n_events"] >= min_events)
    a, b = a[keep], b[keep]
    return pd.DataFrame(
        {
            "key": a.index,
            "val_a": a["auc"].to_numpy(),
            "val_b": b["auc"].to_numpy(),
            "diff": (b["auc"] - a["auc"]).to_numpy(),
            "n_events": a["n_events"].to_numpy(),
        }
    )


dfs = {
    s: build_df(logbook_a, logbook_b, s, args.aggregate, min_events)
    for s in ("either", "male", "female")
}
for s, d in dfs.items():
    print(f"{s}: {len(d)} diseases")

# %%
# Basic scatter: A vs B (plain dots, no chapter coloring)
lo, hi = 0.5, 1.0

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for ax, (sex_key, sex_title) in zip(
    axes, [("either", "Either"), ("male", "Male"), ("female", "Female")]
):
    df = dfs[sex_key]
    log_n = np.log1p(df["n_events"].clip(lower=1))
    alphas = 0.15 + 0.85 * (log_n - log_n.min()) / (log_n.max() - log_n.min() + 1e-8)

    ax.scatter(
        df["val_a"],
        df["val_b"],
        c="steelblue",
        s=40,
        alpha=alphas.values,
        edgecolors="black",
        linewidths=0.5,
        zorder=2,
    )
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(label_a)
    ax.set_title(sex_title)
    ax.set_aspect("equal")

axes[0].set_ylabel(label_b)
fig.suptitle(f"AUC per disease — min {min_events} events", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %%
# Summary statistics (using "either" grouping)
k = 30
df_either = dfs["either"]
delta = df_either["diff"]
print(f"Mean Δ AUC ({label_b} − {label_a}): {delta.mean():.4f}")
print(f"Improved / total: {(delta > 0).sum()} / {len(delta)}")
print(f"\nTop {k} most improved diseases:")
for _, row in df_either.nlargest(k, "diff")[
    ["key", "val_a", "val_b", "diff"]
].iterrows():
    print(
        f"  {row['key']}: {row['val_a']:.3f} → {row['val_b']:.3f} (Δ {row['diff']:+.3f})"
    )

print(f"\nTop K most degraded diseases:")
for _, row in df_either.nsmallest(k, "diff")[
    ["key", "val_a", "val_b", "diff"]
].iterrows():
    print(
        f"  {row['key']}: {row['val_a']:.3f} → {row['val_b']:.3f} (Δ {row['diff']:+.3f})"
    )


# %%
plot_by_chapter(
    df_either,
    value_col="diff",
    ylabel="Δ AUC",
    hline=0,
    title="AUC difference by disease",
)
plt.show()

# %%
# Top improved diseases — horizontal bar plot

# largest at top for barh
_top10 = label_diseases(df_either.nlargest(20, "diff")).sort_values(
    "diff", ascending=True
)
fig, ax = barh_diff(
    _top10,
    xlabel=f"Δ AUC ({label_b} − {label_a})",
    title="Top improved diseases",
)
plt.show()

# %%
# Optionally dump the list of improved diseases (Δ AUC >= 0.02, "either" sex)
if args.improved_yaml is not None:
    improved = df_either[df_either["diff"] >= 0.02]["key"].tolist()
    out_path = AnyPath(__file__).parent / args.improved_yaml
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.dump(improved, f)
    print(f"Wrote {len(improved)} improved diseases to {out_path}")

# %%

# %%
