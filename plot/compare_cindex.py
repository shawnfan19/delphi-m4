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
# # Compare C-index per Disease Between Two Checkpoints
#
# Scatter plot: x-axis = checkpoint A, y-axis = checkpoint B.
# Each point is one disease. Color = ICD-10 chapter. Size = log(n_events).

import json

# %%
from dataclasses import dataclass
from pathlib import Path

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from delphi.data.ukb import UKBReader
from delphi.env import DELPHI_CKPT_READ as DELPHI_CKPT_DIR
from delphi.experiment import CliConfig
from delphi.plot import plot_by_chapter

mpl.rcParams["figure.dpi"] = 300


@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    json: str
    baseline_json: str
    min: int = 50


args = TaskConfig.from_cli()

# %%
ckpt_a_json = Path(DELPHI_CKPT_DIR) / args.baseline_json
ckpt_b_json = Path(DELPHI_CKPT_DIR) / args.json

label_a = str(ckpt_a_json.parent)
label_b = str(ckpt_b_json.parent)

sex = "either"  # "female" | "male" | "either"

min_events = args.min  # drop diseases with fewer events in either checkpoint

# %%
with open(ckpt_a_json) as f:
    data_a = json.load(f)
    if "config" in data_a.keys():
        del data_a["config"]

with open(ckpt_b_json) as f:
    data_b = json.load(f)
    if "config" in data_b.keys():
        del data_b["config"]


# %%
# Collect per-disease rows for all three sex groupings
def _get_cindex(stats, sex_key):
    """Return (c_index, n_events) for a disease entry, deriving 'either' as weighted avg."""
    if sex_key != "either":
        entry = stats.get(sex_key, {})
        return entry.get("c_index"), entry.get("n_events", 0) or 0

    m = stats.get("male", {})
    f = stats.get("female", {})
    ci_m, n_m = m.get("c_index"), m.get("n_events", 0) or 0
    ci_f, n_f = f.get("c_index"), f.get("n_events", 0) or 0
    total = n_m + n_f
    if total == 0:
        return None, 0
    if ci_m is not None and ci_f is not None:
        return (ci_m * n_m + ci_f * n_f) / total, total
    if ci_m is not None:
        return ci_m, total
    if ci_f is not None:
        return ci_f, total
    return None, 0


def build_df(data_a, data_b, sex_key, min_events):
    rows = []
    for key, stats_a in data_a.items():
        if key not in data_b:
            continue
        stats_b = data_b[key]

        ci_a, n_a = _get_cindex(stats_a, sex_key)
        ci_b, n_b = _get_cindex(stats_b, sex_key)

        if ci_a is None or ci_b is None:
            continue
        if n_a < min_events or n_b < min_events:
            continue

        rows.append(
            {
                "key": key,
                "val_a": ci_a,
                "val_b": ci_b,
                "diff": ci_b - ci_a,
                "n_events": n_a,
            }
        )
    return pd.DataFrame(rows)


dfs = {s: build_df(data_a, data_b, s, min_events) for s in ("either", "male", "female")}
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
fig.suptitle(f"C-index per disease — min {min_events} events", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %%
# Summary statistics (using "either" grouping)
k = 30
df_either = dfs["either"]
delta = df_either["diff"]
print(f"Mean Δ c-index ({label_b} − {label_a}): {delta.mean():.4f}")
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
    ylabel="Δ concordance",
    hline=0,
    title="C-index difference by disease",
)
plt.show()

# %%
# Top 10 improved diseases — horizontal bar plot

_labels_df = UKBReader.labels()
_labels_df["icd"] = _labels_df["name"].str.split().str[0].str.upper()
_icd_meta = (
    _labels_df.drop_duplicates("icd")
    .set_index("icd")[["name", "color"]]
    .rename(columns={"name": "disease_name"})
)

_top10 = df_either.nlargest(20, "diff").copy()
_top10["icd"] = _top10["key"].map(lambda k: k.split("_")[0].upper())
_top10 = _top10.join(_icd_meta, on="icd")
_top10["disease_name"] = _top10["disease_name"].fillna(_top10["key"])
_top10["color"] = _top10["color"].fillna("#888888")
_top10 = _top10.sort_values("diff", ascending=True)  # largest at top for barh

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(
    _top10["disease_name"],
    _top10["diff"],
    color=_top10["color"],
    edgecolor="white",
    linewidth=0.5,
)
for y, val in enumerate(_top10["diff"]):
    ax.text(val + 0.001, y, f"{val:+.3f}", va="center", fontsize=8)
ax.set_xlabel(f"Δ C-index ({label_b} − {label_a})")
ax.set_title("Top 10 improved diseases")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
plt.show()

# %%
# import yaml
#
# with open("diseases.yaml", "w") as f:
#     yaml.dump(df_either[df_either["diff"] >= 0.02].key.tolist(), f)
# df_either[df_either["diff"] >= 0.02].key.tolist()

# %%

# %%
