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

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from delphi.env import DELPHI_CKPT_DIR
from delphi.plot import plot_diff_by_chapter

# %% [markdown]
# ## Config — edit these paths

# %%
ckpt_a_json = Path(DELPHI_CKPT_DIR) / "delphi-m4/baseline/cindex-modalities_nmr.json"
ckpt_b_json = Path(DELPHI_CKPT_DIR) / "delphi-m4/nmr/cindex-modalities_nmr.json"

# ckpt_a_json = Path(DELPHI_CKPT_DIR) / "bug_age/baseline_seed43/cindex.json"
# ckpt_b_json = Path(DELPHI_CKPT_DIR) / "bug_age/blood_seed43/cindex.json"

# ckpt_a_json = Path(DELPHI_CKPT_DIR) / "bug_age/baseline/cindex.json"
# ckpt_b_json = Path(DELPHI_CKPT_DIR) / "bug_age/blood/cindex.json"

ckpt_a_json = (
    Path(DELPHI_CKPT_DIR)
    / "delphi-m4/baseline/cindex-min_time_gap-0-max_gap-5-ckpt-ckpt.json"
)
# ckpt_b_json = Path(DELPHI_CKPT_DIR) / "delphi-m4/blood/cindex-min_time_gap-0-max_gap-5-ckpt-ckpt.json"
# ckpt_b_json = Path(DELPHI_CKPT_DIR) / "delphi-m4/urine/cindex.json"
ckpt_b_json = Path(DELPHI_CKPT_DIR) / "delphi-m4/prs/cindex.json"

label_a = "delphi-m4 (full)"
label_b = "delphi-m4 (blood)"

sex = "either"  # "female" | "male" | "either"

min_events = 100  # drop diseases with fewer events in either checkpoint

# %%
with open(ckpt_a_json) as f:
    data_a = json.load(f)

with open(ckpt_b_json) as f:
    data_b = json.load(f)


# %%
# Collect per-disease rows for all three sex groupings
def build_df(data_a, data_b, sex_key, min_events):
    rows = []
    for key, stats_a in data_a.items():
        if key not in data_b:
            continue
        stats_b = data_b[key]

        ci_a = stats_a.get(sex_key, {}).get("c_index")
        ci_b = stats_b.get(sex_key, {}).get("c_index")
        n_a = stats_a.get(sex_key, {}).get("n_events", 0) or 0
        n_b = stats_b.get(sex_key, {}).get("n_events", 0) or 0

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

print(f"\nTop {k} most degraded diseases:")
for _, row in df_either.nsmallest(k, "diff")[
    ["key", "val_a", "val_b", "diff"]
].iterrows():
    print(
        f"  {row['key']}: {row['val_a']:.3f} → {row['val_b']:.3f} (Δ {row['diff']:+.3f})"
    )


# %%
plot_diff_by_chapter(df_either, label_a, label_b, title="C-index difference by disease")
plt.show()

# %%

# %%
