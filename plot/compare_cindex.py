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
from dataclasses import dataclass

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from cloudpathlib import AnyPath

from delphi.data.ukb import UKBReader
from delphi.env import DELPHI_CKPT_READ as DELPHI_CKPT_DIR
from delphi.experiment import CliConfig
from delphi.plot import plot_by_chapter

mpl.rcParams["figure.dpi"] = 300


@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    parquet: str
    baseline_parquet: str
    min: int = 50
    # Filter out cases whose case_time < participant's recruitment time
    # (or whose recruitment time is NaN). Default on; mirrors the semantics
    # of the legacy `after_recruit` flag in apps/c-index-m4.py.
    after_recruit: bool = True
    # If set, write the list of diseases whose c-index improved by >= 0.02
    # (using "either" sex grouping) to this YAML path, relative to this script's dir.
    improved_yaml: None | str = None


args = TaskConfig.from_cli()

# %%
ckpt_a_path = AnyPath(DELPHI_CKPT_DIR) / args.baseline_parquet
ckpt_b_path = AnyPath(DELPHI_CKPT_DIR) / args.parquet

label_a = str(ckpt_a_path.parent)
label_b = str(ckpt_b_path.parent)

min_events = args.min  # drop diseases with fewer events in either checkpoint

# %%
reader = UKBReader()


def load_parquet(path):
    with path.open("rb") as f:
        df = pd.read_parquet(f, engine="pyarrow")
    if args.after_recruit:
        pids = df["participant_id"].unique()
        recruit = dict(zip(pids, reader.recruitment_times(pids)))
        df = df.assign(
            recruit_time=df["participant_id"].map(recruit).astype("float32"),
        )
        df = df.dropna(subset=["recruit_time"])
        df = df[df["case_time"] >= df["recruit_time"]]
    return df


df_a = load_parquet(ckpt_a_path)
df_b = load_parquet(ckpt_b_path)


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


def build_df(df_a, df_b, sex_key, min_events):
    a = per_disease_cindex(df_a, sex_key)
    b = per_disease_cindex(df_b, sex_key)
    shared = a.index.intersection(b.index)
    a = a.loc[shared]
    b = b.loc[shared]
    keep = (a["n_events"] >= min_events) & (b["n_events"] >= min_events)
    a, b = a[keep], b[keep]
    return pd.DataFrame(
        {
            "key": a.index,
            "val_a": a["c_index"].to_numpy(),
            "val_b": b["c_index"].to_numpy(),
            "diff": (b["c_index"] - a["c_index"]).to_numpy(),
            "n_events": a["n_events"].to_numpy(),
        }
    )


dfs = {s: build_df(df_a, df_b, s, min_events) for s in ("either", "male", "female")}
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
# Optionally dump the list of improved diseases (Δ c-index >= 0.02, "either" sex)
if args.improved_yaml is not None:
    improved = df_either[df_either["diff"] >= 0.02]["key"].tolist()
    out_path = AnyPath(__file__).parent / args.improved_yaml
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.dump(improved, f)
    print(f"Wrote {len(improved)} improved diseases to {out_path}")

# %%

# %%
