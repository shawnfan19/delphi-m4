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

from delphi.data.auto import multimodal_reader_cls
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
    # Number of top improved/degraded diseases to print and bar-plot.
    top_k: int = 20
    # If set, save figures to results/<write>/ (by_chapter.png, topk.png).
    write: None | str = None
    # Drop cases whose case_time < reader's `<filter_after>_times(pid)`.
    # E.g., "recruitment" on UKB, "first_biomarker" on AoU. None disables.
    filter_after: None | str = None
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
# If --write is set, persist figures to results/<write>/ (repo root).
OUT_DIR = None
if args.write is not None:
    OUT_DIR = AnyPath(__file__).resolve().parents[1] / "results" / args.write
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name):
    if OUT_DIR is None:
        return
    out_path = OUT_DIR / name
    with out_path.open("wb") as f:
        fig.savefig(f, format="png", bbox_inches="tight")
    print(f"Saved {out_path}")


# %%
ReaderCls = multimodal_reader_cls()
reader = ReaderCls()


def load_parquet(path):
    with path.open("rb") as f:
        return pd.read_parquet(f, engine="pyarrow")


def apply_filter(df, name, label=""):
    if name is None:
        return df
    method = getattr(reader, f"{name}_times", None)
    if method is None:
        raise ValueError(
            f"{type(reader).__name__} doesn't support filter_after={name!r}; "
            f"expected method `{name}_times`"
        )
    pids = df["participant_id"].unique()
    cutoff_arr = method(pids)
    n_pids = len(pids)
    n_nan_pids = int(np.isnan(cutoff_arr).sum())
    n_before = len(df)

    cutoff = dict(zip(pids, cutoff_arr))
    df = df.assign(cutoff=df["participant_id"].map(cutoff).astype("float32"))
    df = df.dropna(subset=["cutoff"])
    n_after_nan = len(df)
    df = df[df["case_time"] >= df["cutoff"]].drop(columns="cutoff")
    n_after = len(df)

    prefix = f"[{label}] " if label else ""
    print(
        f"{prefix}filter_after={name!r}:\n"
        f"  participants: {n_nan_pids}/{n_pids} dropped (NaN cutoff)\n"
        f"  case rows: {n_before} -> {n_after} "
        f"({n_before - n_after_nan} NaN, {n_after_nan - n_after} pre-cutoff)"
    )
    return df


df_a = apply_filter(load_parquet(ckpt_a_path), args.filter_after, label=label_a)
df_b = apply_filter(load_parquet(ckpt_b_path), args.filter_after, label=label_b)


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
k = args.top_k
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
fig, _ = plot_by_chapter(
    df_either,
    value_col="diff",
    ylabel="Δ concordance",
    hline=0,
    ylim=(-0.1, None),
    title="C-index difference by disease",
)
save_fig(fig, "by_chapter.png")
plt.show()

# %%
# Top-k improved diseases — horizontal bar plot

_labels_df = UKBReader.labels()
_labels_df["icd"] = _labels_df["name"].str.split().str[0].str.upper()
_icd_meta = (
    _labels_df.drop_duplicates("icd")
    .set_index("icd")[["name", "color"]]
    .rename(columns={"name": "disease_name"})
)

_topk = df_either.nlargest(k, "diff").copy()
_topk["icd"] = _topk["key"].map(lambda k: k.split("_")[0].upper())
_topk = _topk.join(_icd_meta, on="icd")
_topk["disease_name"] = _topk["disease_name"].fillna(_topk["key"])
_topk["color"] = _topk["color"].fillna("#888888")
_topk = _topk.sort_values("diff", ascending=True)  # largest at top for barh

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(
    _topk["disease_name"],
    _topk["diff"],
    color=_topk["color"],
    edgecolor="white",
    linewidth=0.5,
)
for y, val in enumerate(_topk["diff"]):
    ax.text(val + 0.001, y, f"{val:+.3f}", va="center", fontsize=8)
ax.set_xlabel(f"Δ concordance")
ax.set_title(f"Top {k} improved diseases")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
save_fig(fig, "topk.png")
plt.show()

# %%
# Optionally dump the list of improved diseases (Δ c-index >= 0.02, "either" sex)
if args.improved_yaml is not None:
    # Dump most-improved first. _topk was re-sorted ascending above for the barh
    # display, so re-sort here independently — otherwise the YAML (used as the
    # ordered `events` list for compare_forecast) comes out least-improved first.
    improved = _topk.sort_values("diff", ascending=False)["key"].tolist()
    out_path = AnyPath(__file__).parent / args.improved_yaml
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.dump(improved, f)
    print(f"Wrote {len(improved)} improved diseases to {out_path}")

# %%

# %%
