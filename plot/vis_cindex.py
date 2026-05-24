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
# # Visualize C-index per Disease for a Single Checkpoint
#
# One chapter plot per sex (either / male / female), saved next to the json.

import json

# %%
from dataclasses import dataclass
from pathlib import Path

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from delphi.data.ukb import UKBReader
from delphi.env import DELPHI_CKPT_READ as DELPHI_CKPT_DIR
from delphi.experiment import CliConfig
from delphi.plot import plot_by_chapter

mpl.rcParams["figure.dpi"] = 300


@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    json: str
    min: int = 50


args = TaskConfig.from_cli()

# %%
ckpt_json = Path(DELPHI_CKPT_DIR) / args.json
min_events = args.min

with open(ckpt_json) as f:
    data = json.load(f)
    if "config" in data.keys():
        del data["config"]


# %%
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


def build_df(data, sex_key, min_events):
    rows = []
    for key, stats in data.items():
        ci, n = _get_cindex(stats, sex_key)
        if ci is None:
            continue
        if n < min_events:
            continue
        rows.append({"key": key, "val": ci, "n_events": n})
    return pd.DataFrame(rows)


dfs = {s: build_df(data, s, min_events) for s in ("either", "male", "female")}
for s, d in dfs.items():
    print(f"{s}: {len(d)} diseases")

# %%
# Per-sex chapter plot, saved next to the json
for sex in ("either", "male", "female"):
    fig, ax = plot_by_chapter(
        dfs[sex],
        value_col="val",
        ylabel="C-index",
        hline=None,
        ylim=(0.5, 1.0),
        title=f"C-index by disease — {sex}",
    )
    fig.savefig(ckpt_json.parent / f"cindex_{sex}.png")
    plt.show()

# %%
# Summary statistics (using "either" grouping)
k = 30
df_either = dfs["either"]
print(f"\nTop {k} best diseases (c-index):")
for _, row in df_either.nlargest(k, "val")[["key", "val", "n_events"]].iterrows():
    print(f"  {row['key']}: {row['val']:.3f}  (n={int(row['n_events'])})")

print(f"\nTop {k} worst diseases (c-index):")
for _, row in df_either.nsmallest(k, "val")[["key", "val", "n_events"]].iterrows():
    print(f"  {row['key']}: {row['val']:.3f}  (n={int(row['n_events'])})")

# %%
# Per-chapter mean c-index table (using "either")
_labels_df = UKBReader.labels()
_labels_df["icd"] = _labels_df["name"].str.split().str[0].str.upper()
_icd_meta = (
    _labels_df.drop_duplicates("icd")
    .set_index("icd")[["ICD-10 Chapter (short)"]]
    .rename(columns={"ICD-10 Chapter (short)": "chapter"})
)

_df = df_either.copy()
_df["icd"] = _df["key"].map(lambda k: k.split("_")[0].upper())
_df = _df.join(_icd_meta, on="icd")
_df["chapter"] = _df["chapter"].fillna("Unknown")

chapter_table = (
    _df.groupby("chapter")
    .agg(n_diseases=("val", "size"), mean_c_index=("val", "mean"))
    .sort_values("mean_c_index", ascending=False)
)
print("\nPer-chapter mean c-index:")
print(chapter_table.to_string(float_format=lambda x: f"{x:.3f}"))
