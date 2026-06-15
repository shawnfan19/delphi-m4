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
# One chapter plot per sex (either / male / female), saved next to the parquet.

# %%
from dataclasses import dataclass

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from cloudpathlib import AnyPath

from delphi.data.ukb import UKBReader
from delphi.env import DELPHI_CKPT_READ, DELPHI_CKPT_WRITE
from delphi.experiment import CliConfig
from delphi.plot import plot_by_chapter

mpl.rcParams["figure.dpi"] = 300


@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    parquet: str
    min: int = 50
    # Filter out cases whose case_time < participant's recruitment time.
    after_recruit: bool = True


args = TaskConfig.from_cli()

# %%
ckpt_path = AnyPath(DELPHI_CKPT_READ) / args.parquet
out_dir = AnyPath(str(ckpt_path.parent).replace(DELPHI_CKPT_READ, DELPHI_CKPT_WRITE))
min_events = args.min

with ckpt_path.open("rb") as f:
    df = pd.read_parquet(f, engine="pyarrow")

if args.after_recruit:
    reader = UKBReader()
    pids = df["participant_id"].unique()
    recruit = dict(zip(pids, reader.recruitment_times(pids)))
    df = df.assign(
        recruit_time=df["participant_id"].map(recruit).astype("float32"),
    )
    df = df.dropna(subset=["recruit_time"])
    df = df[df["case_time"] >= df["recruit_time"]]


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


def build_df(df, sex_key, min_events):
    g = per_disease_cindex(df, sex_key)
    g = g[g["n_events"] >= min_events]
    return pd.DataFrame(
        {
            "key": g.index,
            "val": g["c_index"].to_numpy(),
            "n_events": g["n_events"].to_numpy(),
        }
    )


dfs = {s: build_df(df, s, min_events) for s in ("either", "male", "female")}
for s, d in dfs.items():
    print(f"{s}: {len(d)} diseases")

# %%
# Per-sex chapter plot, saved next to the json (under WRITE root)
out_dir.mkdir(parents=True, exist_ok=True)
for sex in ("either", "male", "female"):
    fig, ax = plot_by_chapter(
        dfs[sex],
        value_col="val",
        ylabel="C-index",
        hline=None,
        ylim=(0.5, 1.0),
        title=f"C-index by disease — {sex}",
    )
    with (out_dir / f"cindex_{sex}.png").open("wb") as f:
        fig.savefig(f, format="png")
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
    _df.groupby("chapter").agg(n_diseases=("val", "size"), mean_c_index=("val", "mean"))
    # .sort_values("mean_c_index", ascending=False)
)
print("\nPer-chapter mean c-index:")
print(chapter_table.to_string(float_format=lambda x: f"{x:.3f}"))
