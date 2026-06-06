# +
import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cloudpathlib import AnyPath
from matplotlib.patches import Patch

from delphi.data.ukb import UKBReader
from delphi.env import DELPHI_CKPT_READ as DELPHI_CKPT_DIR
from delphi.experiment import CliConfig
from delphi.plot import plot_by_chapter

plt.rcParams["figure.dpi"] = 150
# -


@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    json: str
    baseline_json: str
    min: int = 50
    # Subset of time horizons (years) to visualize, e.g. horizons=[1,5].
    # None = all horizons present in both checkpoints.
    horizons: None | list = None
    # Number of most-improved diseases to show in the per-horizon bar plot.
    top_k: int = 20


args = TaskConfig.from_cli()
ckpt_a_json = AnyPath(DELPHI_CKPT_DIR) / args.json
ckpt_b_json = AnyPath(DELPHI_CKPT_DIR) / args.baseline_json

with ckpt_a_json.open("r") as f:
    bl_results = json.load(f)  # ckpt_a = args.baseline_json (reference)

with ckpt_b_json.open("r") as f:
    results = json.load(f)  # ckpt_b = args.json (model under study)


def to_dataframe(results: dict) -> pd.DataFrame:
    rows = []
    for horizon, per_disease in results.items():
        for disease, per_sex in per_disease.items():
            for sex, stats in per_sex.items():
                rows.append(
                    {
                        "horizon": int(horizon),
                        "sex": sex,
                        "disease": disease,
                        "auc": stats["auc"],
                        "ctl_count": stats["ctl_count"],
                        "dis_count": stats["dis_count"],
                    }
                )

            # synthetic "either": a sex-pooled AUC can't be recomputed from the
            # per-sex aggregates, so use the dis_count-weighted mean of the
            # female & male AUCs (matching the "either" in plot/compare_auc.py),
            # falling back to whichever sex is present when the other is missing.
            f, m = per_sex.get("female", {}), per_sex.get("male", {})
            f_auc, m_auc = f.get("auc"), m.get("auc")
            f_cnt, m_cnt = f.get("dis_count", 0) or 0, m.get("dis_count", 0) or 0
            f_auc = None if f_auc is None or np.isnan(f_auc) else f_auc
            m_auc = None if m_auc is None or np.isnan(m_auc) else m_auc
            total = f_cnt + m_cnt
            if total == 0:
                either_auc = float("nan")
            elif f_auc is not None and m_auc is not None:
                either_auc = (f_auc * f_cnt + m_auc * m_cnt) / total
            elif f_auc is not None:
                either_auc = f_auc
            elif m_auc is not None:
                either_auc = m_auc
            else:
                either_auc = float("nan")
            rows.append(
                {
                    "horizon": int(horizon),
                    "sex": "either",
                    "disease": disease,
                    "auc": either_auc,
                    "ctl_count": (f.get("ctl_count", 0) or 0)
                    + (m.get("ctl_count", 0) or 0),
                    "dis_count": total,
                }
            )
    return pd.DataFrame(rows)


def auc_per_horizon(df: pd.DataFrame, sex: str, horizons: list) -> list[np.ndarray]:
    return [
        df.loc[(df["horizon"] == h) & (df["sex"] == sex), "auc"].dropna().to_numpy()
        for h in horizons
    ]


def diff_frame(results_df, bl_results_df, h, sex) -> pd.DataFrame:
    """Per-disease candidate−baseline ΔAUC at one (horizon, sex).

    Returns a frame with columns ``key``, ``auc``, ``n_events``, ``diff`` — the
    shape both plot_by_chapter and the top-k bar plot consume.
    """
    cols = ["disease", "auc", "dis_count"]
    a = results_df.loc[
        (results_df["horizon"] == h) & (results_df["sex"] == sex), cols
    ].set_index("disease")
    b = bl_results_df.loc[
        (bl_results_df["horizon"] == h) & (bl_results_df["sex"] == sex), cols
    ].set_index("disease")
    b = b.reindex(a.index)
    out = a.copy()
    out["diff"] = a["auc"] - b["auc"]
    return out.reset_index().rename(columns={"disease": "key", "dis_count": "n_events"})


results_df = to_dataframe(results)
bl_results_df = to_dataframe(bl_results)

# Horizons plottable in both checkpoints. --horizons selects a subset (fail fast
# on any not present); default None uses all shared horizons.
available = sorted(set(results_df["horizon"]) & set(bl_results_df["horizon"]))
if args.horizons is None:
    horizons = available
else:
    requested = [int(h) for h in args.horizons]
    missing = [h for h in requested if h not in available]
    if missing:
        raise ValueError(
            f"requested horizons {missing} not in data; "
            f"available (in both checkpoints): {available}"
        )
    horizons = sorted(set(requested))

# +
for sex in ["either", "female", "male"]:

    _df = auc_per_horizon(results_df, sex=sex, horizons=horizons)
    _bl_df = auc_per_horizon(bl_results_df, sex=sex, horizons=horizons)

    fig, ax = plt.subplots()
    v1 = ax.violinplot(_df)
    v2 = ax.violinplot(_bl_df)
    for b in v1["bodies"]:
        b.set_facecolor("C0")
        b.set_edgecolor("C0")
    for b in v2["bodies"]:
        b.set_facecolor("C1")
        b.set_edgecolor("C1")
    ax.legend(
        handles=[
            Patch(facecolor="C0", label="blood"),
            Patch(facecolor="C1", label="baseline"),
        ]
    )
    ax.set_xticks(np.arange(len(horizons)) + 1, [str(h) for h in horizons])
    ax.set_xlabel("time horizon of prediction (year)")
    ax.set_ylabel("Mann-Whitney AUC")
    ax.set_title(sex)


for h in horizons:
    for sex in ["either", "female", "male"]:
        plot_by_chapter(
            df=diff_frame(results_df, bl_results_df, h, sex),
            value_col="diff",
            ylabel="Δ AUC",
            hline=0,
            title=f"horizon={h}y, {sex}",
        )

# +
# Top-k improved diseases per horizon ("either" sex) — horizontal bar plot,
# colored by ICD-10 chapter (mirrors plot/compare_cindex.py).
_labels_df = UKBReader.labels()
_labels_df["icd"] = _labels_df["name"].str.split().str[0].str.upper()
_icd_meta = (
    _labels_df.drop_duplicates("icd")
    .set_index("icd")[["name", "color"]]
    .rename(columns={"name": "disease_name"})
)

for h in horizons:
    top = (
        diff_frame(results_df, bl_results_df, h, "either")
        .nlargest(args.top_k, "diff")
        .copy()
    )
    top["icd"] = top["key"].map(lambda k: k.split("_")[0].upper())
    top = top.join(_icd_meta, on="icd")
    top["disease_name"] = top["disease_name"].fillna(top["key"])
    top["color"] = top["color"].fillna("#888888")
    top = top.sort_values("diff", ascending=True)  # largest at top for barh

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        top["disease_name"],
        top["diff"],
        color=top["color"],
        edgecolor="white",
        linewidth=0.5,
    )
    for y, val in enumerate(top["diff"]):
        ax.text(val + 0.001, y, f"{val:+.3f}", va="center", fontsize=8)
    ax.set_xlabel("Δ AUC (candidate − baseline)")
    ax.set_title(f"Top {args.top_k} improved diseases — horizon={h}y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

plt.show()
# -
