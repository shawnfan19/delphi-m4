# +
import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cloudpathlib import AnyPath
from matplotlib.patches import Patch

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


args = TaskConfig.from_cli()
ckpt_a_json = AnyPath(DELPHI_CKPT_DIR) / args.baseline_json
ckpt_b_json = AnyPath(DELPHI_CKPT_DIR) / args.json

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
    return pd.DataFrame(rows)


def auc_per_horizon(df: pd.DataFrame, sex: str) -> list[np.ndarray]:
    horizons = sorted(df["horizon"].unique())
    return [
        df.loc[(df["horizon"] == h) & (df["sex"] == sex), "auc"].dropna().to_numpy()
        for h in horizons
    ]


results_df = to_dataframe(results)
bl_results_df = to_dataframe(bl_results)
horizons = sorted(results_df["horizon"].unique())

# +
for sex in ["female", "male"]:

    _df = auc_per_horizon(results_df, sex=sex)
    _bl_df = auc_per_horizon(bl_results_df, sex=sex)

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
    ax.set_xticks(np.arange(len(horizons)) + 1, horizons)
    ax.set_xlabel("time horizon of prediction (year)")
    ax.set_ylabel("Mann-Whitney AUC")
    ax.set_title(sex)


cols = ["disease", "auc", "dis_count"]
for h in horizons:
    for sex in ["female", "male"]:
        _df = results_df.loc[
            (results_df["horizon"] == h) & (results_df["sex"] == sex), cols
        ].set_index("disease")
        _bl_df = bl_results_df.loc[
            (bl_results_df["horizon"] == h) & (bl_results_df["sex"] == sex), cols
        ].set_index("disease")
        _bl_df = _bl_df.reindex(_df.index)

        diff_df = _df.copy()
        diff_df["diff"] = _df["auc"] - _bl_df["auc"]
        diff_df = diff_df.reset_index().rename(
            columns={"disease": "key", "dis_count": "n_events"}
        )

        plot_by_chapter(
            df=diff_df,
            value_col="diff",
            ylabel="Δ AUC",
            hline=0,
            title=f"h={h}y, {sex}",
        )

plt.show()
# -
