# +
import json
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cloudpathlib import AnyPath
from matplotlib.patches import Patch

from delphi.env import DELPHI_CKPT_READ as DELPHI_CKPT_DIR
from delphi.experiment import CliConfig, flexi_list
from delphi.plot import barh_diff, label_diseases, plot_by_chapter

plt.rcParams["figure.dpi"] = 150
# -


@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    json: str
    baseline_json: str
    min: int = 50
    # Legend labels [candidate, baseline]; default to the json / baseline_json
    # paths. (OmegaConf has no real tuple type — it coerces tuples to lists — so
    # this is a list.)
    labels: None | list = None
    # Subset of time horizons (years) to visualize, e.g. horizons=[1,5].
    # None = all horizons present in both checkpoints.
    horizons: None | list = None
    # Events (diseases) to show in the per-horizon bar plot: a single name, an
    # inline list, or a path to a .yaml file (normalized via flexi_list).
    # None = skip the bar plot.
    events: Any = None

    def __post_init__(self):
        if self.events is not None:
            self.events = flexi_list(self.events)
        if self.labels is None:
            self.labels = [self.json, self.baseline_json]
        assert len(self.labels) == 2, "labels must be [candidate, baseline]"


args = TaskConfig.from_cli()
candidate_label, baseline_label = args.labels

with (AnyPath(DELPHI_CKPT_DIR) / args.json).open("r") as f:
    results = json.load(f)  # args.json = candidate / model under study (C0)

with (AnyPath(DELPHI_CKPT_DIR) / args.baseline_json).open("r") as f:
    bl_results = json.load(f)  # args.baseline_json = baseline / reference (C1)


def to_dataframe(results: dict) -> pd.DataFrame:
    rows = []
    for horizon, per_disease in results.items():
        if not horizon.isdigit():  # skip the non-horizon "summary" (C-index) block
            continue
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
            Patch(facecolor="C0", label=candidate_label),
            Patch(facecolor="C1", label=baseline_label),
        ]
    )
    ax.set_xticks(np.arange(len(horizons)) + 1, [str(h) for h in horizons])
    ax.set_xlabel("time horizon of prediction (year)")
    ax.set_ylabel("Mann-Whitney AUC")
    ax.set_title(sex)


# C-index distribution across diseases — Harrell's vs Uno's C as two violin columns,
# candidate (C0) overlaid with baseline (C1), one figure per sex (either/female/male).
# "either" is the n_event-weighted mean of the two sexes' C-indices (mirrors the AUC
# "either"; approximate, since a true pooled-sex C-index needs the raw scores).
def cindex_dataframe(results: dict) -> pd.DataFrame:
    rows = []
    for disease, per_sex in results.get("summary", {}).items():
        for sex, s in per_sex.items():
            rows.append(
                {
                    "disease": disease,
                    "sex": sex,
                    "harrell": s["cindex_harrell"],
                    "uno": s["cindex_uno"],
                    "n_event": s["n_event"],
                }
            )
        f, m = per_sex.get("female", {}), per_sex.get("male", {})
        f_n, m_n = f.get("n_event", 0) or 0, m.get("n_event", 0) or 0
        either = {"disease": disease, "sex": "either", "n_event": f_n + m_n}
        for metric in ("harrell", "uno"):
            fv, mv = f.get(f"cindex_{metric}"), m.get(f"cindex_{metric}")
            fv = None if fv is None or np.isnan(fv) else fv
            mv = None if mv is None or np.isnan(mv) else mv
            if f_n + m_n == 0 or (fv is None and mv is None):
                either[metric] = float("nan")
            elif fv is not None and mv is not None:
                either[metric] = (fv * f_n + mv * m_n) / (f_n + m_n)
            else:
                either[metric] = fv if fv is not None else mv
        rows.append(either)
    return pd.DataFrame(rows)


def cindex_columns(df: pd.DataFrame, sex: str) -> list[np.ndarray]:
    if df.empty:
        return [np.array([]), np.array([])]
    d = df[(df["sex"] == sex) & (df["n_event"] >= args.min)]
    return [d["harrell"].dropna().to_numpy(), d["uno"].dropna().to_numpy()]


results_cidx = cindex_dataframe(results)
bl_results_cidx = cindex_dataframe(bl_results)
if not (results_cidx.empty and bl_results_cidx.empty):
    for sex in ["either", "female", "male"]:
        fig, ax = plt.subplots()
        handles = []
        for df_cidx, color, label in [
            (results_cidx, "C0", candidate_label),
            (bl_results_cidx, "C1", baseline_label),
        ]:
            cols = cindex_columns(df_cidx, sex)
            if all(len(c) >= 2 for c in cols):  # need a distribution to draw a violin
                v = ax.violinplot(cols)
                for b in v["bodies"]:
                    b.set_facecolor(color)
                    b.set_edgecolor(color)
                handles.append(Patch(facecolor=color, label=label))
        ax.axhline(0.5, color="gray", lw=0.8, ls="--")  # chance
        ax.set_xticks([1, 2], ["Harrell's C", "Uno's C"])
        ax.set_ylabel("C-index")
        ax.set_title(f"C-index across diseases ({sex}, ≥{args.min} events)")
        if handles:
            ax.legend(handles=handles)


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
# Per-horizon ΔAUC bar plot for the requested --events ("either" sex), colored
# by ICD-10 chapter (mirrors plot/compare_cindex.py). Skipped when no events.
if args.events is not None:
    for h in horizons:
        bars = diff_frame(results_df, bl_results_df, h, "either")
        bars = bars[bars["key"].isin(args.events)].dropna(subset=["diff"])
        bars = label_diseases(bars)
        # plot in --events order; barh_diff(invert=True) puts the first listed
        # event at the top (barh otherwise draws row 0 at the bottom).
        order = {e: i for i, e in enumerate(args.events)}
        bars = bars.sort_values("key", key=lambda s: s.map(order))
        fig, ax = barh_diff(
            bars,
            xlabel="Δ AUC (candidate − baseline)",
            title=f"Δ AUC for selected events — horizon={h}y",
            invert=True,
        )

plt.show()
# -
