# +
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from cloudpathlib import AnyPath

from delphi.env import DELPHI_CKPT_DIR
from delphi.plot import plot_by_chapter

# -


os.chdir("/hps/nobackup/birney/users/sfan/Delphi")


def agg_stats(
    auc_logbook: dict,
    disease_lst: list,
    aggregate: str = "uniform",
):
    cnt_lst, auc_lst = list(), list()
    for disease in disease_lst:
        age_groups = [k for k in auc_logbook[disease]["female"] if k != "total"]

        bin_aucs, bin_cnts = [], []
        for ag in age_groups:
            f_auc = auc_logbook[disease]["female"][ag]["auc"]
            f_cnt = auc_logbook[disease]["female"][ag]["dis_count"] or 0
            m_auc = auc_logbook[disease]["male"][ag]["auc"]
            m_cnt = auc_logbook[disease]["male"][ag]["dis_count"] or 0

            total = f_cnt + m_cnt
            if total == 0:
                bin_aucs.append(np.nan)
            elif f_auc is not None and m_auc is not None:
                bin_aucs.append((f_auc * f_cnt + m_auc * m_cnt) / total)
            elif f_auc is not None:
                bin_aucs.append(f_auc)
            elif m_auc is not None:
                bin_aucs.append(m_auc)
            else:
                bin_aucs.append(np.nan)
            bin_cnts.append(total)

        bin_aucs = np.array(bin_aucs)
        bin_cnts = np.array(bin_cnts)

        valid = ~np.isnan(bin_aucs)
        if not valid.any():
            auc_lst.append(float("nan"))
            cnt_lst.append(0)
            continue

        if aggregate == "uniform":
            auc = np.nanmean(bin_aucs)
        elif aggregate == "weighted":
            weights = bin_cnts[valid].astype(float)
            auc = (
                np.average(bin_aucs[valid], weights=weights)
                if weights.sum() > 0
                else float("nan")
            )
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate!r}")

        auc_lst.append(auc)
        cnt_lst.append(int(bin_cnts.sum()))

    return np.array(cnt_lst), np.array(auc_lst)


# # compare two models

# +
disease_lst = "config/disease_list/ukb/all_at_least_100.yaml"
json_path = (
    AnyPath(DELPHI_CKPT_DIR)
    / "delphi-m4/blood_seed43/auc-min_time_gap-0.01-ckpt-ckpt.json"
)
bl_json_path = (
    AnyPath(DELPHI_CKPT_DIR)
    / "delphi-m4/baseline_seed43/auc-min_time_gap-0.01-ckpt-ckpt.json"
)

ylabel = "ZLPR"
xlabel = "Baseline"
title = "disease cluster modeling"

# +
with open(disease_lst, "r") as f:
    diseases = yaml.safe_load(f)

with json_path.open("r") as f:
    auc_logbook = json.load(f)
n_dis, aucs = agg_stats(auc_logbook, diseases, "weighted")

with bl_json_path.open("r") as f:
    bl_auc_logbook = json.load(f)
_, bl_aucs = agg_stats(bl_auc_logbook, diseases, "weighted")


delta = aucs - bl_aucs
print(f"mean delta: {np.nanmean(delta)}")
print(f"# improved / # total: {np.sum(delta > 0)} / {delta.size}")

print(f"\ntop 10 diseases improved:")
delta[np.isnan(delta)] = 0
for i in np.argsort(np.array(delta))[-10:]:
    print(diseases[i])
# -


# +
color_max = np.log(np.array(n_dis)).max()
color_values = [np.log(n) / color_max for n in n_dis]
fig, ax = plt.subplots()
scatter = ax.scatter(
    x=bl_aucs,
    y=aucs,
    edgecolor="black",
    cmap="Blues",
    c=color_values,
    linewidth=0.5,
    s=100,
    alpha=0.7,
    marker="o",
    vmin=0,  # Ensure 0 maps to lightest color
    vmax=1,  # Ensure 1 maps to darkest color
)
plt.ylabel(ylabel)
plt.xlabel(xlabel)

plt.ylim(0.5, 1)
plt.xlim(0.5, 1)
sns.lineplot(
    x=[0, 1],
    y=[0, 1],
    ax=ax,
    color="red",
    linestyle="--",
    linewidth=1,
)
plt.colorbar(scatter, ax=ax, label="log(# occurrences)")
plt.title(title)
plt.show()
# -

df = pd.DataFrame({"key": diseases, "diff": aucs - bl_aucs, "n_events": n_dis})
df = df.dropna(subset=["diff"])
plot_by_chapter(
    df, value_col="diff", ylabel="Δ AUC", hline=0, title="AUC difference by disease"
)
plt.show()


# # compare a group of models

# +
disease_lst = "config/disease_list/ukb/all_at_least_25.yaml"
with open(disease_lst, "r") as f:
    diseases = yaml.safe_load(f)

json_paths = {
    "baseline": "fusion/baseline/auc-min_time_gap-0.01-ckpt-ckpt.json",
    "blood": "fusion/blood/auc-min_time_gap-0.01-ckpt-ckpt.json",
    "medications": "fusion/meds+prescriptions/auc-min_time_gap-0.01-ckpt-ckpt.json",
    "surgeries": "fusion/surgeries/auc-min_time_gap-0.01-ckpt-ckpt.json",
    "m4-lite": "fusion/m4-lite/auc-min_time_gap-0.01-ckpt-ckpt.json",
}

# +
auc_dict = dict()
for key, json_path in json_paths.items():
    with (AnyPath(DELPHI_CKPT_DIR) / json_path).open("r") as f:
        auc_logbook = json.load(f)
    n_dis, aucs = agg_stats(auc_logbook, diseases)
    auc_dict[key] = aucs


def remove_nan(auc_lst: list[np.ndarray]):
    return [auc_np[~np.isnan(auc_np)] for auc_np in auc_lst]


plt.violinplot(remove_nan(list(auc_dict.values())), showmeans=True)
plt.ylabel("mann-whitney aucs (sex-adjusted)")
plt.xticks(np.arange(len(auc_dict)) + 1, list(json_paths.keys()))
