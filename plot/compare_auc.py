# +
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from delphi.env import DELPHI_CKPT_DIR
from delphi.plot import plot_diff_by_chapter

# -


os.chdir("/hps/nobackup/birney/users/sfan/Delphi")

# +
male_only_disease_lst = "config/disease_list/ukb/male_only.yaml"
female_only_disease_lst = "config/disease_list/ukb/female_only.yaml"

male_only_diseases = []
if male_only_disease_lst is not None:
    with open(male_only_disease_lst, "r") as f:
        male_only_diseases = yaml.safe_load(f)
female_only_diseases = []
if female_only_disease_lst is not None:
    with open(female_only_disease_lst, "r") as f:
        female_only_diseases = yaml.safe_load(f)


# -


def agg_stats(
    auc_logbook: dict,
    disease_lst: list,
    male_only_diseases: None | list,
    female_only_diseases: None | list,
    age_grp: None | str = None,
):
    if age_grp is None:
        age_grp = "total"
    cnt_lst, auc_lst = list(), list()
    for disease in disease_lst:
        f_auc = auc_logbook[disease]["female"][age_grp]["auc"]
        f_cnt = auc_logbook[disease]["female"][age_grp]["dis_count"]
        m_auc = auc_logbook[disease]["male"][age_grp]["auc"]
        m_cnt = auc_logbook[disease]["male"][age_grp]["dis_count"]
        if disease in male_only_diseases:
            auc = m_auc
            cnt = m_cnt
        elif disease in female_only_diseases:
            auc = f_auc
            cnt = f_cnt
        else:
            if f_auc is None or m_auc is None:
                auc = None
            else:
                auc = (f_auc + m_auc) / 2.0
            cnt = f_cnt + m_cnt
        if auc is None:
            auc = float("nan")
        auc_lst.append(auc)
        cnt_lst.append(cnt)

    return np.array(cnt_lst), np.array(auc_lst)


# # compare two models

# +
disease_lst = "config/disease_list/ukb/all_at_least_100.yaml"
json_path = (
    Path(DELPHI_CKPT_DIR)
    / "delphi-m4/blood_seed43/auc-min_time_gap-0.01-ckpt-ckpt.json"
)
bl_json_path = (
    Path(DELPHI_CKPT_DIR)
    / "delphi-m4/baseline_seed43/auc-min_time_gap-0.01-ckpt-ckpt.json"
)

ylabel = "ZLPR"
xlabel = "Baseline"
title = "disease cluster modeling"

# +
with open(disease_lst, "r") as f:
    diseases = yaml.safe_load(f)

with open(json_path, "r") as f:
    auc_logbook = json.load(f)
n_dis, aucs = agg_stats(auc_logbook, diseases, male_only_diseases, female_only_diseases)

with open(bl_json_path, "r") as f:
    bl_auc_logbook = json.load(f)
_, bl_aucs = agg_stats(
    bl_auc_logbook, diseases, male_only_diseases, female_only_diseases
)


delta = aucs - bl_aucs
print(f"mean delta: {np.nanmean(delta)}")
print(f"# improved / # total: {np.sum(delta > 0)} / {delta.size}")

print(f"\ntop 10 diseases improved:")
delta[np.isnan(delta)] = 0
for i in np.argsort(np.array(delta))[-10:]:
    print(diseases[i])

# +
# plt.violinplot([delta])

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
plot_diff_by_chapter(df, xlabel, ylabel, title="AUC difference by disease")
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

age_grp = "total"

# +
auc_dict = dict()
for key, json_path in json_paths.items():
    with open(Path(DELPHI_CKPT_DIR) / json_path, "r") as f:
        auc_logbook = json.load(f)
    n_dis, aucs = agg_stats(
        auc_logbook, diseases, male_only_diseases, female_only_diseases, age_grp
    )
    auc_dict[key] = aucs


def remove_nan(auc_lst: list[np.ndarray]):
    return [auc_np[~np.isnan(auc_np)] for auc_np in auc_lst]


plt.violinplot(remove_nan(list(auc_dict.values())), showmeans=True)
plt.ylabel("mann-whitney aucs (sex-adjusted)")
plt.xticks(np.arange(len(auc_dict)) + 1, list(json_paths.keys()))
# -
