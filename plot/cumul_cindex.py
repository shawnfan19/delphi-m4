import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from delphi.data.ukb import UKBReader
from delphi.env import DELPHI_CKPT_DIR
from delphi.plot import _icd_from_key

mpl.rcParams["figure.dpi"] = 300


labels = UKBReader.labels()

json_lst = [
    Path(DELPHI_CKPT_DIR) / "m4/baseline/cindex_blood+urine+prs.json",
    Path(DELPHI_CKPT_DIR) / "m4/blood/cindex_blood+urine+prs.json",
    Path(DELPHI_CKPT_DIR) / "m4/blood+urine/cindex_blood+urine+prs.json",
    Path(DELPHI_CKPT_DIR) / "m4/blood+urine+prs/cindex_blood+urine+prs.json",
]


def get_cindex(stats, sex_key):
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


def to_df(json_path, sex="either", min_events=50):

    with open(json_path) as f:
        data = json.load(f)

    biomarkers = data["config"]["biomarkers"]
    del data["config"]

    rows = []
    for key, stats_a in data.items():
        ci_a, n_a = get_cindex(stats_a, sex)
        rows.append(
            {
                "key": key,
                "val": ci_a,
                "n_events": n_a,
            }
        )

    return pd.DataFrame(rows).sort_values(by="key").set_index("key"), biomarkers


df_lst = list()
biomarkers_lst = list()
for json_path in json_lst:
    df, biomarkers = to_df(json_path)
    df_lst.append(df)
    biomarkers_lst.append(biomarkers)

bl_df = df_lst[0]
max_df = df_lst[-1]
max_diff = max_df["val"] - bl_df["val"]
count = bl_df["n_events"]
max_diff = max_diff[(max_diff > 0.02) & (count > 100)]
max_diff = max_diff.sort_values(ascending=True)
diseases = max_diff.index.tolist()


diff_lst = list()
add_bio_lst = list()
for i in range(len(df_lst) - 1):
    diff = df_lst[i + 1].loc[diseases, "val"] - df_lst[i].loc[diseases, "val"]
    diff_lst.append(diff)
    add_bio = set(biomarkers_lst[i + 1]) - set(biomarkers_lst[i])
    add_bio_lst.append(add_bio)


fig, ax = plt.subplots(figsize=(36, 12))
left = np.zeros(len(diseases))
add_bio_lst = ["+blood", "+urine", "+prs"]
for add_bio, diff in zip(add_bio_lst, diff_lst):
    diff = diff.values
    p = ax.barh(diseases, diff, label="".join(add_bio), left=left, alpha=0.3)
    left += diff
plt.legend()
plt.show()
