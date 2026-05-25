# !pip install db-dtypes
import os

import db_dtypes  # registers the 'dbdate' dtype with pandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from cloudpathlib import AnyPath as Path

from delphi.env import DELPHI_DATA_DIR

CWD = Path(os.getcwd())
CWD

with open(CWD.parent / "biomarker.yaml", "r") as f:
    _biomarker_dict = yaml.safe_load(f)
biomarker_dict = dict()
for biomarker, info in _biomarker_dict.items():
    if "aou" in info.keys():
        biomarker_dict[biomarker] = info["aou"]
biomarker_dict

with open(CWD.parent / "panel" / "aou.yaml", "r") as f:
    panels = yaml.safe_load(f)
panels

# +
# panels = {'lft_panel': ['alanine_aminotransferase',
#   'aspartate_aminotransferase',
#   'alkaline_phosphatase',
#   'total_bilirubin',
#   'direct_bilirubin',
#   'albumin',
#   'total_protein']
#          }
# panels
# -

odir = Path(DELPHI_DATA_DIR) / f"aou_uk/biomarker_stats"
rows = []
seen = set()
for panel_name, biomarkers in panels.items():
    data_dir = Path(DELPHI_DATA_DIR) / f"aou_uk/biomarkers/{panel_name}"
    df = pd.read_parquet(data_dir / "data.parquet")
    n_raw = df.shape[0]
    df = df.dropna(subset=biomarkers)
    n_complete = df.shape[0]
    print(panel_name)
    print(f"n_raw: {n_raw}")
    print(f"n_complete: {n_complete}; n_participants: {len(df['person_id'].unique())}")

    for biomarker in biomarkers:
        low, high = biomarker_dict[biomarker]["range"]
        df.loc[(df[biomarker] < low) | (df[biomarker] > high), biomarker] = np.nan

        if biomarker not in seen:
            seen.add(biomarker)
            values = df[biomarker].dropna().to_numpy()
            rows.append(
                {
                    "biomarker": biomarker,
                    "n": int(len(values)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )

        bins = 50

        fig, (ax_hist, ax_box) = plt.subplots(
            2, 1, sharex=False, gridspec_kw={"height_ratios": [4, 1]}, figsize=(6, 6)
        )

        unit_ids = list(biomarker_dict[biomarker]["unit"].keys())

        ax_hist.hist(df[biomarker].dropna(), bins=bins, alpha=0.3)
        for unit_id in unit_ids:
            ax_hist.hist(
                df.loc[df[f"{biomarker}_unit_id"] == unit_id, biomarker].dropna(),
                bins=bins,
                alpha=0.3,
                label=unit_id,
            )
        ax_hist.set_yscale("log")
        ax_hist.set_ylabel("Count (log)")
        ax_hist.legend()
        # Boxplot under it, horizontal so they share the value axis
        ax_box.boxplot(
            df[biomarker].dropna(), vert=False, showmeans=True, showfliers=False
        )
        ax_box.set_xlabel(biomarker)
        fig.suptitle(biomarker)

        fig.tight_layout()
        with (odir / f"{biomarker}.png").open("wb") as f:
            plt.savefig(f, format="png", dpi=300)
        # plt.savefig(odir / f"{biomarker}.png", dpi=300)
        plt.close()

    df = df.dropna(subset=biomarkers)
    n_accept = df.shape[0]
    print(f"n_accept: {n_accept}; n_participants: {len(df['person_id'].unique())}")

with (odir / "stats.csv").open("w") as f:
    pd.DataFrame(rows).to_csv(f, index=False)
