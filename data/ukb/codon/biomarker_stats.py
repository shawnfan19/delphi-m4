"""Per-biomarker UKB distribution plots, sourced directly from the UKB tab
parquet via UKBDatabase.load_fid().

Each biomarker is plotted once across all visits, independent of panel
co-occurrence filtering. Output is a flat directory of PNGs mirroring
the AoU layout, so plots can be compared side-by-side by filename.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from utils_codon import UKBDatabase, load_ukb_biomarker_fids

from delphi.env import DELPHI_DATA_DIR

DATA_DIR = Path(__file__).resolve().parent.parent.parent  # repo .../data
db = UKBDatabase(Path(DELPHI_DATA_DIR) / "ukb")
name_to_fid = load_ukb_biomarker_fids(DATA_DIR)

with open(DATA_DIR / "panel" / "aou.yaml", "r") as f:
    panels = yaml.safe_load(f)

biomarkers = sorted({bm for panel in panels.values() for bm in panel})

odir = Path(DELPHI_DATA_DIR) / "ukb_real_data" / "biomarker_stats"
odir.mkdir(parents=True, exist_ok=True)

rows = []
for biomarker in biomarkers:
    fid = name_to_fid[biomarker]
    df = db.load_fid(fid).apply(pd.to_numeric, errors="coerce")
    # First occurrence per participant: bfill across visit-instance columns
    # (which are in chronological order: instance 0 = init_assess, 1 = repeat, ...)
    # and take the first column, then drop participants with no measurement.
    df = df.reindex(sorted(df.columns), axis=1)
    values = df.bfill(axis=1).iloc[:, 0].dropna().to_numpy()
    print(f"{biomarker} (fid={fid}): n_values={len(values)}")

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
        2,
        1,
        sharex=False,
        gridspec_kw={"height_ratios": [4, 1]},
        figsize=(6, 6),
    )
    ax_hist.hist(values, bins=bins, alpha=0.3)
    ax_hist.set_yscale("log")
    ax_hist.set_ylabel("Count (log)")

    ax_box.boxplot(values, vert=False, showmeans=True, showfliers=False)
    ax_box.set_xlabel(biomarker)

    fig.suptitle(biomarker)
    fig.tight_layout()
    plt.savefig(odir / f"{biomarker}.png", dpi=300)
    plt.close()

pd.DataFrame(rows).to_csv(odir / "stats.csv", index=False)
