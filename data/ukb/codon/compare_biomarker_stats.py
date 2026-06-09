"""Join UKB and AoU per-biomarker stats and surface the largest distribution
shifts in the model's z-score frame (UKB train mean/std).

delta_median_in_ukb_sigma = (median_aou - mean_ukb) / std_ukb
    where the typical AoU value lands after UKB z-scoring; |.| > 1 = severely OOD.
iqr_ratio = iqr_aou / iqr_ukb
    spread comparison; >> 1 means AoU is much wider.
"""

from pathlib import Path

import pandas as pd

from delphi.env import DELPHI_DATA_DIR

DATA_ROOT = Path(DELPHI_DATA_DIR)
ukb = pd.read_csv(DATA_ROOT / "ukb_real_data" / "biomarker_stats" / "stats.csv")
aou = pd.read_csv(DATA_ROOT / "aou_uk" / "biomarker_stats" / "stats.csv")

df = ukb.merge(aou, on="biomarker", suffixes=("_ukb", "_aou"))

df["delta_median_in_ukb_sigma"] = (df["median_aou"] - df["mean_ukb"]) / df["std_ukb"]
df["iqr_ratio"] = (df["q75_aou"] - df["q25_aou"]) / (df["q75_ukb"] - df["q25_ukb"])

df = df.reindex(
    df["delta_median_in_ukb_sigma"].abs().sort_values(ascending=False).index
)

out = DATA_ROOT / "biomarker_stats_diff.csv"
df.to_csv(out, index=False)
print(f"wrote {out}")

cols = [
    "biomarker",
    "n_aou",
    "n_ukb",
    "median_aou",
    "median_ukb",
    "delta_median_in_ukb_sigma",
    "iqr_ratio",
]
print(df[cols].head(20).to_string(index=False))
