# %%
import time
from pathlib import Path

# %%
import numpy as np
import yaml

from delphi.env import IN_RAP

if IN_RAP:
    from utils_rap import (
        build_biomarker,
        load_biomarker_df,
    )
else:
    from utils_codon import (
        build_biomarker,
        load_biomarker_df,
    )

# %%
from delphi.env import DELPHI_DATA_DIR

# %%
odir = Path(DELPHI_DATA_DIR) / "ukb_real_data" / "biomarkers"
odir

# %%
with open("dictionary/panel.yaml", "r") as f:
    biomarkers = yaml.safe_load(f)

# %%
for biomarker, params in biomarkers.items():
    biomarker_df = load_biomarker_df(
        fids=list(params["fids"].keys()), visits=params["visits"]
    )
    if biomarker == "diet":
        biomarker_df = biomarker_df.replace(
            {
                -3: np.nan,  # prefer not answer
                -1: np.nan,  # do not know
                -10: 0,  # less than one
            }
        )
    build_biomarker(
        biomarker_df=biomarker_df,
        features=list(params["fids"].values()),
        odir=odir / biomarker,
    )

# %%
biomarker_df["26501"].isna().sum()

# %%
