# %%
import time
from pathlib import Path

# %%
import numpy as np
import yaml
from utils import (
    build_biomarker,
    load_biomarker_df,
    load_fids,
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
all_fids = list()
for params in biomarkers.values():
    all_fids.extend(params["fids"])
start = time.time()
preload = load_fids(all_fids)
end = time.time()
print(f"preloading took {(end-start) / 60.0}min")

# %%
for biomarker, params in biomarkers.items():
    biomarker_df = load_biomarker_df(fids=params["fids"], visits=params["visits"], preload=preload)
    if biomarker == "diet":
        biomarker_df = biomarker_df.replace(
            {
                -3: np.nan,  # prefer not answer
                -1: np.nan,  # do not know
                -10: 0,  # less than one
            }
        )
    build_biomarker(biomarker_df=biomarker_df, odir=odir / biomarker)
