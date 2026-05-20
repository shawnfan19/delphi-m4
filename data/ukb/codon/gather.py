# %%
from pathlib import Path

# %%
import numpy as np
import yaml
from utils_codon import UKBDatabase, build_biomarker, load_ukb_biomarker_fids

from delphi.env import DELPHI_DATA_DIR

# %%
DATA_DIR = Path(__file__).resolve().parent.parent.parent  # repo .../data
db = UKBDatabase(Path(DELPHI_DATA_DIR) / "ukb")
odir_root = Path(DELPHI_DATA_DIR) / "ukb_real_data" / "biomarkers"

# %%
name_to_fid = load_ukb_biomarker_fids(DATA_DIR)
with open(DATA_DIR / "panel" / "ukb.yaml", "r") as f:
    panels = yaml.safe_load(f)

# %%
for panel_name, panel in panels.items():
    names = panel["biomarkers"]
    fids = [name_to_fid[n] for n in names]
    biomarker_df = db.load_biomarker_df(
        fids=fids,
        visits=panel["visits"],
        name_by_fid=dict(zip(fids, names)),
    )
    if panel_name == "diet":
        biomarker_df = biomarker_df.replace(
            {
                -3: np.nan,  # prefer not answer
                -1: np.nan,  # do not know
                -10: 0,  # less than one
            }
        )
    build_biomarker(
        biomarker_df=biomarker_df,
        features=names,
        odir=odir_root / panel_name,
        time_series=db.long_assessment_age(),
    )
