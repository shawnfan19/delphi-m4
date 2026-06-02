from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from utils_codon import UKBDatabase, build_expansion_pack

from delphi.env import DELPHI_DATA_DIR

db = UKBDatabase(Path(DELPHI_DATA_DIR) / "ukb")
pack_root = Path(DELPHI_DATA_DIR) / "ukb_real_data" / "expansion_packs"

with open(pack_root / "family_hx" / "tokenizer.yaml", "r") as f:
    tokenizer = yaml.safe_load(f)

with open("data/gather_biomarker/family_hx_coding.yaml", "r") as f:
    map_config = yaml.safe_load(f)
lookup = np.zeros((max(map_config.keys()) + 1,), dtype=np.int32)
for key, value in map_config.items():
    lookup[int(key)] = int(value)


def load_visit(fid: str, visit_idx: int = 0) -> pd.DataFrame:
    df = db.load_fid(fid=fid)
    in_visit = df.columns.str.contains(f"f.{fid}.{visit_idx}")
    return df.iloc[:, in_visit]


father_hx_df = load_visit("20107", visit_idx=0)
mother_hx_df = load_visit("20110", visit_idx=0)
sibling_hx_df = load_visit("20111", visit_idx=0)

family_hx_participants = father_hx_df.index.astype(int).to_numpy()

all_hx_df = pd.concat([father_hx_df, mother_hx_df, sibling_hx_df], axis=1)
token_np = all_hx_df.values
accept_mask = (token_np > 0) * (~np.isnan(token_np))
count_np = accept_mask.sum(axis=1)
token_np = token_np[accept_mask]
token_np = token_np.astype(int)
token_np = lookup[token_np]
time_np = np.zeros_like(token_np)

build_expansion_pack(
    token_np=token_np,
    time_np=time_np,
    count_np=count_np,
    subjects=family_hx_participants,
    tokenizer=tokenizer,
    expansion_pack="family_hx",
    odir=pack_root,
)
