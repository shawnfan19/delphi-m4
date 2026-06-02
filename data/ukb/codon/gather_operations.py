from pathlib import Path

import numpy as np
from utils_codon import UKBDatabase, build_expansion_pack

from delphi import DAYS_PER_YEAR
from delphi.env import DELPHI_DATA_DIR

db = UKBDatabase(Path(DELPHI_DATA_DIR) / "ukb")

vocab = db.load_coding(5)
reject_vals = [-1, 99999]
vocab = vocab.loc[~vocab["coding"].isin(reject_vals)]
code_vals = vocab["coding"].unique()
code_map = {code: i + 1 for i, code in enumerate(code_vals)}
vocab = vocab.set_index("coding")
tokenizer_keys = (
    vocab.loc[code_vals, "meaning"].str.replace(" ", "_").str.lower().tolist()
)
tokenizer_values = code_map.values()
tokenizer = dict(zip(tokenizer_keys, tokenizer_values))

max_key = max(code_map.keys())
lookup = np.zeros((max_key + 1,), dtype=int)
for k, v in code_map.items():
    lookup[k] = v

token_df = db.load_fid("20004")
time_df = db.load_fid("20011")
ops_participants = time_df.index.to_numpy().astype(int)
valid_participants = ops_participants[np.isin(ops_participants, token_df.index)]

time_df = time_df.loc[valid_participants]
token_df = token_df.loc[valid_participants]
time_np = time_df.to_numpy().astype(np.float32)
time_np *= DAYS_PER_YEAR
token_np = token_df.to_numpy().astype(int)

accept_mask = (token_np > 0) * (token_np < 99999) * (time_np > 0)
count_np = np.sum(accept_mask, axis=1)
token_np = token_np[accept_mask].ravel()
time_np = time_np[accept_mask].ravel()
token_np = lookup[token_np]

build_expansion_pack(
    token_np=token_np,
    time_np=time_np,
    count_np=count_np,
    subjects=valid_participants,
    tokenizer=tokenizer,
    expansion_pack="self_report_ops",
    odir=Path(DELPHI_DATA_DIR) / "ukb_real_data" / "expansion_packs",
)
