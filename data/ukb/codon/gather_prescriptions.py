# +
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils_codon import UKBDatabase, build_expansion_pack

from delphi.env import DELPHI_DATA_DIR

# -

idir = Path(DELPHI_DATA_DIR) / "ukb/primary_care"

read2bnf = pd.read_csv(idir / "read_v2_drugs_bnf.csv")
read2bnf["read_code"] = read2bnf["read_code"].str.replace(" ", "")
read2bnf = read2bnf.set_index("read_code")["bnf_code"]

read2bnf.head()

code_map = pd.read_csv(
    idir / "atc_to_bnf.csv", converters={"ATC": str, "BNF": str, "BNF_NORM": str}
)
code_map = code_map.dropna(subset=["ATC", "BNF_NORM"])
code_map = code_map[code_map["BNF_NORM"] != "<NA>"]

code_map.head()

bnf2atc = code_map.drop_duplicates("BNF_NORM", keep="first")
bnf2atc = bnf2atc.set_index("BNF_NORM")
bnf2atc = bnf2atc["ATC"]

bnf2atc.head()

drug_first_word = code_map["NAME"].str.split(" ", expand=True)[0]
first_four_digits = code_map["BNF_NORM"].str[:4]
code_map["FINE_BNF"] = drug_first_word + first_four_digits
fine_bnf2atc = code_map.copy()
fine_bnf2atc = fine_bnf2atc.drop_duplicates(subset="FINE_BNF")
fine_bnf2atc = fine_bnf2atc.set_index("FINE_BNF")
fine_bnf2atc = fine_bnf2atc["ATC"]

fine_bnf2atc.head()

chunksize = 500000
df = pd.read_csv(
    idir / "gp_scripts.txt",
    sep="\t",
    chunksize=int(chunksize),
    encoding="ISO-8859-1",
    converters={
        "eid": int,
        "data_provider": str,
        "issue_date": str,
        "bnf_code": str,
        "read_2": str,
        "dmd_code": str,
        "drug_name": str,
        "quantity": str,
    },
)

db = UKBDatabase(Path(DELPHI_DATA_DIR) / "ukb")
mob_df = db.month_of_birth()
mob_participants = mob_df.index.astype(int).to_numpy()

# +
subjects = []
count_list = []
all_tokens = []
all_timesteps = []

hold_out_chunk = None
pbar = tqdm(enumerate(df), leave=False)
for i, chunk in pbar:
    is_last_chunk = len(chunk) < int(1e6)

    # keep only those with month of birth data
    has_mob = chunk["eid"].isin(mob_participants)
    chunk = chunk.loc[has_mob].copy()
    chunk["mob"] = mob_df.loc[chunk["eid"], "year_month"].values  # type: ignore

    if hold_out_chunk is not None:
        chunk = pd.concat([hold_out_chunk, chunk], ignore_index=False)
    subs = chunk["eid"].unique()
    if not is_last_chunk:
        hold_out_sub = subs[-1]
        hold_out_chunk = chunk.loc[chunk["eid"] == hold_out_sub].copy()
        chunk = chunk.loc[chunk["eid"] != hold_out_sub].copy()

    chunk = chunk.drop_duplicates(subset=["eid", "bnf_code"], keep="first")

    chunk["issue_date"] = pd.to_datetime(chunk["issue_date"], format="%d/%m/%Y")
    chunk["timesteps"] = (chunk["issue_date"] - chunk["mob"]).dt.days.values.astype(
        np.float32
    )

    bnf_codes = chunk["bnf_code"].copy()
    # deal with formats like 04.06.03.00.00
    bnf_codes = bnf_codes.str.replace(".", "")

    # recover empty bnf codes from read_2 codes
    chunk["read_2"] = chunk["read_2"].str.replace("00", "")
    read_2 = chunk["read_2"]
    read_2_empty = read_2 == ""
    bnf_empty = bnf_codes == ""
    to_map = bnf_empty & ~read_2_empty
    if to_map.sum() > 0:
        mapped_bnf_codes = read2bnf.loc[chunk.loc[to_map, "read_2"]].values
        bnf_codes.loc[to_map] = mapped_bnf_codes

    atc_codes = pd.Series(index=bnf_codes.index).astype(str)

    # coarse match
    bnf_codes = bnf_codes.str[:7]
    is_coarse = bnf_codes.isin(bnf2atc.index)
    atc_codes[is_coarse] = bnf2atc.loc[bnf_codes[is_coarse]].values

    # fine match
    bnf_codes = bnf_codes.str[:4]
    drug_names = chunk["drug_name"].str.split(" ", expand=True)[0]
    aug_bnf_codes = drug_names + bnf_codes
    is_fine = aug_bnf_codes.isin(fine_bnf2atc.index)
    is_fine = is_fine & ~is_coarse
    atc_codes[is_fine] = fine_bnf2atc.loc[aug_bnf_codes[is_fine]].values

    pbar.set_postfix(
        chunk=i + 1,
        total_count=len(bnf_codes),
        coarse_match=is_coarse.sum(),
        fine_match=is_fine.sum(),
    )

    chunk["atc"] = atc_codes.values
    chunk["atc"] = chunk["atc"].str[:5]

    # first occurrence only
    chunk = chunk.drop_duplicates(subset=["eid", "atc"], keep="first")

    have_tokens = chunk["atc"] != "nan"
    assert chunk["atc"].isna().sum() == 0
    have_dates = (chunk["timesteps"].notna()) & (chunk["timesteps"] >= 0)

    are_valid = have_dates & have_tokens
    tokens = chunk.loc[are_valid, "atc"].values.tolist()
    timesteps = chunk.loc[are_valid, "timesteps"].values.tolist()
    all_tokens.extend(tokens)
    all_timesteps.extend(timesteps)

    unique_subs = chunk.loc[are_valid, "eid"].unique().tolist()
    subjects.extend(unique_subs)

    seq_len = chunk.loc[are_valid, "eid"].value_counts()
    seq_len = seq_len.loc[unique_subs].to_list()
    count_list.extend(seq_len)
# -
raw_tokens = np.array(all_tokens)
tokenizer = {str(atc_code): i + 1 for i, atc_code in enumerate(np.unique(raw_tokens))}
token_np = np.array([tokenizer[str(token)] for token in raw_tokens])


build_expansion_pack(
    token_np=token_np,
    time_np=np.array(all_timesteps),
    count_np=np.array(count_list),
    subjects=np.array(subjects),
    tokenizer=tokenizer,
    expansion_pack="prescriptions",
    odir=Path(DELPHI_DATA_DIR) / "ukb_real_data" / "expansion_packs",
)
