# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import os

# %%
from datetime import datetime
from pathlib import Path

import dxdata
import numpy as np
import pandas as pd
from utils import dataset, month_of_birth

from delphi.env import DELPHI_DATA_DIR

# %%
for _ in dataset.entities:
    print("-> " + _.entity_label_singular + " [" + _.name + "]")
    print(_.entity_description)

# %%
gp_scripts = dataset["gp_scripts"]
gp_scripts.fields

# %%
idir = Path(DELPHI_DATA_DIR)
read2bnf = pd.read_csv(idir / "read_v2_drugs_bnf.csv")
read2bnf["read_code"] = read2bnf["read_code"].str.replace(" ", "")
read2bnf = read2bnf.set_index("read_code")["bnf_code"]
read2bnf.head()

# %%
code_map = pd.read_csv(
    idir / "atc_to_bnf.csv", converters={"ATC": str, "BNF": str, "BNF_NORM": str}
)
code_map = code_map.dropna(subset=["ATC", "BNF_NORM"])
code_map = code_map[code_map["BNF_NORM"] != "<NA>"]
code_map.head()

# %%
bnf2atc = code_map.drop_duplicates("BNF_NORM", keep="first")
bnf2atc = bnf2atc.set_index("BNF_NORM")
bnf2atc = bnf2atc["ATC"]
bnf2atc.head()

# %%
drug_first_word = code_map["NAME"].str.split(" ", expand=True)[0]
first_four_digits = code_map["BNF_NORM"].str[:4]
code_map["FINE_BNF"] = drug_first_word + first_four_digits
fine_bnf2atc = code_map.copy()
fine_bnf2atc = fine_bnf2atc.drop_duplicates(subset="FINE_BNF")
fine_bnf2atc = fine_bnf2atc.set_index("FINE_BNF")
fine_bnf2atc = fine_bnf2atc["ATC"]
fine_bnf2atc.head()

# %%
mob_df = month_of_birth()
mob_participants = mob_df.index.astype(int).to_numpy()

# %%

# %%
df = gp_scripts.retrieve_fields(
    engine=engine,
    fields=[
        gp_scripts.fields[0],
        gp_scripts.fields[2],
        gp_scripts.fields[3],
        gp_scripts.fields[4],
        gp_scripts.fields[6],
    ],
)
df = df.filter((df.issue_date >= "2010-01-01") & (df.issue_date < "2019-12-31"))

from pyspark.sql.functions import col, row_number

# %%
from pyspark.sql.window import Window

window = Window.orderBy("eid", "issue_date")
df_with_row_num = df.withColumn("row_num", row_number().over(window))
df_with_row_num.cache()
print("Caching dataframe with row numbers...")
total_rows = df_with_row_num.count()  # This triggers the caching
print(f"Total rows: {total_rows}")

# %%
import math

chunk_size = 500000
n_chunks = math.ceil(total_rows / chunk_size)
n_chunks

# %%
from tqdm import tqdm

hold_out_chunk = None
count_matrices = []
pbar = tqdm(range(n_chunks), total=n_chunks, leave=False)
for i in pbar:
    start_row = i * chunk_size + 1
    end_row = (i + 1) * chunk_size

    if i == n_chunks - 1:
        # Last chunk - no upper limit
        chunk = df_with_row_num.filter(col("row_num") >= start_row)
    else:
        chunk = df_with_row_num.filter(
            (col("row_num") >= start_row) & (col("row_num") <= end_row)
        )
    chunk = chunk.drop("row_num").toPandas()
    chunk = chunk[chunk["eid"].isin(included_subjects)]

    # first occurrence only
    chunk = chunk.drop_duplicates(subset=["eid", "bnf_code"], keep="first")

    chunk["issue_date"] = pd.to_datetime(chunk["issue_date"], format="%d/%m/%Y")
    chunk["timesteps"] = (chunk["issue_date"] - chunk["mob"]).dt.days.values.astype(
        np.float32
    )
    have_dates = (chunk["timesteps"].notna()) & (chunk["timesteps"] >= 0)

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

    have_tokens = chunk["atc"] != "nan"
    assert chunk["atc"].isna().sum() == 0

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
