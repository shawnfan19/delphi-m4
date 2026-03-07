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

# %%
import dxdata
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import DateType, DoubleType
import os
from pathlib import Path
import time
import yaml
import re
import subprocess

import pandas as pd
import matplotlib.pyplot as plt

from delphi.env import DELPHI_DATA_DIR

from utils import assessment_age

spark = SparkSession.builder \
    .appName("example") \
    .getOrCreate()

# %%
engine = dxdata.connect(dialect="hive+pyspark")
project = os.getenv('DX_PROJECT_CONTEXT_ID')
record = os.popen("dx find data --type Dataset --delimiter ',' | awk -F ',' '{print $5}'").read().rstrip()
record = record.split('\n')[0]
DATASET_ID = project + ":" + record
dataset = dxdata.load_dataset(id=DATASET_ID)
print(f"DATASET_ID: {DATASET_ID}")

# %%
dataset.entities

# %%
main_entity = dataset.primary_entity

# %%
list(main_entity.find_fields(name_regex=f".*p30106_.*"))

# %%

# %%
variant_ct = main_entity.retrieve_fields(fields=[main_entity["p30105"]], engine=engine).toPandas()
variant = main_entity.retrieve_fields(fields=list(main_entity.find_fields(name_regex=f".*p30106_.*")), engine=engine).toPandas()
variant_freq = main_entity.retrieve_fields(fields=list(main_entity.find_fields(name_regex=f".*p30107_.*")), engine=engine).toPandas()

# %%
pids = main_entity.retrieve_fields(
    fields=[main_entity.find_field(name="eid")],
    engine=engine
).toPandas()
pids = pids["eid"].values.astype(int)

# %%
pids

# %%
age = assessment_age()

# %%
init_assess_age = age["init_assess"].values
init_assess_age


# %%

# %%
def encode_variants(variant: pd.DataFrame, variant_freq: pd.DataFrame):
    """
    Encode participant variants into compact binary representation.
    
    Parameters
    ----------
    variant : pd.DataFrame, shape (N, K)
        Each row lists variant strings for a participant. None for empty slots.
        Format: GENE:TRANSCRIPT:EXON:CDNA:PROTEIN
    variant_freq : pd.DataFrame, shape (N, K)
        Position-matched VAFs. None for empty slots.
    
    Outputs
    -------
    gene.bin       : int32 array of gene IDs (contiguous, 1D)
    frequency.bin  : float32 array of VAFs (contiguous, 1D)
    p2i.csv        : participant_id, start, length
    gene_map.csv   : gene name to integer ID mapping
    """
    
    N, K = variant.shape
    assert variant_freq.shape == (N, K), "Shape mismatch between variant and variant_freq"
    # --- Step 1: Extract gene names from variant strings ---
    # "DNMT3A:NM_022552:exon23:c.G2645A:p.R882H" -> "DNMT3A"
    def extract_gene(variant_str):
        if variant_str is None or (isinstance(variant_str, float) and np.isnan(variant_str)):
            return None
        return str(variant_str).split(":")[0]
    # Vectorize gene extraction across the entire dataframe
    gene_df = variant.applymap(extract_gene)
    
    # --- Step 2: Collect all unique genes and build mapping ---
    all_genes = set()
    for col in gene_df.columns:
        all_genes.update(gene_df[col].dropna().unique())
    
    all_genes = sorted(all_genes)  # sorted for reproducibility
    gene_to_id = {gene: idx + 1 for idx, gene in enumerate(all_genes)}  # 1-indexed (0 reserved for "no gene" if needed)
    print(f"Found {len(gene_to_id)} unique genes:")
    for gene, gid in gene_to_id.items():
        print(f"  {gid:3d} -> {gene}")
        
    # --- Step 3: Build contiguous arrays ---
    gene_df = gene_df.replace(gene_to_id)
    gene_arr = gene_df.values
    freq_arr = variant_freq.values
    
    return gene_arr, freq_arr, gene_to_id


# %%

# %%
variant

# %%
gene_arr, freq_arr, gene_to_id = encode_variants(variant, variant_freq)

# %%
is_val = ~np.isnan(gene_arr)
gene_arr = gene_arr[is_val]
freq_arr = freq_arr[is_val]
assert np.isnan(freq_arr).sum() == 0
gene_arr.shape, freq_arr.shape

# %%
seq_len = np.sum(is_val, axis=1)
has_data = seq_len > 0

# %%
has_time = ~np.isnan(init_assess_age)

# %%
is_valid = has_data & has_time

# %%
seq_len = seq_len[is_valid]
starts = np.cumsum(seq_len) - seq_len[0]
pids = pids[is_valid]
timesteps = init_assess_age[is_valid]

# %%
starts, seq_len, pids, timesteps

# %%
starts.shape, seq_len.shape, pids.shape, timesteps.shape

# %%
p2i = pd.DataFrame({
        "pid": pids,
        "start_pos": starts,
        "seq_len": seq_len,
        "time": timesteps,
    })

# %%
out_dir = Path(DELPHI_DATA_DIR) / "ukb_real_data" / "biomarkers" / "chip-lite"
os.makedirs(out_dir, exist_ok=True)

# %%
gene_arr.tofile(out_dir / "gene.bin")
freq_arr.tofile(out_dir / "frequency.bin")
p2i.to_csv(out_dir / "p2i.csv", index=False)

# %%
