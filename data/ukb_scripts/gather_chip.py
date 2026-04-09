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
variant_fields = sorted(list(main_entity.find_fields(name_regex=f".*p30106.*")))
freq_fields = sorted(list(main_entity.find_fields(name_regex=f".*p30107.*")))
ct_field = main_entity["p30105"]

# %%
variant_fields, freq_fields, ct_field

# %%
eid_field = main_entity.find_field(name="eid")
eid_field

# %%
all_fields = variant_fields + freq_fields + [ct_field] + [eid_field]
df = main_entity.retrieve_fields(fields=all_fields, engine=engine).toPandas()

# %%
df = df.set_index("eid")
df.head()

# %%
pids = df.index.values
pids

# %%
pids.shape

# %%
age = assessment_age()

# %%
init_assess_age = age.loc[pids, "init_assess"]
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
variant = df[[f.name for f in variant_fields]]
variant_freq = df[[f.name for f in freq_fields]]
variant_ct = df[ct_field.name]

# %%
variant.head()

# %%
gene_arr, freq_arr, gene_to_id = encode_variants(variant, variant_freq)

# %%
freq_arr.shape, gene_arr.shape


# %%

# %%
def create_vaf_matrix(gene_arr: np.ndarray, freq_arr: np.ndarray, gene_to_id: dict):
    """
    Create a participant-by-gene VAF matrix, summing frequencies 
    for multiple mutations in the same gene.
    
    Parameters
    ----------
    gene_arr : np.ndarray, shape (N, K)
        Encoded gene IDs. Empty slots are None or NaN.
    freq_arr : np.ndarray, shape (N, K)
        Position-matched VAFs. Empty slots are None or NaN.
    gene_to_id : dict
        Mapping of gene names to integer IDs (1-indexed).
        
    Returns
    -------
    vaf_arr : np.ndarray, shape (N, V)
        Matrix where vaf_arr[i, j] is the sum of VAFs for participant i and gene j.
    """
    N, K = gene_arr.shape
    V = len(gene_to_id)
    
    # Initialize the [N, V] matrix with zeros
    vaf_arr = np.zeros((N, V), dtype=float)
    
    # Flatten arrays for vectorized processing
    flat_genes = gene_arr.flatten()
    flat_freqs = freq_arr.flatten()
    
    # Create a boolean mask of valid entries (ignoring None / NaN)
    valid_mask = pd.notna(flat_genes) & pd.notna(flat_freqs)
    
    # Get the row indices (participant indices: 0 to N-1)
    # np.repeat creates an array like [0,0... 1,1... N-1,N-1...]
    row_indices = np.repeat(np.arange(N), K)[valid_mask]
    
    # Get the column indices (gene IDs). 
    # Subtract 1 because gene_to_id is 1-indexed, but numpy arrays are 0-indexed.
    col_indices = flat_genes[valid_mask].astype(int) - 1
    
    # Get the valid VAF values
    valid_freqs = flat_freqs[valid_mask].astype(float)
    
    # In-place unbuffered addition. 
    # If a participant has multiple mutations in the same gene (same row, same col), 
    # np.add.at correctly sums their valid_freqs together.
    np.add.at(vaf_arr, (row_indices, col_indices), valid_freqs)
    
    return vaf_arr


# %%
vaf_arr = create_vaf_matrix(gene_arr, freq_arr, gene_to_id)
vaf_arr

# %%
vaf_arr.shape

# %%
has_data = ~np.isnan(variant_ct.values.ravel())
has_time = ~np.isnan(init_assess_age)
is_valid = has_data & has_time
is_valid.sum()

# %%
vaf_arr = vaf_arr[is_valid]
pids = pids[is_valid]
timesteps = init_assess_age[is_valid]

# %%
seq_len = vaf_arr.shape[1]
starts = seq_len * np.arange(vaf_arr.shape[0])
seq_len = np.full_like(starts, seq_len)

# %%
starts, seq_len, pids, timesteps, vaf_arr.size

# %%
starts.shape, seq_len.shape, pids.shape, timesteps.shape

# %%

# %%
p2i = pd.DataFrame({
        "pid": pids,
        "start_pos": starts,
        "seq_len": seq_len,
        "time": timesteps,
    })
p2i["visit"] = "init_assess"

# %%
odir = Path(DELPHI_DATA_DIR) / "ukb_real_data" / "biomarkers" / "chip-lite"
os.makedirs(odir, exist_ok=True)
vaf_arr.ravel().astype(np.float32).tofile(odir / "data.bin")
p2i.to_csv(odir / "p2i.csv", index=False)
with open(odir / "features.yaml", "w") as f:
    yaml.dump(list(gene_to_id.keys()), f)

# %%
odir

# %%
