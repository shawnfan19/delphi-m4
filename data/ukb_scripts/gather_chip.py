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

from utils import build_biomarker, dataset, engine

# %%
main_entity = dataset.primary_entity

# %%
variant_fields = sorted(list(main_entity.find_fields(name_regex=f".*p30106.*")))
freq_fields = sorted(list(main_entity.find_fields(name_regex=f".*p30107.*")))
ct_field = main_entity["p30105"]
eid_field = main_entity.find_field(name="eid")
eid_field

# %%
variant_fields, freq_fields, ct_field, eid_field

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
is_valid = ~np.isnan(variant_ct.values.ravel())
print(is_valid.sum())

freq_arr = freq_arr[is_valid, :]
gene_arr = gene_arr[is_valid, :]
pids = pids[is_valid]

vaf_arr = create_vaf_matrix(gene_arr, freq_arr, gene_to_id)
vaf_arr.shape

# %%
vaf_df = pd.DataFrame(
    data=vaf_arr,
    columns=list(gene_to_id.keys()),
    index=pd.MultiIndex.from_arrays(
        [pids, np.full_like(pids, "init_assess")], names=["pid", "visit"]
    )
)

build_biomarker(
    biomarker_df=vaf_df,
    features=list(gene_to_id.keys()),
    odir=Path(DELPHI_DATA_DIR) / "ukb_real_data" / "biomarkers" / "chip_lite",
)

# %%
freq_arr = np.nan_to_num(freq_arr)
freq_arr

# %%
n_mutations = (freq_arr > 0).sum(axis=1)

max_clone_size = freq_arr.max(axis=1)

high_risk_genes = {"SRSF2", "SF3B1", "ZRSR2", "TP53", "RUNX1", "IDH1", "IDH2", "JAK2"}
high_risk_ids = [gene_to_id[gene] - 1 for gene in high_risk_genes]
has_high_risk = (vaf_arr[:, high_risk_ids].sum(axis=1) > 0).astype(int)
has_high_risk

# %%
chrs = np.stack((n_mutations, max_clone_size, has_high_risk), axis=1)
chrs.shape

# %%
features = ["n_mutations", "max_clone_size", "has_high_risk_genes"]

chrs_df = pd.DataFrame(
    data=chrs,
    columns=features,
    index=pd.MultiIndex.from_arrays(
        [pids, np.full_like(pids, "init_assess")], names=["pid", "visit"]
    )
)

build_biomarker(
    biomarker_df=chrs_df,
    features=features,
    odir=Path(DELPHI_DATA_DIR) / "ukb_real_data" / "biomarkers" / "chrs",
)

# %%

# %%
