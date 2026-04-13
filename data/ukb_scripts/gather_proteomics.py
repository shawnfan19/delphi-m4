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

from delphi.env import DELPHI_DATA_WRITE as DELPHI_DATA_DIR

from utils_rap import build_biomarker

spark = SparkSession.builder \
    .config("spark.driver.maxResultSize", "4g") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

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
instance0 = dataset["olink_instance_0"]

# %%
spark_df = instance0.retrieve_fields(fields=instance0.fields, engine=engine)
non_id_cols = spark_df.columns[1:]
spark_df = spark_df.dropna(how='all', subset=non_id_cols)
protein0 = spark_df.toPandas()

# %%
protein0.shape

# %%
pd.isna(protein0).sum(axis=1).hist(bins=20);
plt.xlabel("number of missing proteins")

# %%
print(f"Original shape before filtering missingness: {protein0.shape}")
# Calculate the fraction of missing values for each protein column
missing_fractions = protein0[non_id_cols].isna().mean()
# Identify proteins that have 20% or less missing data
proteins_to_keep = missing_fractions[missing_fractions <= 0.20].index.tolist()
dropped_count = len(non_id_cols) - len(proteins_to_keep)
print(f"Dropping {dropped_count} proteins due to >20% missingness.")
# Keep only 'eid' and the valid proteins
protein0 = protein0[['eid'] + proteins_to_keep]
# ==========================================

# %%
participants_to_keep = protein0[proteins_to_keep].isna().mean(axis=1) <= 0.20
participants_to_keep.sum(), (protein0[proteins_to_keep].isna().sum(axis=1) > 600).sum()

# %%
protein0 = protein0.loc[participants_to_keep]

# %%
protein0.head()

# %%
protein_medians = protein0[proteins_to_keep].median()
# Impute the missing values with the calculated medians
protein0[proteins_to_keep] = protein0[proteins_to_keep].fillna(protein_medians)

# %%
protein0["visit"] = "init_assess"
protein0 = protein0.rename(columns={"eid": "pid"})
protein0 = protein0.set_index(["pid", "visit"])

# %%
protein0.head()

# %%

# %%
build_biomarker(
    biomarker_df=protein0,
    features=list(protein0.columns),
    odir=Path(DELPHI_DATA_DIR) / "ukb_real_data" / "biomarkers" / "proteomics",
)

# %%

# %%
instance = dataset["olink_instance_2"]
spark_df = instance.retrieve_fields(fields=instance.fields, engine=engine)
non_id_cols = spark_df.columns[1:]
spark_df_filtered = spark_df.dropna(how='all', subset=non_id_cols)
protein = spark_df_filtered.toPandas()

# %%
protein.shape

# %% [raw]
# protein = protein.set_index("eid")
# protein.head()

# %%
protein.to_csv(out_dir / "instance2.csv")

# %%
instance = dataset["olink_instance_3"]
spark_df = instance.retrieve_fields(fields=instance.fields, engine=engine)
spark_df_filtered = spark_df.dropna(how='all', subset=non_id_cols)
protein = spark_df_filtered.toPandas()

# %%
protein.shape

# %%
protein = protein.set_index("eid")
protein.head()

# %%
protein.to_csv(out_dir / "instance3.csv")

# %%
