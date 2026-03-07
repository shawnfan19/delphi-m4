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
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

# %%
out_dir = Path(DELPHI_DATA_DIR) / "ukb_real_data" / "biomarkers" / "proteomics"
os.makedirs(out_dir, exist_ok=True)

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

# %%
instance0 = dataset["olink_instance_0"]

# %%
spark_df = instance0.retrieve_fields(fields=instance0.fields, engine=engine)
non_id_cols = spark_df.columns[1:]
spark_df_filtered = spark_df.dropna(how='all', subset=non_id_cols)
protein0 = spark_df_filtered.toPandas()

# %%
protein0.shape

# %%
protein0 = protein0.set_index("eid")
protein0.head()

# %%
protein0.to_csv(out_dir / "instance0.csv")

# %%

# %%
instance = dataset["olink_instance_2"]
spark_df = instance.retrieve_fields(fields=instance.fields, engine=engine)
spark_df_filtered = spark_df.dropna(how='all', subset=non_id_cols)
protein = spark_df_filtered.toPandas()

# %%
protein.shape

# %%
protein = protein.set_index("eid")
protein.head()

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
