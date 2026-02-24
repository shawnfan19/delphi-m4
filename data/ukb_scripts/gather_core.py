#!/usr/bin/env python
# coding: utf-8
# %%
import dxdata
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import DateType, DoubleType
import os
import time
import yaml
import re
import subprocess

from delphi.env import DELPHI_DATA_DIR

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
command = f"dx extract_dataset -ddd {DATASET_ID}"
print(f"extracting data dictionaries: {command}")
subprocess.run(command, shell=True) 


# %%
pattern = re.compile(r'.*data_dictionary\.csv$')
files = os.listdir(".")
matching_files = [f for f in files if pattern.match(f)]
assert len(matching_files) == 1
data_dict_path = matching_files[0]
print(f"loaded data dictionary {data_dict_path}")


# %%
data_dict = pd.read_csv(data_dict_path)
data_dict = data_dict.set_index("name")


# %%
with open("dictionary/tokenizer.yaml", "r") as f:
    tokenizer = yaml.safe_load(f)
    
pattern = re.compile(r"^([a-z]\d{2}).*")  # capture first 3 chars
def truncate_token(s):
    return pattern.sub(r"\1", s)

tokenizer = {truncate_token(k): v for k, v in tokenizer.items()}
tokenizer = pd.Series(tokenizer)


# %%
main_entity = dataset.primary_entity
eid_f = main_entity.find_field(name="eid")
sex_f = main_entity.find_field(title="Sex")
year_f = main_entity.find_field(title="Year of birth")
month_f = main_entity.find_field(title="Month of birth")
death_f = dataset['death'].find_field(title="Date of death")
assessment_f = main_entity["p53_i0"]
bmi_f = main_entity["p21001_i0"]
smoking_f = main_entity["p1239_i0"]
alcohol_f = main_entity["p1558_i0"]


# %%
cancer_codes = {}
cancer_codes['type'] = []
cancer_codes['date'] = []
for i in range(22):
    cancer_codes['type'].append(main_entity.find_field(name="p40006_i" + str(i)))
    cancer_codes['date'].append(main_entity.find_field(name="p40005_i" + str(i)))


# %%
def get_first_occ_fields(main_entity):
    fo_fields = []
    for field in main_entity.fields:
        parts = field.name.split("_")
        if len(str(parts[0])) > 3:
            field_num = int(parts[0][1:])
            if (field_num >= 130000 and field_num <= 132604):
                if field.title.startswith("Date"):
                    fo_fields.append(field)
    return fo_fields


fo_fields = get_first_occ_fields(main_entity)
fo_field_names = [f.name for f in fo_fields]
fid2icd = data_dict.loc[fo_field_names, "title"]
fid2icd = fid2icd.str.split(" ", expand=True)[1]
assert (fid2icd.str.len() == 3).all()
fid2icd = fid2icd.to_dict()


# %%
fields_to_get = [eid_f, sex_f, year_f, month_f, assessment_f, bmi_f, smoking_f, alcohol_f] + cancer_codes['type'] + cancer_codes['date'] + fo_fields
df_main = main_entity.retrieve_fields(fields=fields_to_get, engine=engine)

# %%
# handle death separately
df_death = dataset['death'].retrieve_fields(fields=[eid_f, death_f], engine=engine)
# deduplicate by taking first occurrence
df_death = df_death.groupBy("eid").agg(F.min(death_f.name).alias("death_date"))
df = df_main.join(df_death, on="eid", how="left")

# %%
df = df.withColumn(
    "sex",
    F.when(F.col(sex_f.name) == 0, "female")
    .otherwise("male")
)

df = df.withColumn(
    "dob",
    F.make_date(F.col(year_f.name), F.col(month_f.name), F.lit(1))     # year, month, day=1
)

df = df.withColumn(
    "doa",
    F.to_date(F.col(assessment_f.name))
)

df = df.withColumn(
    "bmi_status",
    F.when(F.col(bmi_f.name) > 28, "bmi_high")
    .when(F.col(bmi_f.name) > 22, "bmi_mid")
    .otherwise("bmi_low")
)

df = df.withColumn(
    "smoking_status",
    F.when(F.col(smoking_f.name) == 1, "smoking_high")
    .when(F.col(smoking_f.name) == 2, "smoking_mid")
    .otherwise("smoking_low")
)

df = df.withColumn(
    "alcohol_status",
    F.when(F.col(alcohol_f.name) == 1, "alcohol_high")
    .when(F.col(alcohol_f.name) < 4, "alcohol_mid")
    .otherwise("alcohol_low")
)

# %%
# truncate cancer codes to 3 digits before stacking
for type_f in cancer_codes["type"]:
    df = df.withColumn(
        type_f.name,
        F.when(
            F.col(type_f.name).rlike("^[A-Z]\d{3}$"),
            F.regexp_extract(F.col(type_f.name), r"([A-Z]\d{2})", 1)
        ).otherwise(F.col(type_f.name))
    )

# %%
stack_args = list()
stack_args.append("sex")
stack_args.append("dob")
stack_args.append("bmi_status")
stack_args.append("doa")
stack_args.append("smoking_status")
stack_args.append("doa")
stack_args.append("alcohol_status")
stack_args.append("doa")
for f in fo_fields:
    stack_args.append(F.lit(fid2icd[f.name]))
    stack_args.append(f.name)
for type_f, date_f in zip(cancer_codes["type"], cancer_codes["date"]):
    stack_args.append(type_f.name)
    stack_args.append(date_f.name)
stack_args.append(F.lit("death"))
stack_args.append("death_date")


# %%
df_long = df.select(
    "eid",
    "dob",
    F.stack(F.lit(int(len(stack_args) / 2)), *stack_args).alias("token", "date")
).where(F.col("date").isNotNull())

df_long = df_long.withColumn(
    "age",
    F.datediff(F.col("date"), F.col("dob"))
)

exclude_dates = [
    '1900-01-01',
    '1901-01-01',
    '1902-02-02',
    '1903-03-03',
    '1909-09-09',
    '2037-07-07'
]
exclude_date_literals = [F.lit(d).cast(DateType()) for d in exclude_dates]
df_long = df_long.filter(~F.col('date').isin(exclude_date_literals))
# still ~200 entires would have negative age; not sure why but remove 
df_long = df_long.where(F.col("age") >= 0)

# %%
df_long = df_long.dropDuplicates(["eid", "token"])

# %%
map_df = tokenizer.reset_index()
map_df.columns = ["token", "token_id"]
map_df = spark.createDataFrame(map_df)

df_long = df_long.withColumn("token", F.lower("token"))

df_filtered = df_long.join(
    F.broadcast(map_df), 
    df_long.token == map_df.token,
    "left_semi"
)
df_final = df_filtered.join(
    F.broadcast(map_df),
    df_filtered.token == map_df.token,
    "left"
)


# %%
start = time.time()
pd_df = df_final.toPandas()
end = time.time()
print(f"dataframe conversion took {(end - start) / 60.0} min")


# %%
all_data = pd_df[["eid", "age", "token_id"]].to_numpy()
all_data = all_data[np.lexsort((all_data[:,1], all_data[:,0]))]
subjects, timesteps, tokens = all_data[:,0], all_data[:,1], all_data[:,2]

odir = os.path.join(DELPHI_DATA_DIR, "ukb_real_data")
if not os.path.exists(odir):
    os.makedirs(odir)
    
tokens.astype(np.uint32).tofile(
    os.path.join(odir, "data.bin")
)
timesteps.astype(np.uint32).tofile(
    os.path.join(odir, "time.bin")
)


# %%
pids, idx, counts = np.unique(subjects, return_index=True, return_counts=True)
pids = pids.astype(np.uint32)
s = np.argsort(idx)
pids = pids[s]
counts = counts[s]


# %%
p2i = pd.DataFrame(
    {
        "pid": pids,
        "start_pos": 0,
        "seq_len": 0,
    }
)
p2i = p2i.set_index("pid")
p2i.loc[pids, "seq_len"] = counts
p2i.loc[pids, "start_pos"] = np.cumsum(counts) - counts
p2i.to_csv(
    os.path.join(odir, "p2i.csv")
)


# %%
train_proportion = 0.8
total = len(pids)
rng = np.random.default_rng(42)
perm_pids = pids[rng.permutation(total)]
train_pids = perm_pids[:int(train_proportion * total)]
val_pids = perm_pids[int(train_proportion * total):]
train_pids = np.sort(train_pids)
val_pids = np.sort(val_pids)

os.makedirs(os.path.join(odir, "participants"), exist_ok=True)
train_pids.tofile(os.path.join(odir, "participants", "train_fold.bin"))
val_pids.tofile(os.path.join(odir, "participants", "val_fold.bin"))
pids.tofile(os.path.join(odir, "participants", "all.bin"))


# %%
command = f"dx upload -r {DELPHI_DATA_DIR}"
print(f"uploading data to workspace: {command}")
start = time.time()
subprocess.run(command, shell=True) 
end = time.time()
print(f"data upload took {(end - start) / 60.0} min")


# %%

