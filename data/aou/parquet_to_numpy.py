# %%
# +
import os
import numpy as np
import pandas as pd

from google.cloud import storage

from utils import Client, WORKSPACE_CDR, DATA_BUCKET
# -

df = pd.read_parquet(f"gs://{DATA_BUCKET}/aou_uk/tokens.parquet")

df.head()

vc = df['person_id'].value_counts()
single_token_pids = vc[vc == 1].index

single_token_pids

df = df[~df.person_id.isin(single_token_pids)]

df.shape

subjects = df.person_id.values.astype(np.uint32)
tokens = df.token.values.astype(np.uint32)
timesteps = df.age_in_days.values.astype(np.uint32)

os.makedirs("tmp", exist_ok=True)
tokens.astype(np.uint32).tofile("tmp/data.bin")
timesteps.astype(np.uint32).tofile("tmp/time.bin")

# +
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
p2i.to_csv("tmp/p2i.csv")
# -

client = storage.Client()
bucket = client.bucket(DATA_BUCKET)
blob_data = bucket.blob("aou_uk/data.bin")
blob_time = bucket.blob("aou_uk/time.bin")
blob_data.upload_from_filename("tmp/data.bin")
blob_time.upload_from_filename("tmp/time.bin")



# %%
