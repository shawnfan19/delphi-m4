# +
import argparse
import gzip
import pickle
import pprint
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import yaml
from tqdm import trange

from delphi.data.ukb import MultimodalUKBDataset
from delphi.env import DELPHI_CKPT_DIR
from delphi.experiment import load_ckpt
from delphi.multimodal import Modality
from delphi.shap import MultimodalShapMasker, multimodal_shap_forward, to_shap_array

# +
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="delphi-m4/delphi-m4/ckpt.pt")
parser.add_argument("--immediate", action="store_true")
parser.add_argument("--fname", type=str)
parser.add_argument("--subsample", type=int)

if "ipykernel" in sys.modules:
    print(f"running in jupyter notebook")
    args = parser.parse_args([])
    args.ckpt = "delphi-m4/blood/ckpt.pt"
    args.immediate = True
    args.subsample = 1000
else:
    args = parser.parse_args()

print("args:")
pprint.pp(vars(args))
# -

ckpt = Path(DELPHI_CKPT_DIR) / args.ckpt
model, ckpt_dict = load_ckpt(ckpt)


data_args = ckpt_dict["data_args"].copy()
data_args["subject_list"] = "participants/val_fold.bin"
data_args["stats_subject_list"] = ckpt_dict["data_args"]["subject_list"]
data_args["deterministic"] = True
data_args["must_have_biomarkers"] = data_args["biomarkers"]
pprint.pp(data_args)

ds = MultimodalUKBDataset(**data_args)
# select participants based on biomarker values
# dynamically truncate tokens after biomarker occurrence


biomarker_features = dict()
for k, biomarker in ds.mod_ds.items():
    biomarker_features[k] = biomarker.features


# +
shap_pickle = dict()
if args.subsample is None:
    total = len(ds)
else:
    total = args.subsample

for i in trange(total, leave=False):
    x, t, bio_dict, bio_t, bio_m, _, _ = ds[i]
    pid = ds.participants[i]

    sample, _, _ = to_shap_array(
        (x, t, bio_dict, bio_t, bio_m),
        detokenizer=ds.detokenizer,
        biomarker_features=biomarker_features,
    )
    all_x, all_t, all_m = sample

    masker = MultimodalShapMasker(biomarker_features=biomarker_features)
    sizes = masker._measurement_sizes(sample)
    bio_m_flat = all_m[all_m != 1]
    bio_t_flat = all_t[all_m != 1]
    meas_features, meas_timesteps = [], []
    offset = 0
    for size in sizes:
        modval = int(bio_m_flat[offset])
        t_meas = float(bio_t_flat[offset])
        # meas_features.append(f"{Modality(modval).name}@{t_meas:.0f}")
        meas_features.append(f"{Modality(modval).name}")
        meas_timesteps.append(t_meas)
        offset += size

    shap_model = partial(
        multimodal_shap_forward, biomarker_features=biomarker_features, model=model
    )
    explainer = shap.Explainer(
        shap_model,
        masker,
        feature_names=np.array([meas_features]),
        output_names=list(ckpt_dict["tokenizer"].keys()),
    )
    shap_values = explainer([sample])

    shap_pickle[int(pid)] = {
        "shap": shap_values.values[0].astype(np.float16),  # (n_measurements, vocab)
        "features": meas_features,
        "timesteps": np.array(meas_timesteps).astype(np.float16),
    }
# -


meas_features

shap_pickle["tokenizer"] = ds.tokenizer

fname = args.fname if args.fname else "shap_missingness.pickle.gz"
with gzip.open(ckpt.parent / fname, "wb") as f:
    pickle.dump(shap_pickle, f)
