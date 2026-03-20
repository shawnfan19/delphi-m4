# +
import argparse
import gzip
import pickle
import pprint
import sys
from pathlib import Path

import numpy as np
import shap
from tqdm import trange

from delphi.data.ukb import MultimodalUKBDataset
from delphi.env import DELPHI_CKPT_DIR
from delphi.experiment import load_ckpt
from delphi.shap import MultimodalShapMasker, MultimodalShapModel

# +
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="delphi-m4/delphi-m4/ckpt.pt")
parser.add_argument("--biomarker_only", action="store_true")
parser.add_argument("--use_background", action="store_true")
parser.add_argument("--fname", type=str)
parser.add_argument("--subsample", type=int)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--half", action="store_true", help="Run model in float16")

if "ipykernel" in sys.modules:
    print(f"running in jupyter notebook")
    args = parser.parse_args([])
    args.ckpt = "interpret/blood_0.1/ckpt.pt"
    args.subsample = 1000
else:
    args = parser.parse_args()

print("args:")
pprint.pp(vars(args))
# -

ckpt = Path(DELPHI_CKPT_DIR) / args.ckpt
model, ckpt_dict = load_ckpt(ckpt)
if args.half:
    model.half()


data_args = ckpt_dict["data_args"].copy()
data_args["subject_list"] = "participants/val_fold.bin"
data_args["stats_subject_list"] = ckpt_dict["data_args"]["subject_list"]
data_args["deterministic"] = True
data_args["must_have_biomarkers"] = data_args["biomarkers"]
pprint.pp(data_args)

ds = MultimodalUKBDataset(**data_args)
biomarker_features = dict()
for modality, mod_ds in ds.mod_ds.items():
    biomarker_features[modality] = mod_ds.features
print(biomarker_features)
if args.use_background:
    biomarker_background = dict()
    for modality, mod_ds in ds.mod_ds.items():
        biomarker_background[modality] = mod_ds.background
else:
    biomarker_background = None

# +
shap_pickle = dict()
if args.subsample is None:
    total = len(ds)
else:
    total = args.subsample

masker = MultimodalShapMasker()

for i in trange(total, leave=False):
    x, t, bio_dict, bio_t, bio_m, _, _ = ds[i]
    pid = ds.participants[i]

    out = (x, t, bio_dict, bio_t, bio_m)

    shap_model = MultimodalShapModel(
        model=model,
        biomarker_only=args.biomarker_only,
        biomarker_features=biomarker_features,
        biomarker_background=biomarker_background,
        data=out,
    )
    explainer = shap.Explainer(
        shap_model, masker, feature_names=np.arange(len(shap_model.dummy()))
    )
    shap_values = explainer([shap_model.dummy()], batch_size=args.batch_size)

    shap_vals = shap_values.values[0]  # (mask_size, vocab)
    entry = {}

    is_no_event = x == 1
    entry["x"] = x[~is_no_event]
    entry["t"] = t[~is_no_event]
    if not args.biomarker_only:
        token_shap = shap_vals[: shap_model.n_tokens].astype(np.float16)
        entry["shap"] = token_shap[~is_no_event]

    entry["bio_t"] = bio_t
    entry["bio_m"] = bio_m
    bio_shap = shap_vals[-shap_model.n_biomarker_features :]
    entry["bio_shap"] = bio_shap.astype(np.float16)
    entry["bio_x"] = bio_dict

    shap_pickle[int(pid)] = entry
# -


shap_pickle["tokenizer"] = ds.tokenizer
shap_pickle["biomarker_features"] = biomarker_features
if args.fname is None:
    args.fname = "shap"
    if args.biomarker_only:
        args.fname += "_bio"
    if args.use_background:
        args.fname += "_bg"
with gzip.open(ckpt.parent / f"{args.fname}.pickle.gz", "wb") as f:
    pickle.dump(shap_pickle, f)
