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

from delphi.data.ukb import UKBDataset
from delphi.env import DELPHI_CKPT_DIR
from delphi.experiment import load_ckpt
from delphi.shap import ShapMasker, ShapModel

# +
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="delphi-2m/ckpt.pt")
parser.add_argument("--fname", type=str, default="shap")
parser.add_argument("--subsample", type=int)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--half", action="store_true", help="Run model in float16")

if "ipykernel" in sys.modules:
    print("running in jupyter notebook")
    args = parser.parse_args([])
    args.subsample = 100
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
data_args["deterministic"] = True
pprint.pp(data_args)

ds = UKBDataset(**data_args)

# +
shap_pickle = dict()
total = len(ds) if args.subsample is None else args.subsample

masker = ShapMasker()

for i in trange(total, leave=False):
    x, t, _, _ = ds[i]
    pid = ds.participants[i]

    shap_model = ShapModel(model=model, data=(x, t))
    explainer = shap.Explainer(
        shap_model, masker, feature_names=np.arange(len(shap_model.dummy()))
    )
    shap_values = explainer([shap_model.dummy()], batch_size=args.batch_size)

    shap_vals = shap_values.values[0]  # (n_tokens, vocab)

    is_no_event = x == 1
    entry = {
        "x": x[~is_no_event],
        "t": t[~is_no_event],
        "shap": shap_vals[~is_no_event].astype(np.float16),
    }
    shap_pickle[int(pid)] = entry
# -

shap_pickle["tokenizer"] = ds.tokenizer
with gzip.open(ckpt.parent / f"{args.fname}.pickle.gz", "wb") as f:
    pickle.dump(shap_pickle, f)
