import os

os.chdir("/hps/nobackup/birney/users/sfan/Delphi")

import argparse
import pickle
import pprint

# +
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from tqdm import tqdm, trange

from delphi.data.shap import ShapMasker
from delphi.data.ukb import UKBDataset
from delphi.env import DELPHI_CKPT_DIR, DELPHI_DATA_DIR

# from delphi.model import (
#     Delphi2M as Delphi,
#     Delphi2MConfig as DelphiConfig
# )
from delphi.legacy.model import Delphi, DelphiConfig
from delphi.legacy.utils import get_batch, get_p2i
from delphi.model.transformer import shap_forward

delphi_labels = pd.read_csv("notebook/delphi_labels_chapters_colours_icd.csv")
labels = pd.read_csv("data/ukb_simulated_data/labels.csv", header=None, sep="\t")

# +
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="delphi-2m-og/ckpt.pt")

if "ipykernel" in sys.modules:
    print(f"running in jupyter notebook")
    args = parser.parse_args([])
else:
    args = parser.parse_args()

print("args:")
pprint.pp(vars(args))
# -

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = Path(DELPHI_CKPT_DIR) / args.ckpt
ckpt_dict = torch.load(
    ckpt,
    map_location=torch.device("cpu") if not torch.cuda.is_available() else None,
)
model = Delphi(DelphiConfig(**ckpt_dict["model_args"]))
pprint.pp(ckpt_dict["model_args"])
model.load_state_dict(ckpt_dict["model"])
model.eval()
model = model.to(device)

# +
# exclude_lifestyle = ckpt_dict["config"].get("exclude_lifestyle", False)
# no_event_mode = ckpt_dict["config"].get("no_event_mode", "legacy-random")
# ds = UKBDataset(
#     data_dir="ukb_real_data",
#     subject_list="participants/val_fold.bin",
#     perturb=False,
#     no_event_mode=no_event_mode,
#     exclude=exclude_lifestyle,
#     block_size=model.config.block_size,
# )
# total = len(ds)
# -

DATA_ROOT = Path(DELPHI_DATA_DIR) / "ukb_real_data"
val = np.fromfile(f"{DATA_ROOT}/val.bin", dtype=np.uint32).reshape(-1, 3)
val_p2i = get_p2i(val)
total = len(val_p2i)


# +
shaply_val = []
for person_idx in trange(total):

    # x0, t0, _, t1 = ds[person_idx]
    x0, t0, _, t1 = get_batch(
        [person_idx],
        val,
        val_p2i,
        select="left",
        block_size=64,
        device=device,
        padding="random",
        cut_batch=True,
    )
    x0, t0 = x0[t0 > -1], x0[t0 > -1]
    if (x0.numel() == 0) or (t0.numel() == 0) or (t1.numel() == 0):
        print(f"empty sequence found; skipping {person_idx}")
        continue

    x0 = x0.detach().cpu().numpy()
    t0 = t0.detach().cpu().numpy()
    t1 = t1.detach().cpu().numpy()
    time_passed = t1.max() - t0
    person_token_ids = x0

    masker = ShapMasker()
    shap_model = partial(shap_forward, model=model, doi=labels.index.values)
    explainer = shap.Explainer(
        shap_model, masker, feature_names=x0, output_names=labels[0].values
    )
    shap_values = explainer(
        [
            (x0, t0),
        ]
    )
    shaply_val.append(
        (x0, shap_values.values.astype(np.float16), time_passed, [person_idx] * len(x0))
    )


all_tokens = np.concatenate([i[0] for i in shaply_val])
all_values = np.concatenate([i[1] for i in shaply_val], axis=1)[0]
all_times_passed = np.concatenate([i[2] for i in shaply_val], axis=0)
all_people = np.concatenate([i[3] for i in shaply_val])
# -
with open(ckpt.parent / "shap_agg.pickle", "wb") as f:
    pickle.dump(
        {
            "tokens": all_tokens,
            "values": all_values,
            "times": all_times_passed,
            "model": ckpt.parent,
            "people": all_people,
        },
        f,
    )
