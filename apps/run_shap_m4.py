# +
import argparse
import pickle
import pprint
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
import yaml
from tqdm import trange

from delphi.data.shap import MultimodalShapMasker, shap_forward, to_shap_array
from delphi.data.ukb import MultimodalUKBDataset
from delphi.env import DELPHI_CKPT_DIR
from delphi.model.multimodal import DelphiM4, DelphiM4Config

delphi_labels = pd.read_csv("notebook/delphi_labels_chapters_colours_icd.csv")

# +
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="delphi-m4/delphi-m4/ckpt.pt")
parser.add_argument("--immediate", action="store_true")
parser.add_argument("--fname", type=str)

if "ipykernel" in sys.modules:
    print(f"running in jupyter notebook")
    args = parser.parse_args([])
    args.ckpt = "shap/blood/ckpt.pt"
    args.immediate = True
else:
    args = parser.parse_args()

print("args:")
pprint.pp(vars(args))

# +
ckpt = Path(DELPHI_CKPT_DIR) / args.ckpt

ckpt_dict = torch.load(
    ckpt, map_location=torch.device("cpu") if not torch.cuda.is_available() else None
)
model_cfg = DelphiM4Config(**ckpt_dict["model_args"])
model = DelphiM4(model_cfg)
model.load_state_dict(ckpt_dict["model"])

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print(f"model: {ckpt} [iter: {ckpt_dict['iter_num']}]")
# -
data_args = ckpt_dict["data_args"].copy()
data_args["subject_list"] = "participants/val_fold.bin"
data_args["stats_subject_list"] = ckpt_dict["data_args"]["subject_list"]
data_args["deterministic"] = True
pprint.pp(data_args)


ds = MultimodalUKBDataset(**data_args)


biomarker_features = dict()
biomarker_background = dict()
for k, biomarker in ds.mod_ds.items():
    biomarker_features[k] = biomarker.features
    biomarker_background[k] = biomarker.mask

# +
flat_biomarker_features = list()
for _modality, features in biomarker_features.items():
    flat_biomarker_features.extend([f"{_modality.name}.{f}" for f in features])

shap_tokenizer = ds.tokenizer.copy()
offset = len(shap_tokenizer)
for i, f in enumerate(flat_biomarker_features):
    shap_tokenizer[f] = offset + i
# -


# +
shap_pickle = dict()
total = len(ds)

for i in trange(total, leave=False):
    x, t, bio_dict, bio_t, bio_m, _, _ = ds[i]
    pid = ds.participants[i]

    sample, features, bio_bg = to_shap_array(
        (x, t, bio_dict, bio_t, bio_m),
        detokenizer=ds.detokenizer,
        biomarker_features=biomarker_features,
        biomarker_background=biomarker_background,
    )
    all_x, all_t, all_m = sample
    feature_tokens = np.array([shap_tokenizer[f] for f in features])
    no_event = np.array(["no_event" in feature for feature in features]).astype(bool)

    masker = MultimodalShapMasker(bio_bg)
    shap_model = partial(shap_forward, model=model)
    explainer = shap.Explainer(
        shap_model,
        masker,
        feature_names=np.array([features]),
        output_names=delphi_labels["name"].values,
    )
    shap_values = explainer([sample])

    shap_pickle[int(pid)] = {
        "shap": shap_values.values[0, ~no_event, :].astype(np.float16),
        "features": feature_tokens[~no_event],
        "timesteps": all_t[~no_event],
    }

with open(ckpt.parent / f"shap.pickle", "wb") as f:
    pickle.dump(shap_pickle, f)

with open(ckpt.parent / "shap_tokenizer.yaml", "w") as f:
    yaml.dump(shap_tokenizer, f, default_flow_style=False)
# -
