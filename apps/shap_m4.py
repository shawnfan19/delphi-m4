# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import argparse
import copy
import pprint

# %%
import sys
from collections import defaultdict
from functools import partial
from itertools import groupby
from pathlib import Path

import numpy as np
import shap
import torch

from delphi.data.ukb import MultimodalUKBDataset, ShapMasker
from delphi.data.utils import collate_batch, sort_by_time
from delphi.env import DELPHI_CKPT_DIR
from delphi.model.multimodal import DelphiM4, DelphiM4Config
from delphi.multimodal import Modality

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="delphi-m4/delphi-m4/ckpt.pt")
parser.add_argument("--fname", type=str)

if "ipykernel" in sys.modules:
    print(f"running in jupyter notebook")
    args = parser.parse_args([])
    args.ckpt = "fusion/blood-early/ckpt.pt"
else:
    args = parser.parse_args()

print("args:")
pprint.pp(vars(args))

# %%
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

# %%
data_args = ckpt_dict["data_args"].copy()
data_args["subject_list"] = "participants/val_fold.bin"
data_args["stats_subject_list"] = ckpt_dict["data_args"]["subject_list"]
pprint.pp(data_args)

# %%
ds = MultimodalUKBDataset(**data_args)

# %%
MultimodalOut = tuple[
    np.ndarray, np.ndarray, dict[Modality, list[np.ndarray]], np.ndarray, np.ndarray
]

ShapArray = tuple[np.ndarray, np.ndarray, np.ndarray]


# order-preserving np.unique
def np_unique_1d(array: np.ndarray):
    _, idx = np.unique(array, return_index=True)
    return array[np.sort(idx)]


def to_shap_array(s: MultimodalOut) -> ShapArray:

    all_x, all_t, all_m = list(), list(), list()
    x, t, bio_x_dict, bio_t, bio_m = s
    all_x.append(x.astype(np.float32))
    all_t.append(t)
    all_m.append(np.ones_like(x))

    modvals = np_unique_1d(bio_m)
    for modval in modvals:
        for _bio_x, _bio_t in zip(bio_x_dict[Modality(modval)], bio_t[bio_m == modval]):
            all_x.append(_bio_x)
            all_t.append(np.full_like(_bio_x, fill_value=_bio_t))
            all_m.append(np.full_like(_bio_x, fill_value=modval))
    all_x = np.concatenate(all_x)
    all_t = np.concatenate(all_t)
    all_m = np.concatenate(all_m)

    return all_x, all_t, all_m


def from_shap_array(s: ShapArray) -> MultimodalOut:

    all_x, all_t, all_m = s
    token_mask = all_m == 1
    x, t = all_x[token_mask].astype(np.int64), all_t[token_mask].astype(np.float32)

    bio_x_dict, bio_t, bio_m = dict(), list(), list()

    # modvals = np_unique_1d(all_m)
    modvals = np.unique(all_m)
    for modval in modvals:
        if modval <= 1:
            continue
        mod_mask = all_m == modval
        mod_x, mod_t = all_x[mod_mask], all_t[mod_mask]
        modality = Modality(modval)
        bio_x_dict[modality] = list()

        for uniq_mod_t in np.sort(np.unique(mod_t)):
            bio_x_dict[modality].append(mod_x[mod_t == uniq_mod_t])
            bio_t.append(uniq_mod_t)
            bio_m.append(modval)
    bio_t, bio_m = np.array(bio_t).astype(np.float32), np.array(bio_m).astype(np.int64)
    bio_t, bio_m = sort_by_time(bio_t, bio_m)

    return x, t, bio_x_dict, bio_t, bio_m


def collapse_consec_duplicates(array: np.ndarray):
    result = [(key, len(list(group))) for key, group in groupby(array)]
    vals, counts = zip(*result)
    return np.array(vals), np.array(counts)


class MultimodalShapMask(shap.maskers.Masker):

    def __init__(
        self,
        biomarker_masks: dict[Modality, np.ndarray],
    ):
        self.base_masker = ShapMasker()
        self.biomarker_masks = biomarker_masks
        self.biomarker_sizes = {k: v.size for k, v in self.biomarker_masks.items()}
        self.slice_dict = None

    def shape(self, s: ShapArray):
        return (1, len(s[0]))

    def mask_shapes(self, s: ShapArray):
        return [(len(s[0]),)]

    def __call__(self, mask, s: ShapArray) -> ShapArray:
        mask = self._standardize_mask(mask, s)
        all_x, all_t, all_m = s
        all_x, all_t, all_m = all_x.copy(), all_t.copy(), all_m.copy()

        is_tok = all_m == 1
        x, t = all_x[is_tok], all_t[is_tok]
        tok_mask = mask[is_tok]
        ((x,), (t,)) = self.base_masker(mask=tok_mask, s=(x, t))

        all_bio_x, all_bio_t, all_bio_m = all_x[~is_tok], all_t[~is_tok], all_m[~is_tok]
        bio_mask = mask[~is_tok]

        modvals, modval_counts = collapse_consec_duplicates(all_bio_m)

        biomarker_mask = list()
        for i, modval in enumerate(modvals):
            if modval == 0:
                continue
            modality = Modality(modval)
            assert (
                modval_counts[i] % self.biomarker_sizes[modality] == 0
            ), f"{modval_counts[i]}, {self.biomarker_sizes[modality]}"
            n = int(modval_counts[i] / self.biomarker_sizes[modality])
            for _ in range(n):
                biomarker_mask.append(self.biomarker_masks[modality])
        biomarker_mask = np.concatenate(biomarker_mask)
        assert biomarker_mask.size == bio_mask.size
        all_bio_x[~bio_mask] = biomarker_mask[~bio_mask]

        all_x = np.concatenate([x, all_bio_x])
        all_t = np.concatenate([t, all_bio_t])

        return (
            (all_x,),
            (all_t,),
            (all_m,),
        )


def parse_features(
    s: ShapArray, detokenizer: dict, biomarker_features: dict[Modality, list]
):

    all_x, all_t, all_m = s
    features = list()

    is_tok = all_m == 1
    x, t = all_x[is_tok], all_t[is_tok]
    all_bio_m = all_m[~is_tok]

    token_features = [f"{detokenizer[int(_x)]} – {int(_t)}" for _x, _t in zip(x, t)]
    features.extend(token_features)

    modvals, modval_counts = collapse_consec_duplicates(all_bio_m)
    biomarker_sizes = {k: len(v) for k, v in biomarker_features.items()}
    for i, modval in enumerate(modvals):
        if modval == 0:
            continue
        modality = Modality(modval)
        assert (
            modval_counts[i] % biomarker_sizes[modality] == 0
        ), f"{modval_counts[i]}, {biomarker_sizes[modality]}"
        n = int(modval_counts[i] / biomarker_sizes[modality])
        for _ in range(n):
            _features = [f"{f}" for f in biomarker_features[modality]]
            features.extend(_features)

    return features


def dummy_forward(x_lst, t_lst, bio_x_dict_lst, bio_t_lst, bio_m_lst):

    return np.random.random((x_lst.shape[0], 5))


def shap_forward(
    all_x_lst: list[np.ndarray],
    all_t_lst: list[np.ndarray],
    all_m_lst: list[np.ndarray],
    model,
    doi: list[int],
):
    x_lst, t_lst = list(), list()
    biomarker, bio_t_lst, bio_m_lst = defaultdict(list), list(), list()

    for all_x, all_t, all_m in zip(all_x_lst, all_t_lst, all_m_lst):
        x, t, bio_x_dict, bio_t, bio_m = from_shap_array((all_x, all_t, all_m))
        x_lst.append(x)
        t_lst.append(t)
        for modality in bio_x_dict.keys():
            biomarker[modality].extend(bio_x_dict[modality])
        bio_t_lst.append(bio_t)
        bio_m_lst.append(bio_m)

    device = next(model.parameters()).device
    idx = collate_batch(x_lst)
    idx = torch.tensor(idx, dtype=torch.long).to(device)
    age = collate_batch(t_lst, fill_value=-1e4)
    age = torch.tensor(age, dtype=torch.float32).to(device)
    for modality, bio_x_lst in biomarker.items():
        biomarker[modality] = torch.from_numpy(np.stack(bio_x_lst)).to(device)
    mod_age = collate_batch(bio_t_lst, fill_value=-1e4)
    mod_age = torch.tensor(mod_age, dtype=torch.float32).to(device)
    mod_idx = collate_batch(bio_m_lst)
    mod_idx = torch.tensor(mod_idx, dtype=torch.long).to(device)

    outputs, _, _ = model.forward(idx, age, biomarker, mod_age, mod_idx)
    logits = outputs["logits"]
    doi_logits = logits[:, -1, doi].detach().cpu().numpy()

    return doi_logits


# %%

# %%
# x, t, bio_x_dict, bio_t, bio_m, _, _ = ds[20]
# all_x, all_t, all_m = to_shap_array((x, t, bio_x_dict, bio_t, bio_m))
# assert all_x.shape == all_t.shape == all_m.shape
# _x, _t, _bio_x_dict, _bio_t, _bio_m = from_shap_array((all_x, all_t, all_m))

# %%

# %%
biomarker_masks = dict()
for modality, biomarker in ds.mod_ds.items():
    if not isinstance(biomarker.mask, np.ndarray):
        biomarker_masks[modality] = np.array([biomarker.mask])
    else:
        biomarker_masks[modality] = biomarker.mask
biomarker_masks

# %%
x, t, bio_x_dict, bio_t, bio_m, _, _ = ds[100]
all_x, all_t, all_m = to_shap_array((x, t, bio_x_dict, bio_t, bio_m))
masker = MultimodalShapMask(biomarker_masks)
mask = np.zeros_like(all_x).astype(bool)
masker(mask=mask, s=(all_x, all_t, all_m))

# %%
biomarker_features = dict()
for modality, biomarker in ds.mod_ds.items():
    biomarker_features[modality] = biomarker.features
biomarker_features

# %%
s = (x, t, bio_x_dict, bio_t, bio_m)
all_x, all_t, all_m = to_shap_array(s)
s = (all_x, all_t, all_m)
features = parse_features(
    s=s, detokenizer=ds.detokenizer, biomarker_features=biomarker_features
)
len(features), masker.shape(s)

# %%
features

# %%
sample = to_shap_array((x, t, bio_x_dict, bio_t, bio_m))
masker = MultimodalShapMask(biomarker_masks)
# shap_model = partial(model.shap_forward, doi=[1269, 1000])
shap_model = partial(shap_forward, model=model, doi=[1269, 1000])
explainer = shap.Explainer(
    shap_model, masker, feature_names=features, output_names=["death"]
)
shap_values = explainer(
    [
        sample,
    ]
)

# %%

# %%
shap_values.data = np.array([features])
shap_values.shape

# %%
shap_values.data

# %%
shap.plots.waterfall(shap_values[0, ..., 0])

# %%
