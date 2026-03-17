# +
import argparse
import gzip
import pickle
import pprint
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from delphi.data.ukb import MultimodalUKBDataset
from delphi.env import DELPHI_CKPT_DIR
from delphi.experiment import eval_iter, load_ckpt, move_batch_to_device
from delphi.multimodal import Modality

# +
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="delphi-m4/delphi-m4/ckpt.pt")
parser.add_argument("--modality", type=str, help="Biomarker modality, e.g. LIPID")
parser.add_argument(
    "--abs", action="store_true", help="Store absolute value of gradients"
)
parser.add_argument("--subsample", type=int)
parser.add_argument("--fname", type=str)

if "ipykernel" in sys.modules:
    args = parser.parse_args([])
    args.ckpt = "delphi-m4/blood/ckpt.pt"
    args.modality = "LIPID"
    args.subsample = 1000
else:
    args = parser.parse_args()

pprint.pp(vars(args))

# +
ckpt = Path(DELPHI_CKPT_DIR) / args.ckpt
model, ckpt_dict = load_ckpt(ckpt)
model.eval()
device = next(model.parameters()).device

modality = Modality[args.modality.upper()]

data_args = ckpt_dict["data_args"].copy()
data_args["subject_list"] = "participants/val_fold.bin"
data_args["stats_subject_list"] = ckpt_dict["data_args"]["subject_list"]
data_args["deterministic"] = True
data_args["must_have_biomarkers"] = data_args["biomarkers"]
data_args["biomarker_dropout"] = False
data_args["first_time_only"] = True
pprint.pp(data_args)

ds = MultimodalUKBDataset(**data_args)

biomarker_features = {}
for mod, mod_ds in ds.mod_ds.items():
    biomarker_features[mod] = mod_ds.features
tokenizer = ckpt_dict["tokenizer"]
model_targets = model.targets.to(device)
model_targets = model_targets[model_targets != 1]
target_idx = model_targets.cpu().tolist()
print(
    f"modality: {modality.name}, n_features: {len(biomarker_features[modality])}, n_targets: {len(model_targets)}"
)


def remove_after_biomarker(x, t, bio_x_dict, bio_t, bio_m, biomarker):
    b_t = bio_t[bio_m == Modality(biomarker).value].max()

    # filter disease events
    x_mask = t.ravel() <= b_t
    x = x[:, x_mask].clone()
    t = t[:, x_mask].clone()

    # filter biomarker measurements from other modalities
    bio_mask = bio_t.ravel() <= b_t
    bio_x_dict = {
        mod: v[bio_mask[bio_m.ravel() == mod.value]].clone()
        for mod, v in bio_x_dict.items()
    }
    bio_t = bio_t[:, bio_mask].clone()
    bio_m = bio_m[:, bio_mask].clone()

    return x, t, bio_x_dict, bio_t, bio_m


# -
def _sal_forward(
    bio_x, *, model, x0, t0, bio_x_rest, bio_T, bio_M, modality, target_idx
):
    bio_x_dict = {**bio_x_rest, modality: bio_x}
    out, _, _ = model(x0, t0, bio_x_dict, bio_T, bio_M)
    return out["logits"][:, -1, target_idx]  # (B, n_targets)


# +
results = {}
n = len(ds) if args.subsample is None else args.subsample

for batch_idx in tqdm(
    eval_iter(total_size=n, batch_size=1),
    total=n,
):
    batch = ds.get_batch(batch_idx)
    batch = move_batch_to_device(batch, device)
    x0, t0, bio_X_dict, bio_t, bio_m, x1, t1 = batch

    x0, t0, bio_X_dict, bio_t, bio_m = remove_after_biomarker(
        x0, t0, bio_X_dict, bio_t, bio_m, modality
    )

    out, _, _ = model(x0, t0, bio_X_dict, bio_t, bio_m)
    logits = out["logits"][:, -1, target_idx].detach().cpu().numpy()
    pid = ds.participants[batch_idx][0]

    bio_x = bio_X_dict[modality].float().detach()  # (n_meas, n_features)

    forward_func = partial(
        _sal_forward,
        model=model,
        x0=x0,
        t0=t0,
        bio_x_rest={m: v.detach() for m, v in bio_X_dict.items() if m != modality},
        bio_T=bio_t,
        bio_M=bio_m,
        modality=modality,
        target_idx=target_idx,
    )

    with torch.enable_grad():
        jac = torch.func.jacfwd(forward_func)(
            bio_x
        )  # (1, n_targets, n_meas, n_features)
    jac = jac[0]  # (n_targets, n_meas, n_features)
    jac = jac.reshape(jac.shape[0], -1)  # (n_targets, n_meas * n_features)
    jac = jac.T.detach().cpu().numpy()  # (n_meas * n_features, n_targets)

    if data_args["z_score_biomarkers"]:
        jac = jac / np.expand_dims(ds.mod_ds[modality].std, axis=1)

    raw_bio_x = dict()
    for m, v in bio_X_dict.items():
        raw_bio_x[m] = ds.mod_ds[m].untransform(v.cpu().numpy())

    results[int(pid)] = {
        "logits": logits.astype(np.float32),
        "jacobian": jac.astype(np.float32),  # (n_meas * n_features, n_targets)
        "x": x0.ravel().cpu().numpy(),
        "t": t0.ravel().cpu().numpy(),
        "bio_t": bio_t.ravel().cpu().numpy(),
        "bio_m": bio_m.ravel().cpu().numpy(),
        "bio_x": raw_bio_x,
    }
# -


# +
results["targets"] = model_targets.cpu().numpy()
results["tokenizer"] = tokenizer
results["biomarker"] = modality.name
results["biomarker_features"] = biomarker_features

fname = args.fname or f"saliency-{args.modality.upper()}-ckpt-{ckpt.stem}.pkl.gz"
out_path = ckpt.parent / fname
with gzip.open(out_path, "wb") as f:
    pickle.dump(results, f)

print(f"saved to {out_path}")
