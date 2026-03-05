# +
import argparse
import gzip
import math
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
pprint.pp(data_args)

ds = MultimodalUKBDataset(**data_args)

tokenizer = ckpt_dict["tokenizer"]
detokenizer = {v: k for k, v in tokenizer.items()}

model_targets = model.targets.to(device)
model_targets = model_targets[model_targets != 1]
target_names = [detokenizer[int(tid)] for tid in model_targets]
target_idx = model_targets.cpu().tolist()

feature_names = ds.mod_ds[modality].features  # list[str], length n_features
print(
    f"modality: {modality.name}, n_features: {len(feature_names)}, n_targets: {len(model_targets)}"
)


# -


def _sal_forward(
    bio_x, *, model, x0, t0, bio_x_rest, bio_T, bio_M, x1, t1, modality, target_idx
):
    bio_x_dict = {**bio_x_rest, modality: bio_x}
    out, _, _ = model(x0, t0, bio_x_dict, bio_T, bio_M, x1, t1)
    return out["logits"][:, -1, target_idx]  # (B, n_targets)


# +
results = {}
n = len(ds) if args.subsample is None else args.subsample

for batch_idx in tqdm(
    eval_iter(total_size=n, batch_size=1),
    total=math.ceil(n / 1),
):
    batch = ds.get_batch(batch_idx)
    batch = move_batch_to_device(batch, device)
    x0, t0, bio_X_dict, bio_T, bio_M, x1, t1 = batch
    pids = ds.participants[batch_idx]

    n_meas = (bio_M[0] == modality.value).sum().item()
    bio_x = bio_X_dict[modality].float().detach()  # (n_meas, n_features)

    forward_func = partial(
        _sal_forward,
        model=model,
        x0=x0,
        t0=t0,
        bio_x_rest={m: v.detach() for m, v in bio_X_dict.items() if m != modality},
        bio_T=bio_T,
        bio_M=bio_M,
        x1=x1,
        t1=t1,
        modality=modality,
        target_idx=target_idx,
    )

    with torch.no_grad():
        out, _, _ = model(x0, t0, bio_X_dict, bio_T, bio_M, x1, t1)
        timestamp = out["age"][:, -1]  # (B,) — age at last target position

    with torch.enable_grad():
        jac = torch.func.jacfwd(forward_func)(
            bio_x
        )  # (1, n_targets, n_meas, n_features)
    jac = jac[0]  # (n_targets, n_meas, n_features)
    jac = jac.reshape(jac.shape[0], -1)  # (n_targets, n_meas * n_features)
    jac = jac.T  # (n_meas * n_features, n_targets)
    if args.abs:
        jac = jac.abs()
    jac = jac.detach().cpu().numpy()

    for b, pid in enumerate(pids):
        results[int(pid)] = {
            "jacobian": jac.astype(np.float16),  # (n_meas * n_features, n_targets)
            "timestamp": float(timestamp[b]),
        }
# -


# +
results["targets"] = target_names
results["tokenizer"] = tokenizer
results["modality"] = modality.name

fname = args.fname or f"saliency-{args.modality.upper()}-ckpt-{ckpt.stem}.pkl.gz"
out_path = ckpt.parent / fname
with gzip.open(out_path, "wb") as f:
    pickle.dump(results, f)

print(f"saved {len(results) - 3} participants -> {out_path}")
