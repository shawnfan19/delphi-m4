# +
import argparse
import pprint
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from delphi.data import MultimodalDataset
from delphi.data.transform import BiomarkerTransform, MultimodalPrompt, TokenTransform
from delphi.data.ukb import (
    Biomarker,
    MultimodalUKBReader,
    filter_participants_with_biomarkers,
)
from delphi.env import DELPHI_CKPT_DIR
from delphi.experiment import eval_iter, load_ckpt, move_batch_to_device

# +
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="delphi-m4/delphi-m4/ckpt.pt")
parser.add_argument("--modality", type=str, help="Biomarker name, e.g. lipid")
parser.add_argument(
    "--abs", action="store_true", help="Store absolute value of gradients"
)
parser.add_argument("--subsample", type=int)
parser.add_argument("--fname", type=str)

if "ipykernel" in sys.modules:
    args = parser.parse_args([])
    args.ckpt = "delphi-m4/blood/ckpt.pt"
    args.modality = "lipid"
    args.subsample = 1000
else:
    args = parser.parse_args()

pprint.pp(vars(args))

# +
ckpt = Path(DELPHI_CKPT_DIR) / args.ckpt
model, ckpt_dict = load_ckpt(ckpt)
model.eval()
device = next(model.parameters()).device

mod_name = args.modality.lower()
biomarker2idx = model.config.biomarker2idx
assert (
    mod_name in biomarker2idx
), f"modality {mod_name!r} not in ckpt biomarker2idx: {sorted(biomarker2idx)}"

reader_args = ckpt_dict["reader_args"]
pprint.pp(
    {
        "reader_args": reader_args,
        "token_transform_args": ckpt_dict["token_transform_args"],
        "biomarker_transform_args": ckpt_dict.get("biomarker_transform_args"),
    }
)

# pass dict (not list) so reader uses the ckpt's index assignments
reader = MultimodalUKBReader(
    biomarkers=biomarker2idx or None,
    expansion_packs=reader_args["expansion_packs"],
)
token_transform = TokenTransform.from_ckpt(ckpt_dict)
biomarker_transform = BiomarkerTransform.from_ckpt(ckpt_dict)
if biomarker_transform is not None:
    biomarker_transform = biomarker_transform.replace(
        first_time_only=True, dropout=None, deterministic=True
    )

val_pids = MultimodalUKBReader.participants("val")
total_val = val_pids.size
val_pids = filter_participants_with_biomarkers(
    val_pids, biomarkers=[mod_name], any=True
)
print(f"{val_pids.size} / {total_val} val pids (has {mod_name})")

# cutoff at the (first/only) measurement time of the target modality
cutoff = Biomarker.first_occurrence_times(mod_name, val_pids)
prompt_transform = MultimodalPrompt(
    prompt_age={int(pid): float(age) for pid, age in zip(val_pids, cutoff)},
    biomarker2idx=reader.biomarker2idx,
    append_no_event=False,
)

ds = MultimodalDataset(
    reader=reader,
    pids=val_pids,
    token_transform=token_transform,
    biomarker_transform=biomarker_transform,
    prompt_transform=prompt_transform,
)

biomarker_features = {name: bm.features for name, bm in reader.biomarkers.items()}
model_targets = model.targets.to(device)
model_targets = model_targets[model_targets != 1]
target_idx = model_targets.cpu().tolist()
print(
    f"modality: {mod_name}, "
    f"n_features: {len(biomarker_features[mod_name])}, "
    f"n_targets: {len(model_targets)}"
)


# -
def saliency_forward(
    bio_x, *, model, x0, t0, bio_x_rest, bio_T, bio_M, mod_name, target_idx
):
    bio_x_dict = {**bio_x_rest, mod_name: bio_x}
    out, _, _ = model(x0, t0, biomarker=bio_x_dict, mod_age=bio_T, mod_idx=bio_M)
    return out["logits"][:, -1, target_idx]  # (B, n_targets)


# +
pids_list = []
jacobians_list = []
logits_list = []
n = len(ds) if args.subsample is None else args.subsample

for batch_idx in tqdm(
    eval_iter(total_size=n, batch_size=1),
    total=n,
):
    batch = ds.get_batch(batch_idx)
    batch = move_batch_to_device(batch, device)
    x0, t0, bio_X_dict, bio_t, bio_m, x1, t1 = batch

    out, _, _ = model(x0, t0, biomarker=bio_X_dict, mod_age=bio_t, mod_idx=bio_m)
    logits = out["logits"][:, -1, target_idx].detach().cpu().numpy()
    pid = ds.participants[batch_idx][0]

    bio_x = bio_X_dict[mod_name].float().detach()  # (n_meas, n_features)

    forward_func = partial(
        saliency_forward,
        model=model,
        x0=x0,
        t0=t0,
        bio_x_rest={m: v.detach() for m, v in bio_X_dict.items() if m != mod_name},
        bio_T=bio_t,
        bio_M=bio_m,
        mod_name=mod_name,
        target_idx=target_idx,
    )

    with torch.enable_grad():
        jac = torch.func.jacfwd(forward_func)(
            bio_x
        )  # (1, n_targets, n_meas, n_features)
    jac = jac[0]  # (n_targets, n_meas, n_features)
    jac = jac.reshape(jac.shape[0], -1)  # (n_targets, n_meas * n_features)
    jac = jac.T.detach().cpu().numpy()  # (n_meas * n_features, n_targets)

    # extinguish: NaN out targets already occurred in this person's history
    occurred = set(x0.ravel().cpu().tolist())
    extinguished = np.array([t in occurred for t in target_idx])
    jac[:, extinguished] = np.nan
    logits[:, extinguished] = -np.inf

    pids_list.append(int(pid))
    jacobians_list.append(jac.astype(np.float32))
    logits_list.append(logits.ravel().astype(np.float32))
# -


# +
pids = np.array(pids_list, dtype=np.int64)
jacobians = np.stack(jacobians_list)  # (N, n_features, n_targets)
logits = np.stack(logits_list)  # (N, n_targets)

dirname = args.fname or f"saliency-{mod_name}"
out_dir = ckpt.parent / dirname
out_dir.mkdir(exist_ok=True)

np.save(out_dir / "pids.npy", pids)
np.save(out_dir / "jacobians.npy", jacobians)
np.save(out_dir / "logits.npy", logits)

print(f"saved to {out_dir}  (N={len(pids)}, jacobians={jacobians.shape})")
