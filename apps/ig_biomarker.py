# +
import argparse
import pprint
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from delphi.data.ukb import MultimodalUKBDataset
from delphi.data.utils import remove_after
from delphi.env import DELPHI_CKPT_DIR
from delphi.experiment import eval_iter, load_ckpt, move_batch_to_device
from delphi.explain.integrated_gradients import integrated_jacobian
from delphi.explain.utils import pack_bio, unpack_bio
from delphi.multimodal import Modality

# +
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="delphi-m4/delphi-m4/ckpt.pt")
parser.add_argument("--n-steps", type=int, default=50)
parser.add_argument(
    "--targets", type=str, help="Path to .yaml list of target disease names"
)
parser.add_argument("--subsample", type=int)
parser.add_argument("--fname", type=str)

if "ipykernel" in sys.modules:
    args = parser.parse_args([])
    args.ckpt = "delphi-m4/blood/ckpt.pt"
    args.subsample = 1000
else:
    args = parser.parse_args()

pprint.pp(vars(args))

# +
ckpt = Path(DELPHI_CKPT_DIR) / args.ckpt
model, ckpt_dict = load_ckpt(ckpt)
model.eval()
device = next(model.parameters()).device

data_args = ckpt_dict["data_args"].copy()
data_args["subject_list"] = "participants/val_fold.bin"
data_args["stats_subject_list"] = ckpt_dict["data_args"]["subject_list"]
data_args["deterministic"] = True
data_args["must_have_biomarkers"] = data_args["biomarkers"]
data_args["biomarker_dropout"] = False
data_args["first_time_only"] = True
pprint.pp(data_args)

ds = MultimodalUKBDataset(**data_args)

# ordered list of modalities present in the dataset
modalities = list(ds.mod_ds.keys())
biomarker_features = {}
for mod in modalities:
    biomarker_features[mod] = ds.mod_ds[mod].features

tokenizer = ckpt_dict["tokenizer"]
model_targets = model.targets.to(device)
model_targets = model_targets[model_targets != 1]
target_idx = model_targets.cpu().tolist()

if args.targets:
    with open(args.targets) as f:
        disease_names = yaml.safe_load(f)
    target_set = set(target_idx)
    target_idx = [tokenizer[d] for d in disease_names if tokenizer[d] in target_set]

# flat feature names: repeat each modality's features for each measurement row
all_features = []
for mod in modalities:
    all_features.extend(biomarker_features[mod])
baseline_parts = []
for mod in modalities:
    baseline_parts.append(ds.mod_ds[mod].background)
baseline_flat = torch.tensor(
    np.concatenate(baseline_parts), dtype=torch.float32, device=device
)

ig_mode = "reverse" if len(target_idx) < len(all_features) else "forward"
print(
    f"modalities: {[m.name for m in modalities]}, "
    f"n_features_total: {len(all_features)}, n_targets: {len(target_idx)}, "
    f"ig_mode: {ig_mode}"
)


# -
def ig_forward(
    flat, *, model, x0, t0, bio_x_dict_shapes, bio_T, bio_M, modalities, target_idx
):
    bio_x_dict = unpack_bio(flat, bio_x_dict_shapes, modalities)
    out, _, _ = model(x0, t0, bio_x_dict, bio_T, bio_M)
    return out["logits"][:, -1, target_idx].squeeze(0)  # (n_targets,)


# +
pids_list = []
ig_list = []
logits_list = []
n = len(ds) if args.subsample is None else args.subsample

for batch_idx in tqdm(
    eval_iter(total_size=n, batch_size=1),
    total=n,
):
    batch = ds.get_batch(batch_idx)
    batch = move_batch_to_device(batch, device)
    x0, t0, bio_X_dict, bio_t, bio_m, x1, t1 = batch

    cutoff_t = bio_t.max()
    x0, t0, bio_X_dict, bio_t, bio_m = remove_after(
        x0, t0, bio_X_dict, bio_t, bio_m, cutoff_t
    )

    out, _, _ = model(x0, t0, bio_X_dict, bio_t, bio_m)
    logits = out["logits"][:, -1, target_idx].detach().cpu().numpy()
    pid = ds.participants[batch_idx][0]

    inputs = pack_bio(bio_X_dict, modalities).float().detach()
    baselines = baseline_flat

    forward_func = partial(
        ig_forward,
        model=model,
        x0=x0,
        t0=t0,
        bio_x_dict_shapes=bio_X_dict,
        bio_T=bio_t,
        bio_M=bio_m,
        modalities=modalities,
        target_idx=target_idx,
    )

    with torch.enable_grad():
        ig = integrated_jacobian(
            forward_func,
            inputs,
            baselines,
            n_steps=args.n_steps,
            mode=ig_mode,
        )  # (n_targets, n_flat)

    ig = ig.detach().cpu().numpy()  # (n_targets, n_flat)

    # extinguish: NaN out targets already occurred in this person's history
    occurred = set(x0.ravel().cpu().tolist())
    extinguished = np.array([t in occurred for t in target_idx])
    ig[extinguished, :] = np.nan
    logits[:, extinguished] = -np.inf

    pids_list.append(int(pid))
    ig_list.append(ig.astype(np.float32))
    logits_list.append(logits.ravel().astype(np.float32))
# -


# +
pids = np.array(pids_list, dtype=np.int64)
ig_attrs = np.stack(ig_list)  # (N, n_targets, n_flat)
logits = np.stack(logits_list)  # (N, n_targets)

dirname = args.fname or "ig-biomarker"
out_dir = ckpt.parent / dirname
out_dir.mkdir(exist_ok=True)

np.save(out_dir / "pids.npy", pids)
np.save(out_dir / "attributions.npy", ig_attrs)
np.save(out_dir / "logits.npy", logits)
np.save(out_dir / "features.npy", np.array(all_features))
np.save(out_dir / "targets.npy", target_idx)

print(f"saved to {out_dir}  (N={len(pids)}, attributions={ig_attrs.shape})")
