# +
import argparse
import pprint
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from delphi.data.ukb import MultimodalUKBDataset
from delphi.data.utils import remove_after_np
from delphi.env import DELPHI_CKPT_DIR
from delphi.experiment import load_ckpt
from delphi.multimodal import Modality

# +
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str)
parser.add_argument("--saliency-dir", type=str)
parser.add_argument("--modality", type=str)
parser.add_argument("--feature", type=str)
parser.add_argument("--target", type=str)
parser.add_argument("--alpha", type=float, default=1.0)
parser.add_argument("--top-k", type=int, default=20)
parser.add_argument("--bin-lo", type=float, default=None)
parser.add_argument("--bin-hi", type=float, default=None)

if "ipykernel" in sys.modules:
    args = parser.parse_args([])
    args.ckpt = "interpret/blood/ckpt.pt"
    args.alpha = 100

    args.modality = "RENAL"
    args.saliency_dir = "saliency-RENAL"
    args.feature = "creatinine"
    args.target = "n18_(chronic_renal_failure)"
    args.bin_lo = 50
    args.bin_hi = 100

    # args.modality = "LFT"
    # args.saliency_dir = "saliency-LFT"
    # args.feature = "gamma_glutamyltransferase"
    # args.target = "k70_(alcoholic_liver_disease)"
    # args.bin_lo = 0
    # args.bin_hi = 100

    # args.modality = "LIPID"
    # args.saliency_dir = "saliency-LIPID"
    # args.feature = "ldl_direct"
    # args.target = "i21_(acute_myocardial_infarction)"
    # args.bin_lo = 3.9
    # args.bin_hi = 4.3
else:
    args = parser.parse_args()

pprint.pp(vars(args))

# +
ckpt_path = Path(DELPHI_CKPT_DIR) / args.ckpt
model, ckpt_dict = load_ckpt(ckpt_path)
tokenizer = ckpt_dict["tokenizer"]
detokenizer = {v: k for k, v in tokenizer.items()}

model_targets = model.targets
model_targets = model_targets[model_targets != 1].cpu().numpy()
target_idx = model_targets.tolist().index(tokenizer[args.target])

# load saliency outputs
saliency_dir = ckpt_path.parent / args.saliency_dir
jacobians = np.load(saliency_dir / "jacobians.npy", mmap_mode="r")
sal_pids = np.load(saliency_dir / "pids.npy")

data_args = ckpt_dict["data_args"].copy()
data_args["subject_list"] = sal_pids
data_args["stats_subject_list"] = ckpt_dict["data_args"]["subject_list"]
data_args["deterministic"] = True
data_args["must_have_biomarkers"] = data_args["biomarkers"]
data_args["z_score_biomarkers"] = False
data_args["biomarker_dropout"] = False
data_args["first_time_only"] = True

ds = MultimodalUKBDataset(**data_args)

# align jacobians with ds.participants (must_have_biomarkers may have dropped some)
keep_mask = np.isin(sal_pids, ds.participants)
jacobians = jacobians[keep_mask]
sal_pids = sal_pids[keep_mask]

# +
modality = Modality[args.modality.upper()]
bio = ds.mod_ds[modality]
feature_idx = bio.feat2idx[args.feature]
print(f"explaining saliency of {args.feature} → {args.target}")

# y: saliency value per participant
y = jacobians[:, feature_idx, target_idx].copy()

# +
# build covariate matrix: token presence + biomarker values + age
# using ds[i] + remove_after_np to get temporally consistent covariates
exclude_ids = {tokenizer.get("padding", 0), tokenizer.get("no_event", 1)}
token_names = sorted(
    [(name, tok) for name, tok in tokenizer.items() if tok not in exclude_ids],
    key=lambda x: x[1],
)
token_name_list = [name for name, _ in token_names]
token_ids = np.array([tok for _, tok in token_names])

all_bio_features = []
for mod, mod_ds in ds.mod_ds.items():
    all_bio_features.extend([f"{mod.name}:{f}" for f in mod_ds.features])
target_feature_col = all_bio_features.index(f"{modality.name}:{args.feature}")

n = len(ds)
token_matrix = np.zeros((n, len(token_ids)), dtype=np.float32)
bio_values = np.zeros((n, len(all_bio_features)), dtype=np.float32)
age = np.zeros(n, dtype=np.float32)

for i in tqdm(range(n), desc="building covariate matrix"):
    x0, t0, bio_x_dict, bio_t, bio_m, x1, t1 = ds[i]

    # remove tokens and biomarkers after the target modality measurement
    cutoff_t = bio_t[bio_m == modality.value].max()
    x0, t0, bio_x_dict, bio_t, bio_m = remove_after_np(
        x0, t0, bio_x_dict, bio_t, bio_m, cutoff_t
    )

    # token presence
    present = np.isin(token_ids, x0)
    token_matrix[i] = present.astype(np.float32)

    # biomarker values
    col = 0
    for mod, mod_ds in ds.mod_ds.items():
        n_feat = mod_ds.n_features
        if mod in bio_x_dict and len(bio_x_dict[mod]) > 0:
            bio_values[i, col : col + n_feat] = bio_x_dict[mod][0]
        col += n_feat

    # age at measurement (years)
    age[i] = cutoff_t / 365.25

# drop tokens with zero or near-zero prevalence
prevalence = token_matrix.sum(axis=0)
min_count = max(10, n * 0.01)
keep = prevalence >= min_count
token_matrix = token_matrix[:, keep]
token_name_list = [n for n, k in zip(token_name_list, keep) if k]
print(f"tokens kept: {token_matrix.shape[1]} / {len(token_ids)}")
# -

# assemble feature matrix
X = np.concatenate(
    [token_matrix, bio_values, age[:, None]],
    axis=1,
)
feature_names = token_name_list + all_bio_features + ["age"]

# drop participants with NaN saliency (target disease already occurred)
valid = ~np.isnan(y)
X = X[valid]
y = y[valid]
bio_values = bio_values[valid]
age = age[valid]
print(f"valid (non-NaN) saliency: {valid.sum()} / {len(valid)}")

# scatter: saliency vs biomarker value
raw_vals = bio_values[:, target_feature_col]
fig, ax = plt.subplots()
sc = ax.scatter(raw_vals, y, s=1, alpha=0.3, c=age, cmap="viridis", rasterized=True)
fig.colorbar(sc, ax=ax, label="age (years)")
if args.bin_lo is not None or args.bin_hi is not None:
    lo = args.bin_lo if args.bin_lo is not None else -np.inf
    hi = args.bin_hi if args.bin_hi is not None else np.inf
    if args.bin_lo is not None:
        ax.axvline(lo, color="k", ls="--", lw=0.8)
    if args.bin_hi is not None:
        ax.axvline(hi, color="k", ls="--", lw=0.8)
ax.set_xlabel(args.feature)
ax.set_ylabel("saliency")
ax.set_title(args.target)
fig.tight_layout()
plt.show()

y.std()

# +
# filter to biomarker bin if specified
if args.bin_lo is not None or args.bin_hi is not None:
    raw_vals = bio_values[:, target_feature_col]
    lo = args.bin_lo if args.bin_lo is not None else -np.inf
    hi = args.bin_hi if args.bin_hi is not None else np.inf
    bin_mask = (raw_vals >= lo) & (raw_vals < hi)
    X = X[bin_mask]
    y = y[bin_mask]
    print(f"bin [{lo}, {hi}): {bin_mask.sum()} / {len(bin_mask)} participants")

print(f"X shape: {X.shape}, y shape: {y.shape}")

# +
# standardize
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1.0
X = (X - X_mean) / X_std
X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
feature_names.append("intercept")

# ridge regression on GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
X_t = torch.tensor(X, dtype=torch.float32, device=device)
y_t = torch.tensor(y, dtype=torch.float32, device=device)

XtX = X_t.T @ X_t
Xty = X_t.T @ y_t
I = torch.eye(XtX.shape[0], device=device)
I[-1, -1] = 0  # don't regularize intercept
w = torch.linalg.solve(XtX + args.alpha * I, Xty)
w = w.cpu().numpy()

# R² score
y_pred = X @ w
ss_res = ((y - y_pred) ** 2).sum()
ss_tot = ((y - y.mean()) ** 2).sum()
r2 = 1 - ss_res / ss_tot
print(f"R² = {r2:.4f}")

# +
# top-k positive and negative coefficients
k = args.top_k
sorted_idx = np.argsort(w)

print(f"\ntop {k} positive coefficients (increases saliency of {args.feature}):")
for idx in sorted_idx[::-1][:k]:
    print(f"  {w[idx]:+.4f}  {feature_names[idx]}")

print(f"\ntop {k} negative coefficients (decreases saliency of {args.feature}):")
for idx in sorted_idx[:k]:
    print(f"  {w[idx]:+.4f}  {feature_names[idx]}")
# -


#
