# +
import pprint
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import patsy
import torch

from delphi.data.ukb import Biomarker, MultimodalUKBReader
from delphi.env import DELPHI_CKPT_DIR
from delphi.experiment import CliConfig, load_ckpt


# +
@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    ckpt: str = "interpret/blood/ckpt.pt"
    saliency_dir: str = "saliency-RENAL"
    modality: str = "renal"
    feature: str = "creatinine"
    target: str = "n18_(chronic_renal_failure)"
    # per-sample ridge penalty on the unit-scale standardized design;
    # lam ~ (1 - rho) sets the feature correlation it meaningfully shrinks.
    lam: float = 0.1
    top_k: int = 20


args = TaskConfig.from_cli()

args.modality = "renal"
args.saliency_dir = "saliency-RENAL"
args.feature = "creatinine"
args.target = "n18_(chronic_renal_failure)"

args.modality = "lft"
args.saliency_dir = "saliency-LFT"
args.feature = "gamma_glutamyltransferase"
args.target = "k70_(alcoholic_liver_disease)"

# args.modality = "wbc"
# args.saliency_dir = "saliency-WBC"
# args.feature = "haemoglobin_concentration"
# args.target = "c19_malignant_neoplasm_of_rectosigmoid_junction"
#
args.modality = "lipid"
args.saliency_dir = "saliency-LIPID"
args.feature = "ldl_direct"
args.target = "i21_(acute_myocardial_infarction)"

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

# sal_pids already aligns row-for-row with the saved jacobians, so no further
# participant filtering is needed.
mod_name = args.modality.lower()
reader_args = ckpt_dict["reader_args"]
reader = MultimodalUKBReader(
    biomarkers=reader_args["biomarkers"],
    expansion_packs=reader_args["expansion_packs"],
)
assert mod_name in reader.biomarkers, f"{mod_name!r} not in {sorted(reader.biomarkers)}"
bio = reader.biomarkers[mod_name]
feature_idx = bio.feat2idx[args.feature]
print(f"explaining saliency of {args.feature} → {args.target}")

# cutoff = target modality's (first/only) measurement time, per participant
cutoff = Biomarker.first_occurrence_times(mod_name, sal_pids)

# y: saliency value per participant
y = jacobians[:, feature_idx, target_idx].copy()

# +
# build covariate matrix: past-disease history + biomarker values + age,
# all evaluated at the cutoff (the target modality's measurement time).

# past-disease history: tokens whose first occurrence is at/before the cutoff.
# use the base tokenizer so columns line up with reader.event_times (base-indexed).
base_tok = reader.base_tokenizer
exclude_ids = {base_tok.get("padding", 0), base_tok.get("no_event", 1)}
token_names = sorted(
    [(name, tok) for name, tok in base_tok.items() if tok not in exclude_ids],
    key=lambda x: x[1],
)
token_name_list = [name for name, _ in token_names]
token_ids = np.array([tok for _, tok in token_names])

present = reader.event_times(sal_pids) <= cutoff[:, None]  # (N, base_vocab)
token_matrix = present[:, token_ids].astype(np.float32)

# biomarker values: each modality's first-occurrence vector, zeroed where the
# measurement is absent or falls after the cutoff (temporal consistency).
all_bio_features = []
for name, bm in reader.biomarkers.items():
    all_bio_features.extend([f"{name}:{f}" for f in bm.features])
target_feature_col = all_bio_features.index(f"{mod_name}:{args.feature}")

n = len(sal_pids)
bio_values = np.zeros((n, len(all_bio_features)), dtype=np.float32)
col = 0
for name, bm in reader.biomarkers.items():
    n_feat = bm.n_features
    foc = Biomarker.first_occurrence_times(name, sal_pids)  # NaN if absent
    vals = bm.to_array(sal_pids)  # (N, n_feat), NaN if absent
    vals[~(foc <= cutoff)] = 0.0  # absent (NaN) or measured after cutoff -> 0
    bio_values[:, col : col + n_feat] = vals
    col += n_feat

age = (cutoff / 365.25).astype(np.float32)

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

# scatter: saliency vs biomarker value, with the fitted value-only spline overlaid
raw_vals = bio_values[:, target_feature_col]
fig, ax = plt.subplots()
sc = ax.scatter(raw_vals, y, s=1, alpha=0.3, c=age, cmap="viridis", rasterized=True)
fig.colorbar(sc, ax=ax, label="age (years)")

# fitted value-only model: saliency ~ spline(value); overlay the smooth curve
value_dm = patsy.dmatrix("cr(v, df=5)", {"v": raw_vals})
value_beta, *_ = np.linalg.lstsq(np.asarray(value_dm), y, rcond=None)
grid = np.linspace(raw_vals.min(), raw_vals.max(), 200)
grid_dm = patsy.build_design_matrices([value_dm.design_info], {"v": grid})[0]
ax.plot(
    grid,
    np.asarray(grid_dm) @ value_beta,
    color="crimson",
    lw=2,
    label="fitted value spline",
)
ax.legend()

ax.set_xlabel(args.feature)
ax.set_ylabel("saliency")
ax.set_title(args.target)
fig.tight_layout()
plt.show()

y.std()

# +
# regress out the biomarker value with an unpenalized natural cubic spline:
# replace its single linear column with a flexible basis, so the context
# coefficients become "effect on saliency holding the biomarker value fixed"
# (Frisch-Waugh-Lovell). This supersedes binning and uses all the data.
target_vals = bio_values[:, target_feature_col].astype(float)  # aligned to X rows
S = np.asarray(patsy.dmatrix("cr(x, df=5) - 1", {"x": target_vals}), dtype=np.float32)

value_col = token_matrix.shape[1] + target_feature_col  # value column within X
X = np.delete(X, value_col, axis=1)
feature_names.pop(value_col)
spline_cols = np.arange(X.shape[1], X.shape[1] + S.shape[1])
X = np.concatenate([X, S], axis=1)
feature_names += [f"value_spline{i}" for i in range(S.shape[1])]

print(f"X shape: {X.shape}, y shape: {y.shape}")

# +
# standardize
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1.0
X = (X - X_mean) / X_std
X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
feature_names.append("intercept")

# ridge regression on GPU.
# per-sample penalty: minimize (1/N)||y - Xw||^2 + lam*||w||^2, whose normal
# equations are (X^T X + N*lam*I) w = X^T y. lam therefore lives on the unit
# scale of the standardized design (~the correlation matrix), independent of N.
# the intercept and the value-spline are left unpenalized so the
# spline fully absorbs the value effect instead of leaking it into context.
device = "cuda" if torch.cuda.is_available() else "cpu"
X_t = torch.tensor(X, dtype=torch.float32, device=device)
y_t = torch.tensor(y, dtype=torch.float32, device=device)

n_obs = X_t.shape[0]
XtX = X_t.T @ X_t
Xty = X_t.T @ y_t

penalty = torch.ones(X_t.shape[1], device=device)
penalty[-1] = 0.0  # intercept
penalty[torch.as_tensor(spline_cols, device=device)] = 0.0  # value control
w = torch.linalg.solve(XtX + n_obs * args.lam * torch.diag(penalty), Xty)
w = w.cpu().numpy()

# R²: full model, and a value-only baseline (spline + intercept, unpenalized OLS)
# so we can report how much context explains beyond the biomarker value itself.
ss_tot = ((y - y.mean()) ** 2).sum()
y_pred = X @ w
r2 = 1 - ((y - y_pred) ** 2).sum() / ss_tot

value_cols = torch.as_tensor(
    np.append(spline_cols, X_t.shape[1] - 1), device=device  # spline + intercept
)
X0 = X_t[:, value_cols]
w0 = torch.linalg.solve(X0.T @ X0, X0.T @ y_t)
y_pred0 = (X0 @ w0).cpu().numpy()
r2_0 = 1 - ((y - y_pred0) ** 2).sum() / ss_tot

print(f"R² (value spline + context) = {r2:.4f}")
print(f"R² (value spline only)      = {r2_0:.4f}")
print(f"context beyond value        = {r2 - r2_0:.4f}")

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
