# +
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import patsy
import torch
from statsmodels.nonparametric.smoothers_lowess import lowess

from delphi.env import DELPHI_CKPT_DIR
from delphi.experiment import CliConfig


def load_saliency(path):
    """Load the saliency artifact as a dict, supporting both output layouts.

    A ``path`` ending in ``.npz`` is the single compressed file written by
    apps/saliency_biomarker.py (jacobians/logits/pids plus the covariate context:
    token_matrix, bio_values, age, and the axis-label arrays). Anything else is
    the legacy directory of ``.npy`` files, which only ever held jacobians + pids
    (no covariates); its jacobian is a read-only ``mmap_mode="r"`` array, so
    ``.copy()`` a slice before mutating it.
    """
    path = Path(path)
    if str(path).endswith(".npz"):
        with np.load(path) as sal:
            out = {k: sal[k] for k in sal.files}
        # apps/ig_biomarker.py stores the per-participant feature->target attribution
        # under `attributions`; the gradient-saliency artifact uses `jacobians`. They
        # play the same role as the regression response here, so normalize to one key.
        if "jacobians" not in out and "attributions" in out:
            out["jacobians"] = out["attributions"]
        return out
    return {
        "jacobians": np.load(path / "jacobians.npy", mmap_mode="r"),
        "pids": np.load(path / "pids.npy"),
    }


# +
@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    saliency: str = "interpret/blood/saliency-RENAL.npz"  # relative to DELPHI_CKPT_DIR
    feature: str = "creatinine"
    target: str = "n18_(chronic_renal_failure)"
    # per-sample ridge penalty on the unit-scale standardized design;
    # lam ~ (1 - rho) sets the feature correlation it meaningfully shrinks.
    lam: float = 0.1
    top_k: int = 20
    value_trim: float = 0.01
    # response scale: "log" = d(log-intensity)/d value (saliency); "intensity" =
    # d(intensity)/d value = lambda * saliency. Selects y for ALL plots + regression.
    scale: str = "log"
    # if set, write the figures as PNGs here (relative to DELPHI_CKPT_DIR unless
    # absolute) instead of plt.show(); filenames are built from feature/target/scale.
    figdir: None | str = None


args = TaskConfig.from_cli()
args.print()


def emit(fig, kind):
    """Save ``fig`` under args.figdir (named by feature/target/scale) or plt.show()."""
    if not args.figdir:
        plt.show()
        return
    figdir = Path(args.figdir)
    if not figdir.is_absolute():
        figdir = Path(DELPHI_CKPT_DIR) / figdir
    figdir.mkdir(parents=True, exist_ok=True)
    safe = lambda s: re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_")
    path = (
        figdir / f"{safe(args.feature)}__{safe(args.target)}__{args.scale}__{kind}.png"
    )
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"saved figure: {path}")


# +
# load the saliency artifact: jacobians + the covariate context that
# apps/saliency_biomarker.py saved alongside them, so this script needs no model
# or reader — only the path to the .npz (relative to DELPHI_CKPT_DIR).
saliency_path = Path(DELPHI_CKPT_DIR) / args.saliency
sal = load_saliency(saliency_path)
required = [
    "jacobians",
    "pids",
    "age",
    "token_matrix",
    "token_names",
    "bio_values",
    "bio_names",
    "feature_names",
    "target_names",
]
missing = [k for k in required if k not in sal]
assert not missing, (
    f"{saliency_path} is missing {missing}; re-run apps/saliency_biomarker.py to "
    "emit the covariate context (this script needs a post-refactor .npz)"
)

# resolve the jacobian axes + value column by name (no model/reader needed).
# feature_names/bio_names are qualified "modality:feature"; args.feature is the bare
# name, matched within the single-modality feature axis, then used to find the col.
feat_axis = sal["feature_names"].tolist()  # jacobian feature axis ("modality:feature")
target_names = sal["target_names"].tolist()  # jacobian target axis
bio_names = sal["bio_names"].tolist()  # bio_values columns ("modality:feature")
feat_suffixes = [f.split(":", 1)[1] for f in feat_axis]
assert args.feature in feat_suffixes, f"{args.feature!r} not in {feat_suffixes}"
assert args.target in target_names, f"{args.target!r} not among the saved targets"
feature_idx = feat_suffixes.index(args.feature)
target_idx = target_names.index(args.target)
target_feature_col = bio_names.index(feat_axis[feature_idx])
assert args.scale in ("log", "intensity"), f"unknown scale {args.scale!r}"
print(f"explaining {args.feature} → {args.target} ({args.scale} scale)")

# y: response on the chosen scale — "log" = saved saliency d(log-intensity)/d value;
# "intensity" = lambda * saliency (lambda = exp(target logit)). All plots + the
# regression below use this y.
saliency = sal["jacobians"][:, feature_idx, target_idx].copy()
logit = sal["logits"][:, target_idx].copy()  # target log-intensity (= log lambda)
if args.scale == "intensity":
    y = np.exp(logit) * saliency
else:
    y = saliency
response_label = "saliency" if args.scale == "log" else "absolute change in intensity"
yscale = "linear" if args.scale == "log" else "log"

# +
# rebuild the design matrix from the saved covariate context (disease-history
# presence + biomarker values + age, all at the cutoff). token_matrix is saved
# un-pruned; the prevalence filter below is an analysis choice kept in this script.
sal_pids = sal["pids"]
n = len(sal_pids)
token_matrix = sal["token_matrix"].astype(np.float32)
token_names = sal["token_names"].tolist()
bio_values = sal["bio_values"]
age = sal["age"]

# drop tokens with zero or near-zero prevalence
prevalence = token_matrix.sum(axis=0)
min_count = max(10, n * 0.01)
keep = prevalence >= min_count
token_matrix = token_matrix[:, keep]
token_names = [t for t, k in zip(token_names, keep) if k]
print(f"tokens kept: {token_matrix.shape[1]} / {len(keep)}")

# Group ordinal lifestyle tokens (bmi/alcohol/smoking) by prefix for the stratified
# panels. mid is the reference level (dropped upstream by the producer). feature_to_group
# maps each surviving level -> its prefix; group_ref records the reference.
levels = ("low", "mid", "high")
onehot = {}
for i, t in enumerate(token_names):
    pre, _, lvl = t.rpartition("_")
    if pre and lvl in levels:
        onehot.setdefault(pre, {})[lvl] = i
onehot = {p: lv for p, lv in onehot.items() if len(lv) >= 2}

feature_to_group, group_ref, ref_cols = {}, {}, []
for pre, lv in onehot.items():
    ref = "mid"  # reference level, dropped upstream by the producer
    group_ref[pre] = ref
    if ref in lv:  # older artifacts may still carry the mid column; drop it here
        ref_cols.append(lv[ref])
    feature_to_group.update({token_names[c]: pre for x, c in lv.items() if x != ref})
if ref_cols:
    drop = np.ones(token_matrix.shape[1], dtype=bool)
    drop[ref_cols] = False
    token_matrix = token_matrix[:, drop]
    token_names = [t for t, k in zip(token_names, drop) if k]
    print(
        f"one-hot groups {sorted(onehot)} -> dropped refs {group_ref}, "
        f"{token_matrix.shape[1]} token cols kept"
    )
# -

# assemble feature matrix
X = np.concatenate([token_matrix, bio_values, age[:, None]], axis=1)
feature_names = token_names + bio_names + ["age"]

# drop participants with NaN saliency (target disease already occurred)
valid = ~np.isnan(y)
X = X[valid]
y = y[valid]
logit = logit[valid]
bio_values = bio_values[valid]
age = age[valid]
print(f"valid (non-NaN) saliency: {valid.sum()} / {len(valid)}")

if args.value_trim > 0:
    feat_val = bio_values[:, target_feature_col]
    lo, hi = np.quantile(feat_val, [args.value_trim, 1 - args.value_trim])
    keep = (feat_val >= lo) & (feat_val <= hi)
    X, y, logit, bio_values, age = (
        X[keep],
        y[keep],
        logit[keep],
        bio_values[keep],
        age[keep],
    )
    print(f"value-trim to [{lo:.3g}, {hi:.3g}]: kept {keep.sum()} / {keep.size}")


def fit_value_spline(values, response, df=5, n_grid=200):
    """Fit ``response ~ cr(value, df)``; return ``(grid, fitted curve)``."""
    dm = patsy.dmatrix(f"cr(v, df={df})", {"v": values})
    beta, *_ = np.linalg.lstsq(np.asarray(dm), response, rcond=None)
    grid = np.linspace(values.min(), values.max(), n_grid)
    design = patsy.build_design_matrices([dm.design_info], {"v": grid})[0]
    return grid, np.asarray(design) @ beta


# overview plot: the response vs the explained feature's value, with the fitted
# value-only spline overlaid.
raw_vals = bio_values[:, target_feature_col]
grid, grid_curve = fit_value_spline(raw_vals, y)

fig, ax = plt.subplots()
sc = ax.scatter(raw_vals, y, c=logit, s=1, alpha=0.1, cmap="viridis", rasterized=True)
fig.colorbar(sc, ax=ax, label="target logit (log-intensity)")
ax.plot(grid, grid_curve, color="crimson", lw=2, label="fitted value spline")
ax.legend()
ax.set_xlabel(args.feature)
ax.set_ylabel(response_label)
ax.set_yscale(yscale)
ax.set_title(f"{args.feature} → {args.target}")
fig.tight_layout()
emit(fig, "overview")

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
# keep the pre-standardized design so the panels can colour by raw feature values
X_raw = X
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

print(
    f"\ntop {k} positive coefficients (increases {response_label} of {args.feature}):"
)
for idx in sorted_idx[::-1][:k]:
    print(f"  {w[idx]:+.4f}  {feature_names[idx]}")

print(
    f"\ntop {k} negative coefficients (decreases {response_label} of {args.feature}):"
)
for idx in sorted_idx[:k]:
    print(f"  {w[idx]:+.4f}  {feature_names[idx]}")
# -

# +
# Per top context feature (by |coef|, excluding the value-spline basis + intercept):
# split participants into strata by that feature and overlay a per-stratum LOWESS
# curve of saliency vs the explained feature's value. Separation between the curves
# at the same x shows the model's saliency depends on context, not just the value.
# rank context features by |coef|; collapse each one-hot group to a single entry so a
# grouped variable (bmi/alcohol/smoking) appears as one panel, not several.
top, seen = [], set()
for i in np.argsort(-np.abs(w)):
    name = feature_names[i]
    if name == "intercept" or name.startswith("value_spline"):
        continue
    grp = feature_to_group.get(name)
    if grp is not None:
        if grp in seen:
            continue
        seen.add(grp)
        top.append(("group", grp, w[i]))
    else:
        top.append(("feature", i, w[i]))
    if len(top) == 5:
        break

# Feature type is implied by the design block: disease-history + lifestyle/sex
# tokens are 0/1 (two strata), biomarker values and age are continuous (tertiles).
token_set = set(token_names)


def strata(name, vals):
    """Strata for a single feature: levels of a binary token, else value tertiles."""
    if name in token_set:
        return [(f"{v:g}", vals == v) for v in np.unique(vals)]
    lo, hi = np.quantile(vals, [1 / 3, 2 / 3])
    return [
        ("low", vals <= lo),
        ("mid", (vals > lo) & (vals <= hi)),
        ("high", vals > hi),
    ]


def group_strata(prefix):
    """3-level strata for a one-hot group: each surviving level plus the dropped
    reference (= neither level set, i.e. the reference band + the few unrecorded)."""
    ref = group_ref[prefix]
    cols = {
        lv: X_raw[:, feature_names.index(f"{prefix}_{lv}")]
        for lv in levels
        if lv != ref and f"{prefix}_{lv}" in feature_names
    }
    ref_mask = np.logical_and.reduce([c == 0 for c in cols.values()])
    out = []
    for lv in levels:
        if lv in cols:
            out.append((lv, cols[lv] == 1))
        elif lv == ref:
            out.append((f"{lv} (ref)", ref_mask))
    return out


fig, axes = plt.subplots(1, len(top), figsize=(4.6 * len(top), 4.2), sharey=True)
axes = np.atleast_1d(axes)
for ax, (kind, key, coef) in zip(axes, top):
    if kind == "group":
        groups = group_strata(key)
        label_for = key
    else:
        groups = strata(feature_names[key], X_raw[:, key])
        label_for = feature_names[key]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(groups)))
    for (label, mask), color in zip(groups, colors):
        if mask.sum() < 50:
            continue
        xm, ym = raw_vals[mask], y[mask]
        ax.scatter(xm, ym, s=1, alpha=0.01, color=color, rasterized=True)
        # smooth LOWESS evaluated on a grid over the stratum's supported range
        # (2-98th pct), not at the discrete/sparse data values, so the curve stays
        # smooth and isn't drawn into the noisy tail; it=1 adds outlier robustness.
        g = np.linspace(*np.quantile(xm, [0.02, 0.98]), 80)
        sm = lowess(ym, xm, xvals=g, frac=0.4, it=1)
        ax.plot(g, sm, color=color, lw=2.5, label=f"{label} (n={mask.sum()})")
    ax.set_xlabel(args.feature)
    ax.set_yscale(yscale)
    ax.set_title(f"{label_for}  (w={coef:+.3f})")
    ax.legend(title=label_for, fontsize=7)
axes[0].set_ylabel(response_label)
fig.suptitle(
    f"{response_label} of {args.feature} → {args.target}, stratified by context"
)
fig.tight_layout()
emit(fig, "strata")
# -


#
