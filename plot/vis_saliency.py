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

# %% [markdown]
# # Biomarker -> disease association heatmap
#
# One cell per (biomarker feature, disease) = the **signed specificity** of the
# model's saliency:
#
#     cell[f, d] = mean_p saliency[f, d]  /  mean_t |mean_p saliency[f, t]|
#
# The numerator is the mean (over participants) saliency d(log-intensity)/d(value),
# already per-SD because the stored jacobian is taken w.r.t. the z-scored input —
# so it is comparable across biomarkers. The denominator normalises by that
# biomarker's average |saliency| across a baseline set of diseases (all model
# targets, or just the listed ones), so a marker that fires for *everything*
# (e.g. a broad morbidity signal) is divided down and a *specific* association
# stands out. The sign carries direction; a noisy marker whose saliency averages
# to ~0 (large magnitude, no consistent direction) is muted automatically.

# %%
from dataclasses import dataclass
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from cloudpathlib import AnyPath

from delphi.env import DELPHI_CKPT_DIR, DELPHI_RESULTS_DIR
from delphi.experiment import CliConfig, flexi_list


# %%
@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    # directory (relative to DELPHI_CKPT_DIR) holding the saliency-<panel>.npz files
    ckpt: str = "cross-cohort/blood+urine"
    # diseases (heatmap columns): a single token, an inline list, or a .yaml path
    # (normalized via flexi_list). Order is preserved.
    diseases: Any = f"{DELPHI_RESULTS_DIR}/blood/improved.yaml"
    # biomarker panels (npz stems, e.g. ["renal_panel", "cbc"]): a single name, an
    # inline list, or a .yaml path. None = every saliency-*.npz in the ckpt dir.
    # Rows are all features within the selected panels, in panel order.
    biomarkers: Any = None
    # specificity denominator: "all" = average |saliency| over every model target;
    # "listed" = average over only the diseases above.
    specificity_baseline: str = "all"
    # if set, also save the figure to results/<write>/saliency_heatmap.png
    write: None | str = None

    def __post_init__(self):
        self.diseases = flexi_list(self.diseases)
        if self.biomarkers is not None:
            self.biomarkers = flexi_list(self.biomarkers)


args = TaskConfig.from_cli()
args.print()
assert args.specificity_baseline in ("all", "listed")

# %%
ckpt_dir = AnyPath(DELPHI_CKPT_DIR) / args.ckpt

# resolve which panels (npz files) to load
if args.biomarkers is None:
    panels = sorted(
        p.name[len("saliency-") : -len(".npz")] for p in ckpt_dir.glob("saliency-*.npz")
    )
else:
    panels = list(args.biomarkers)
assert panels, f"no saliency-*.npz panels found under {ckpt_dir}"
print(f"panels: {panels}")
print(f"diseases: {len(args.diseases)}")


# %%
def load_saliency(path):
    with np.load(AnyPath(path), allow_pickle=True) as z:
        return z["jacobians"], z["feature_names"], z["target_names"]


# %%
# Accumulate the signed-specificity matrix one panel at a time (panel jacobians are
# multi-GB, so we never hold more than one in memory).
row_labels = []  # "panel:feature"
rows = []  # each: signed-specificity vector over the listed diseases

for panel in panels:
    path = ckpt_dir / f"saliency-{panel}.npz"
    jac, feat_names, target_names = load_saliency(path)  # (N, n_feat, n_targets)
    target_names = target_names.tolist()
    missing = [d for d in args.diseases if d not in target_names]
    assert not missing, f"diseases not in {path.name} target_names: {missing}"
    disease_idx = [target_names.index(d) for d in args.diseases]

    m_signed = np.nanmean(jac, axis=0)  # (n_feat, n_targets)
    m_abs = np.nanmean(np.abs(jac), axis=0)  # (n_feat, n_targets)
    if args.specificity_baseline == "all":
        baseline = np.nanmean(m_abs, axis=1)  # (n_feat,) over all targets
    else:
        baseline = np.nanmean(m_abs[:, disease_idx], axis=1)
    baseline = np.where(baseline > 0, baseline, np.nan)

    signed_spec = m_signed[:, disease_idx] / baseline[:, None]  # (n_feat, n_diseases)
    for fi, fname in enumerate(feat_names.tolist()):
        row_labels.append(fname)
        rows.append(signed_spec[fi])
    del jac
    print(f"  {path.name}: {len(feat_names)} features")

H = np.vstack(rows)  # (n_total_features, n_diseases)
print(f"heatmap: {H.shape[0]} features x {H.shape[1]} diseases")

# %%
# diverging, 0-centered; robust symmetric limit so a single extreme cell doesn't
# wash out the contrast.
vmax = float(np.nanpercentile(np.abs(H), 98)) or float(np.nanmax(np.abs(H)))
norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

fig, ax = plt.subplots(
    figsize=(max(6, 0.55 * H.shape[1] + 4), max(4, 0.30 * H.shape[0] + 1))
)
im = ax.imshow(H, aspect="auto", cmap="RdBu_r", norm=norm)
ax.set_xticks(range(H.shape[1]))
ax.set_xticklabels(args.diseases, rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(H.shape[0]))
ax.set_yticklabels(row_labels, fontsize=7)
ax.set_xlabel("disease")
ax.set_ylabel("biomarker feature")
fig.colorbar(
    im, ax=ax, label="signed specificity (mean saliency / cross-disease avg |saliency|)"
)
ax.set_title(
    f"Biomarker->disease association ({args.ckpt}, baseline={args.specificity_baseline})"
)
fig.tight_layout()

if args.write is not None:
    out_dir = AnyPath(DELPHI_RESULTS_DIR) / args.write
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "saliency_heatmap.png"
    with out_path.open("wb") as f:
        fig.savefig(f, format="png", bbox_inches="tight", dpi=200)
    print(f"saved {out_path}")
plt.show()
