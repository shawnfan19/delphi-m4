"""Compare biomarker-value distributions between disease cases and controls.

Two subplots per (disease, feature) pair:
  Left  — all cases (have disease token) vs all controls (no disease token).
  Right — cases restricted to those whose biomarker measurement preceded
          the first disease token timestamp (probes treatment-bias inversion:
          measured-pre-disease cases should look higher / disease-shifted,
          whereas the full-case distribution may look inverted because many
          measurements happen post-diagnosis under treatment).

Controls are the same set in both plots — only the case set narrows on
the right. Ns are annotated in the legend.

Dataset-agnostic via delphi.data.auto.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from delphi.data.auto import detect_dataset, multimodal_reader_cls
from delphi.experiment import CliConfig

mpl.rcParams["figure.dpi"] = 300


@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    disease: str  # token name, e.g. e78_(disorders_of_lipoprotein_metabolism_...)
    feature: str  # shared feature name, e.g. cholesterol, glycated_haemoglobin
    write: str = "results/biomarker_disease"


args = TaskConfig.from_cli()
args.print()

mm_cls = multimodal_reader_cls()
dataset_name = os.environ.get("DELPHI_DATASET") or detect_dataset()
OUT_DIR = Path(args.write) / dataset_name
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Resolve which biomarker directory contains the requested feature.
biomarker = None
for name in mm_cls.biomarker_cls.catalog():
    bio = mm_cls.biomarker_cls(name)
    if args.feature in bio.features:
        biomarker = name
        break
if biomarker is None:
    raise ValueError(
        f"feature {args.feature!r} not found in any biomarker "
        f"({mm_cls.biomarker_cls.catalog()})"
    )
print(f"feature {args.feature!r} resolves to biomarker {biomarker!r}")

reader = mm_cls(biomarkers=[biomarker])
pack = reader.biomarkers[biomarker]

if args.disease not in reader.tokenizer:
    raise ValueError(f"disease token {args.disease!r} not in {dataset_name} tokenizer")
disease_token_id = reader.tokenizer[args.disease]
feat_idx = pack.feat2idx[args.feature]

# Intersect biomarker pids with the base reader's pids (drop orphans).
base_pids = mm_cls.participants("all")
bio_pids = np.array(list(pack.pid2idx.keys()))
bio_pids = bio_pids[np.isin(bio_pids, base_pids)]
print(f"{len(bio_pids)} participants with a {biomarker} measurement")

# Biomarker values + first-occurrence times: aligned to bio_pids.
print(f"loading {args.feature} values + times for {len(bio_pids)} participants...")
values = pack.to_array(bio_pids)[:, feat_idx].astype(np.float64)
bio_times = mm_cls.biomarker_cls.first_occurrence_times(biomarker, bio_pids).astype(
    np.float64
)

# Disease times: one batched call — column of the disease token in the
# (N, vocab_size) first-occurrence matrix. NaN = absent.
print(f"computing event_times over {len(bio_pids)} participants...")
event_t = reader.event_times(bio_pids)
disease_times = event_t[:, disease_token_id].astype(np.float64)
del event_t  # free the (N, vocab_size) matrix early
has_disease = ~np.isnan(disease_times)

case = has_disease
control = ~has_disease
case_pre = case & (bio_times < disease_times)

n_case, n_control, n_case_pre = int(case.sum()), int(control.sum()), int(case_pre.sum())
print(f"control: {n_control}, case: {n_case}, case pre-disease: {n_case_pre}")

# Render: two subplots sharing x and y for direct visual comparison.
bins = 60
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

axes[0].hist(
    values[control],
    bins=bins,
    alpha=0.45,
    label=f"control (n={n_control})",
    density=True,
)
axes[0].hist(
    values[case],
    bins=bins,
    alpha=0.45,
    label=f"case (n={n_case})",
    density=True,
)
axes[0].set_xlabel(args.feature)
axes[0].set_ylabel("density")
axes[0].set_title("all cases vs controls")
axes[0].legend()

axes[1].hist(
    values[control],
    bins=bins,
    alpha=0.45,
    label=f"control (n={n_control})",
    density=True,
)
axes[1].hist(
    values[case_pre],
    bins=bins,
    alpha=0.45,
    label=f"case, bio measured before disease (n={n_case_pre})",
    density=True,
)
axes[1].set_xlabel(args.feature)
axes[1].set_title("biomarker measured BEFORE disease onset")
axes[1].legend()

fig.suptitle(f"{dataset_name}: {args.disease} × {args.feature}")
fig.tight_layout()

disease_short = args.disease.split("_")[0]
out_path = OUT_DIR / f"{disease_short}_{args.feature}.png"
with out_path.open("wb") as f:
    plt.savefig(f, format="png", bbox_inches="tight")
print(f"Saved {out_path}")
plt.close()
