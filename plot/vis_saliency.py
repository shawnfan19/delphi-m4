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

# %%
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from cloudpathlib import AnyPath
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

from delphi.data.ukb import Biomarker
from delphi.env import DELPHI_CKPT_DIR, DELPHI_DATA_DIR
from delphi.experiment import load_ckpt


# %%
def load_saliency(dirpath: str | os.PathLike):
    dirpath = AnyPath(dirpath)
    jacobians = np.load(dirpath / "jacobians.npy", mmap_mode="r")
    logits = np.load(dirpath / "logits.npy", mmap_mode="r")
    pids = np.load(dirpath / "pids.npy")
    return jacobians, logits, pids


def load_ckpt_meta(ckpt_path):
    model, ckpt_dict = load_ckpt(ckpt_path)
    tokenizer = ckpt_dict["tokenizer"]
    targets = model.targets
    targets = targets[targets != 1].cpu().numpy()
    data_args = ckpt_dict["data_args"]
    return tokenizer, targets, data_args


def load_biomarker(modality: str, data_args: dict) -> Biomarker:
    return Biomarker(
        path=AnyPath(DELPHI_DATA_DIR)
        / "ukb_real_data"
        / "biomarkers"
        / modality.lower(),
        stats_subjects=np.fromfile(
            AnyPath(DELPHI_DATA_DIR) / "ukb_real_data" / data_args["subject_list"],
            dtype=np.uint32,
        ),
        z_score=data_args.get("z_score_biomarkers", False),
    )


# %%
ckpt_path = AnyPath(DELPHI_CKPT_DIR) / "interpret/blood/ckpt.pt"
saliency_dir = ckpt_path.parent / "saliency-RENAL"
modality_name = "RENAL"

sal_matrix, logits, pids = load_saliency(saliency_dir)
tokenizer, targets, data_args = load_ckpt_meta(ckpt_path)
detokenizer = {v: k for k, v in tokenizer.items()}

bio = load_biomarker(modality_name, data_args)
features = bio.features
biomarker_val, _ = bio.to_array(pids, first_time_only=True)
bio_timesteps = bio.first_occurrence_times(pids)

# %%
sal_matrix.shape, logits.shape, biomarker_val.shape


# %%
def plot_saliency_vs_value(
    sal_matrix: np.ndarray,
    biomarker_values: np.ndarray,
    feature_names: list,
    target_idx: int,
    target_name=None,
    age=None,
    n_cols=3,
    figsize_per_subplot=(4, 3.5),
    lowess_frac=None,
):
    import textwrap

    age_years = age / 365.25 if age is not None else None

    k = len(feature_names)
    n_rows = int(np.ceil(k / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_subplot[0] * n_cols, figsize_per_subplot[1] * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    display_name = target_name if target_name else f"target {target_idx}"

    for idx, feat in enumerate(feature_names):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        x = biomarker_values[:, idx]
        y = sal_matrix[:, idx, target_idx].astype(np.float32)

        scatter_kw = dict(alpha=0.2, s=10, rasterized=True)
        if age_years is not None:
            scatter_kw.update(c=age_years, cmap="viridis")
        else:
            scatter_kw.update(c="steelblue")

        sc = ax.scatter(x, y, **scatter_kw)

        if lowess_frac is not None:
            sort_idx = np.argsort(x)
            smoothed = sm_lowess(y[sort_idx], x[sort_idx], frac=lowess_frac)
            ax.plot(smoothed[:, 0], smoothed[:, 1], "r-", linewidth=2)

        ax.set_xlabel(feat)
        ax.set_ylabel("hazard ratio")
        ax.set_title(textwrap.fill(display_name, width=25), wrap=True, pad=15)

    if age_years is not None:
        fig.colorbar(sc, ax=axes, label="age (years)")

    plt.show()


# %%

# %%
# Example usage: pick a target disease by name or index
# target_name = "i21_(acute_myocardial_infarction)"
# target_name = "k74_(fibrosis_and_cirrhosis_of_liver)"
# target_name = "m10_(gout)"
# target_name = "i20_(angina_pectoris)"
# target_name =  'o47_(false_labour)'
# target_name = "e11_(non-insulin-dependent_diabetes_mellitus)"
# target_name = "n18_(chronic_renal_failure)"
# target_name = "k70_(alcoholic_liver_disease)"
# target_name = "d50_(iron_deficiency_anaemia)"
# target_name = "e78_(disorders_of_lipoprotein_metabolism_and_other_lipidaemias)"
target_name = "n18_(chronic_renal_failure)"
target_idx = targets.tolist().index(tokenizer[target_name])
# print(f"target: {target_name} (idx={target_idx})")

plot_saliency_vs_value(
    sal_matrix=np.exp(sal_matrix),
    biomarker_values=biomarker_val,
    feature_names=features,
    target_name=target_name,
    target_idx=target_idx,
    age=bio_timesteps,
    lowess_frac=0.3,
)

# %%
# sal_matrix * np.expand_dims(np.exp(logits), axis=1)
# sal_matrix * hazards * np.exp(-hazards * horizon) * horizon

# %%
hazards = np.expand_dims(np.exp(logits), axis=1)
horizon = 365.25 * 5

# %%
# X = np.exp(sal_matrix)
mean_saliency = np.exp(np.nanmean(sal_matrix, axis=0))

# %%
import pprint

k = 10
for i in range(sal_matrix.shape[1]):

    saliency_per_feature = mean_saliency[i, :]
    max_target_idx = np.argsort(saliency_per_feature)[::-1][:k]
    max_targets = targets[max_target_idx]
    print(features[i])
    top_diseases = [detokenizer[j] for j in max_targets]
    top_score = [round(float(saliency_per_feature[j]), 2) for j in max_target_idx]
    # print(top_diseases)
    pprint.pp(list(zip(top_diseases, top_score)))


# %% [markdown]
# ## Heatmap: feature saliency across diseases

# %%
import yaml

disease_yaml = "diseases.yaml"  # list of disease names
modalities = [
    "APO",
    "CRP",
    "CYSC",
    "DHT",
    "HBA1C",
    "IGF1",
    "LFT",
    "LIPID",
    "RENAL",
    "SHBG",
    "URATE",
    "VITD",
    "WBC",
]
saliency_dirs = [f"saliency-{modality}-ckpt-ckpt" for modality in modalities]

# %%
with open(disease_yaml) as f:
    disease_list = yaml.safe_load(f)

ckpt_path = AnyPath(DELPHI_CKPT_DIR) / "delphi-m4/blood/ckpt.pt"
tokenizer, targets, data_args = load_ckpt_meta(ckpt_path)

# resolve disease token indices once
disease_indices = [targets.tolist().index(tokenizer[d]) for d in disease_list]

all_features = []
all_mean_saliency = []  # each entry: (n_features, n_diseases)
all_var_saliency = []

for modality_name, sal_dir in zip(modalities, saliency_dirs):
    jacobians, _, _ = load_saliency(ckpt_path.parent / sal_dir)
    bio = load_biomarker(modality_name, data_args)

    # jacobians: (N, n_features, n_targets) — scale from z-score to raw units
    mean_sal = np.nanmean(jacobians, axis=0)  # (n_features, n_targets)
    mean_sal = mean_sal * bio.std[:, np.newaxis]
    var_sal = np.nanvar(jacobians, axis=0)
    var_sal = var_sal * (bio.std[:, np.newaxis] ** 2)

    mean_sal = mean_sal[:, disease_indices]  # (n_features, n_diseases)
    var_sal = var_sal[:, disease_indices]

    all_features.extend(bio.features)
    all_mean_saliency.append(mean_sal)
    all_var_saliency.append(var_sal)

heatmap_data = np.concatenate(all_mean_saliency, axis=0)  # (total_features, n_diseases)
var_heatmap_data = np.concatenate(
    all_var_saliency, axis=0
)  # (total_features, n_diseases)

# %%

# %%
heatmap = heatmap_data

# %%
heatmap = np.clip(heatmap_data, max=2)

# %%
fig, ax = plt.subplots(
    figsize=(max(6, len(all_features) * 0.8), max(4, len(disease_list) * 0.4))
)
norm = plt.matplotlib.colors.TwoSlopeNorm(vcenter=1.0)
im = ax.imshow(np.exp(heatmap).T, aspect="auto", cmap="RdBu_r", norm=norm)
ax.set_xticks(range(len(all_features)))
ax.set_xticklabels(all_features, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(disease_list)))
ax.set_yticklabels(disease_list, fontsize=8)
ax.set_xlabel("Feature")
ax.set_ylabel("Disease")
fig.colorbar(im, ax=ax, label="Mean saliency (hazard ratio)")
fig.suptitle("Biomarker feature saliency across diseases")
plt.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(
    figsize=(max(6, len(all_features) * 0.8), max(4, len(disease_list) * 0.4))
)
im = ax.imshow(var_heatmap_data.T, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(len(all_features)))
ax.set_xticklabels(all_features, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(disease_list)))
ax.set_yticklabels(disease_list, fontsize=8)
ax.set_xlabel("Feature")
ax.set_ylabel("Disease")
fig.colorbar(im, ax=ax, label="Jacobian variance")
fig.suptitle("Saliency variance across participants (nonlinearity indicator)")
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt

x_lst = list()
y_lst = list()
for i in range(mean_saliency.shape[0]):
    y_lst.append(mean_saliency[i, :])
    x_lst.append(2 * i + np.random.rand(y.size))

plt.scatter(x_lst, y_lst)

# %%

# %%
