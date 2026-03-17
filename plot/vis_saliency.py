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

import argparse
import gzip

# %%
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
from tqdm import tqdm

from delphi.env import DELPHI_CKPT_DIR
from delphi.multimodal import Modality


# %%
def load_saliency(filepath: str | os.PathLike):
    with gzip.open(filepath, "rb") as f:
        data = pickle.load(f)
    targets = data.pop("targets")
    tokenizer = data.pop("tokenizer")
    modality = data.pop("biomarker")
    features = data.pop("biomarker_features")
    return data, targets, tokenizer, modality, features[Modality[modality]]


def get_saliency_matrix(n_features: int, data: dict):

    sal_list = []
    for pid in data.keys():
        jac = data[pid]["jacobian"]  # (n_meas * n_features, n_targets)
        jac = jac.reshape(
            -1, n_features, jac.shape[-1]
        )  # (n_meas, n_features, n_targets)
        sal_list.append(jac)  # (n_features,)
    sal_matrix = np.concatenate(sal_list)  # (n_pids, n_features)

    return sal_matrix  # (n_meas, n_features, n_targets)


def get_mean_saliency(n_features: int, data: dict, std: np.ndarray = None):
    n_targets = next(iter(data.values()))["jacobian"].shape[-1]
    total = 0
    mean_sal = np.zeros((n_features, n_targets), dtype=np.float64)
    for pid in tqdm(data, total=len(data), leave=False):
        jac = data[pid]["jacobian"].reshape(-1, n_features, n_targets)
        if std is not None:
            jac = jac * std[np.newaxis, :, np.newaxis]
        # jac = np.exp(jac)
        mean_sal += jac.sum(axis=0)
        total += jac.shape[0]
    mean_sal /= total
    return mean_sal  # (n_features, n_targets)


def get_var_saliency(n_features: int, data: dict, std: np.ndarray = None):
    """Welford's online variance of per-measurement Jacobians."""
    n_targets = next(iter(data.values()))["jacobian"].shape[-1]
    total = 0
    mean = np.zeros((n_features, n_targets), dtype=np.float64)
    m2 = np.zeros((n_features, n_targets), dtype=np.float64)
    for pid in tqdm(data, total=len(data), leave=False):
        jac = data[pid]["jacobian"].reshape(-1, n_features, n_targets)
        if std is not None:
            jac = jac * std[np.newaxis, :, np.newaxis]
        for k in range(jac.shape[0]):
            total += 1
            delta = jac[k] - mean
            mean += delta / total
            delta2 = jac[k] - mean
            m2 += delta * delta2
    return m2 / total  # (n_features, n_targets)


def get_biomarker_values(data: dict, modality: str):

    bio_x_list = list()
    for pid in data.keys():
        bio_x = data[pid]["bio_x"]
        bio_x_list.extend(bio_x[Modality[modality]])

    return np.array(bio_x_list)


def get_biomarker_timesteps(data: dict, modality: str):

    modval = Modality[modality].value
    bio_t_list = list()
    for pid in data.keys():
        bio_m = data[pid]["bio_m"]
        is_mod = bio_m == modval
        bio_t_list.append(np.unique(data[pid]["bio_t"][is_mod]))

    return np.concatenate(bio_t_list)


def get_timesteps_and_logits(data: dict, modality: str):

    modval = Modality[modality].value
    timesteps_list = list()
    logits = list()
    for pid in data.keys():
        n = (data[pid]["bio_m"] == modval).sum()
        t = max(data[pid]["t"].max(), data[pid]["bio_t"].max())

        timesteps_list.append(np.full(n, t))
        # logits.append(np.repeat(data[pid]["logits"][None, :], n, axis=0))
        logits.append(np.repeat(data[pid]["logits"], n, axis=0))

    return np.concatenate(timesteps_list), np.concatenate(logits)


# %%
saliency_path = "interpret/blood/saliency-LIPID-ckpt-ckpt.pkl.gz"
sal_data, targets, tokenizer, modality_name, features = load_saliency(
    Path(DELPHI_CKPT_DIR) / saliency_path
)
detokenizer = {v: k for k, v in tokenizer.items()}

# %%
sal_matrix = get_saliency_matrix(data=sal_data, n_features=len(features))
biomarker_val = get_biomarker_values(data=sal_data, modality=modality_name)
sal_timesteps, logits = get_timesteps_and_logits(data=sal_data, modality=modality_name)
bio_timesteps = get_biomarker_timesteps(data=sal_data, modality=modality_name)

# %%
((sal_timesteps - bio_timesteps) > 0).sum(), logits.shape, sal_matrix.shape


# %%
def plot_saliency_vs_value(
    sal_matrix: np.ndarray,
    biomarker_values: np.ndarray,
    feature_names: list,
    target_idx: int,
    target_name=None,
    n_cols=3,
    figsize_per_subplot=(4, 3.5),
    lowess_frac=None,
):
    """
    For each feature in the modality, scatter biomarker value (x) vs
    saliency for a given target disease (y), with optional LOWESS trend.

    Saliency is rescaled from per-z-unit to per-raw-unit (dividing by σ_f),
    and y-axis ticks show the corresponding % change in hazard rate.

    If horizon is provided (array of years per participant), scatter points
    and LOWESS lines are stratified by time horizon using STRATA bins.
    """
    import textwrap

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

        ax.scatter(
            x,
            y,
            alpha=0.3,
            s=10,
            c="steelblue",
            rasterized=True,
        )
        if lowess_frac is not None:
            sort_idx = np.argsort(x)
            smoothed = sm_lowess(y[sort_idx], x[sort_idx], frac=lowess_frac)
            ax.plot(smoothed[:, 0], smoothed[:, 1], "r-", linewidth=2)

        # ax.axhline(y=1, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel(feat)
        ax.set_ylabel("hazard ratio")
        ax.set_title(textwrap.fill(display_name, width=25), wrap=True, pad=15)

    plt.tight_layout()
    plt.show()


# %%

# %%
# Example usage: pick a target disease by name or index
# target_name = "i21_(acute_myocardial_infarction)"
# target_name = "k74_(fibrosis_and_cirrhosis_of_liver)"
# target_name = "j11_(influenza,_virus_not_identified)"
# target_name = "m10_(gout)"
# target_name = "o23_(infections_of_genito-urinary_tract_in_pregnancy)"
# target_name = "b02_(zoster_[herpes_zoster])"
# target_name = "i20_(angina_pectoris)"
# target_name =  'o47_(false_labour)'
# target_name = "e11_(non-insulin-dependent_diabetes_mellitus)"
# target_name = "n18_(chronic_renal_failure)"
# target_name = "k70_(alcoholic_liver_disease)"
# target_name = "d50_(iron_deficiency_anaemia)"
target_name = "e78_(disorders_of_lipoprotein_metabolism_and_other_lipidaemias)"
target_idx = targets.tolist().index(tokenizer[target_name])
# print(f"target: {target_name} (idx={target_idx})")

plot_saliency_vs_value(
    sal_matrix=np.exp(sal_matrix),
    biomarker_values=biomarker_val,
    feature_names=features,
    target_name=target_name,
    target_idx=target_idx,
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
mean_saliency = np.exp(sal_matrix.mean(axis=0))

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

from delphi.data.ukb import Biomarker
from delphi.env import DELPHI_DATA_DIR

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
saliency_files = [
    f"interpret/blood/saliency-{modality}-ckpt-ckpt.pkl.gz" for modality in modalities
]

# %%
with open(disease_yaml) as f:
    disease_list = yaml.safe_load(f)

all_features = []
all_mean_saliency = []  # each entry: (n_features, n_diseases)
all_var_saliency = []

for sal_file in saliency_files:
    sal_data, targets, tokenizer, modality_name, features = load_saliency(
        Path(DELPHI_CKPT_DIR) / sal_file
    )
    # normalize: convert per-raw-unit to per-z-unit by multiplying by σ
    bio = Biomarker(
        path=Path(DELPHI_DATA_DIR)
        / "ukb_real_data"
        / "biomarkers"
        / modality_name.lower(),
        stats_subjects=np.fromfile(
            Path(DELPHI_DATA_DIR) / "ukb_real_data" / "participants/train_fold.bin",
            dtype=np.uint32,
        ),
    )
    mean_sal = get_mean_saliency(n_features=len(features), data=sal_data, std=bio.std)
    var_sal = get_var_saliency(n_features=len(features), data=sal_data, std=bio.std)

    # resolve disease indices in this file's target list
    disease_indices = []
    for d in disease_list:
        tok = tokenizer[d]
        disease_indices.append(targets.tolist().index(tok))

    mean_sal = mean_sal[:, disease_indices]  # (n_features, n_diseases)
    var_sal = var_sal[:, disease_indices]

    all_features.extend(features)
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
