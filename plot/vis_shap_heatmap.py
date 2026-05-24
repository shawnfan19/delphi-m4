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
import argparse
import gzip
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from cloudpathlib import AnyPath
from matplotlib.patches import Patch
from tqdm import tqdm

from delphi.data.ukb import load_label_meta
from delphi.multimodal import Modality

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--shap_path", type=str)
parser.add_argument("--top_k_diseases", type=int)
parser.add_argument("--figsize_scale", type=float, default=0.1)
parser.add_argument("--top_k_tokens", type=int)
parser.add_argument("--output", type=str, help="Path to save figure (optional)")

if "ipykernel" in sys.modules:
    args = parser.parse_args([])
    args.shap_path = "delphi-m4/prescriptions/shap.pickle.gz"
    args.output = "shap_biomarker_heatmap"
else:
    args = parser.parse_args()

# %% Load pickle
shap_path = AnyPath(args.shap_path)
if not shap_path.exists():
    from delphi.env import DELPHI_CKPT_READ, DELPHI_CKPT_WRITE

    shap_path = AnyPath(DELPHI_CKPT_READ) / args.shap_path
    out_dir = AnyPath(
        str(shap_path.parent).replace(DELPHI_CKPT_READ, DELPHI_CKPT_WRITE)
    )
else:
    out_dir = shap_path.parent

with shap_path.open("rb") as raw, gzip.open(raw, "rb") as f:
    shap_pickle = pickle.load(f)

tokenizer = shap_pickle.pop("tokenizer")
biomarker_features = shap_pickle.pop("biomarker_features")
detokenizer = {v: k for k, v in tokenizer.items()}

# %%

# %%
biomarker_features

# %%

# %% Build flattened feature list and modality offsets
feature_names = []
mod_offset = {}
for mod, feat_list in biomarker_features.items():
    mod_offset[mod] = len(feature_names)
    feature_names.extend([f"{mod.name}.{f}" for f in feat_list])

n_features = len(feature_names)
print(f"{n_features} biomarker features across {len(biomarker_features)} modalities")

# %% Accumulate mean |SHAP| across participants
sample_entry = next(iter(shap_pickle.values()))

if "bio_shap" in sample_entry.keys():
    n_vocab = sample_entry["bio_shap"].shape[-1]
elif "shap" in sample_entry.keys():
    n_vocab = sample_entry["shap"].shape[-1]
else:
    raise ValueError

sum_abs = np.zeros((n_features, n_vocab), dtype=np.float64)
counts = np.zeros(n_features, dtype=np.int64)

tok_sum_abs = np.zeros((n_vocab, n_vocab), dtype=np.float64)
tok_counts = np.zeros(n_vocab, dtype=np.int64)

for pid, entry in tqdm(shap_pickle.items()):

    if "bio_shap" in entry.keys():
        bio_m = entry["bio_m"]
        bio_shap = entry["bio_shap"].astype(np.float64)

        local_offset = 0
        for mval in bio_m:
            mod = Modality(int(mval))
            n_feat = len(biomarker_features[mod])
            global_offset = mod_offset[mod]

            sum_abs[global_offset : global_offset + n_feat] += np.abs(
                bio_shap[local_offset : local_offset + n_feat]
            )
            counts[global_offset : global_offset + n_feat] += 1
            local_offset += n_feat

    # Token SHAP accumulation
    if "shap" in entry and "x" in entry:
        tok_x = entry["x"]
        tok_shap = entry["shap"].astype(np.float64)
        for i, token_id in enumerate(tok_x):
            tok_sum_abs[token_id] += np.abs(tok_shap[i])
            tok_counts[token_id] += 1

present = counts > 0
mean_abs = np.zeros_like(sum_abs)
mean_abs[present] = sum_abs[present] / counts[present, np.newaxis]

tok_present = tok_counts > 0
tok_mean_abs = np.zeros_like(tok_sum_abs)
tok_mean_abs[tok_present] = (
    tok_sum_abs[tok_present] / tok_counts[tok_present, np.newaxis]
)

print(f"{present.sum()}/{n_features} features have data")
print(f"{tok_present.sum()}/{n_vocab} tokens have data")
print(f"Participants: {len(shap_pickle)}")

# %%
tok_sum_abs.shape

# %% Filter and sort diseases by ICD chapter
label_meta = load_label_meta()

to_exclude = [
    "Technical",
    "Smoking, Alcohol and BMI",
    "Sex",
    "XVI. Perinatal Conditions",
    "Death",
]

chapter_order = [
    "I. Infectious Diseases",
    "II. Neoplasms",
    "III. Blood & Immune Disorders",
    "IV. Metabolic Diseases",
    "V. Mental Disorders",
    "VI. Nervous System Diseases",
    "VII. Eye Diseases",
    "VIII. Ear Diseases",
    "IX. Circulatory Diseases",
    "X. Respiratory Diseases",
    "XI. Digestive Diseases",
    "XII. Skin Diseases",
    "XIII. Musculoskeletal Diseases",
    "XIV. Genitourinary Diseases",
    "XV. Pregnancy & Childbirth",
    "XVII. Congenital Abnormalities",
]

disease_indices = np.arange(n_vocab)

# Exclude chapters
keep_mask = np.array(
    [
        d < len(label_meta)
        and label_meta.loc[d, "ICD-10 Chapter (short)"] not in to_exclude
        for d in disease_indices
    ]
)
disease_indices = disease_indices[keep_mask]
score_matrix = mean_abs[:, keep_mask]

# Sort by chapter order
sort_order = np.argsort(
    [
        chapter_order.index(label_meta.loc[d, "ICD-10 Chapter (short)"])
        for d in disease_indices
    ]
)
disease_indices = disease_indices[sort_order]
score_matrix = score_matrix[:, sort_order]
disease_names = [detokenizer.get(d, f"Disease {d}") for d in disease_indices]

# Top-K diseases by max mean |SHAP|
if args.top_k_diseases:
    max_per_disease = np.max(score_matrix, axis=0)
    top_idx = np.sort(np.argsort(max_per_disease)[::-1][: args.top_k_diseases])
    score_matrix = score_matrix[:, top_idx]
    disease_indices = disease_indices[top_idx]
    disease_names = [disease_names[i] for i in top_idx]


# %% Plot heatmap
def get_tick_coords(arr):
    arr = np.asarray(arr)
    return np.where(arr[1:] != arr[:-1])[0]


col_chapters = label_meta.loc[disease_indices, "ICD-10 Chapter (short)"].values
col_colors = label_meta.loc[disease_indices, "color"].values

modality_of_feature = [f.split(".")[0] for f in feature_names]
unique_modalities = list(dict.fromkeys(modality_of_feature))
modality_color_map = {
    m: plt.cm.tab20(i / max(len(unique_modalities) - 1, 1))
    for i, m in enumerate(unique_modalities)
}
row_colors = [modality_color_map[m] for m in modality_of_feature]

n_feat, n_dis = score_matrix.shape
fig_w = max(8, n_dis * args.figsize_scale)
fig_h = max(4, n_feat * args.figsize_scale)

g = sns.clustermap(
    score_matrix,
    row_cluster=False,
    col_cluster=False,
    row_colors=row_colors,
    col_colors=col_colors,
    cmap="magma",
    vmin=0,
    figsize=(fig_w, fig_h),
    rasterized=True,
    xticklabels=disease_names,
    yticklabels=feature_names,
)

# Chapter boundary ticks on x-axis
x_tick_coords = get_tick_coords(col_chapters)
g.ax_heatmap.set_xticks(x_tick_coords)
g.ax_heatmap.set_xticklabels([])

# Modality boundary ticks on y-axis
y_tick_coords = get_tick_coords(modality_of_feature)
g.ax_heatmap.set_yticks(y_tick_coords)
g.ax_heatmap.set_yticklabels([])

g.ax_heatmap.tick_params(
    length=0,
    width=0.5,
    labelsize=8,
    grid_alpha=0.6,
    grid_linewidth=0.35,
    grid_color="gray",
)
g.ax_cbar.tick_params(length=0.5, width=0.6, labelsize=8)

# Chapter labels along x-axis
chapter_color_pairs = label_meta[["ICD-10 Chapter (short)", "color"]].drop_duplicates(
    "ICD-10 Chapter (short)"
)
chapter_color_pairs = chapter_color_pairs[
    ~chapter_color_pairs["ICD-10 Chapter (short)"].isin(to_exclude)
]
for ch, color in chapter_color_pairs.values:
    positions = np.where(col_chapters == ch)[0]
    if len(positions) > 0:
        g.ax_heatmap.text(
            positions.mean(), -2, ch, va="bottom", rotation=00, ha="center", size=7
        )

# Modality labels along y-axis
modality_arr = np.array(modality_of_feature)
for m in unique_modalities:
    positions = np.where(modality_arr == m)[0]
    if len(positions) > 0:
        g.ax_heatmap.text(-2, positions.mean(), m, va="center", ha="right", size=8)

# Legends
chapter_handles = [
    Patch(facecolor=color, label=ch) for ch, color in chapter_color_pairs.values
]
modality_handles = [
    Patch(facecolor=modality_color_map[m], label=m) for m in unique_modalities
]

legend1 = g.ax_heatmap.legend(
    handles=chapter_handles,
    title="ICD-10 Chapter",
    bbox_to_anchor=(1.3, 1),
    loc="upper left",
    fontsize=7,
    title_fontsize=8,
)
g.ax_heatmap.add_artist(legend1)
g.ax_heatmap.legend(
    handles=modality_handles,
    title="Modality",
    bbox_to_anchor=(1.3, 0.3),
    loc="upper left",
    fontsize=7,
    title_fontsize=8,
)

plt.suptitle(
    f"Mean |SHAP| per Biomarker Feature ({n_feat} features, {n_dis} diseases, n={len(shap_pickle):,})",
    y=1.02,
    size=10,
)

if args.output:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.output}.png"
    with out_path.open("wb") as f:
        g.savefig(f, format="png", dpi=300, bbox_inches="tight")
    print(f"Saved to {out_path}")

plt.show()

# %%
score_matrix.shape

# %%
for feature_idx, feature_name in enumerate(feature_names):
    topk_target_idx = np.argsort(score_matrix[feature_idx, :])[-10:]

    print(feature_name, [detokenizer[i] for i in topk_target_idx])

# %% Build token-by-targets heatmap
### Filter to tokens that have data and are valid disease labels
tok_row_mask = tok_present & np.array([i < len(label_meta) for i in range(n_vocab)])
tok_row_indices = np.where(tok_row_mask)[0]

# Exclude same chapters as before on the row side
tok_row_mask2 = np.array(
    [
        label_meta.loc[d, "ICD-10 Chapter (short)"] not in to_exclude
        for d in tok_row_indices
    ]
)
tok_row_indices = tok_row_indices[tok_row_mask2]

# Reuse the same column filtering (disease_indices already filtered & sorted)
tok_score_matrix = tok_mean_abs[np.ix_(tok_row_indices, disease_indices)]
tok_row_names = [detokenizer.get(d, f"Token {d}") for d in tok_row_indices]

# Sort rows by ICD chapter
tok_row_chapters = label_meta.loc[tok_row_indices, "ICD-10 Chapter (short)"].values
tok_row_sort = np.argsort(
    [
        chapter_order.index(ch) if ch in chapter_order else len(chapter_order)
        for ch in tok_row_chapters
    ]
)
tok_row_indices = tok_row_indices[tok_row_sort]
tok_score_matrix = tok_score_matrix[tok_row_sort]
tok_row_names = [tok_row_names[i] for i in tok_row_sort]
tok_row_chapters = tok_row_chapters[tok_row_sort]
tok_row_colors = label_meta.loc[tok_row_indices, "color"].values

# Top-K tokens by max mean |SHAP|
if args.top_k_tokens:
    max_per_token = np.max(tok_score_matrix, axis=1)
    top_idx = np.sort(np.argsort(max_per_token)[::-1][: args.top_k_tokens])
    tok_score_matrix = tok_score_matrix[top_idx]
    tok_row_indices = tok_row_indices[top_idx]
    tok_row_names = [tok_row_names[i] for i in top_idx]
    tok_row_chapters = tok_row_chapters[top_idx]
    tok_row_colors = tok_row_colors[top_idx]

n_tok, n_dis_tok = tok_score_matrix.shape
tok_fig_w = max(8, n_dis_tok * args.figsize_scale)
tok_fig_h = max(4, n_tok * args.figsize_scale)

g2 = sns.clustermap(
    tok_score_matrix,
    row_cluster=False,
    col_cluster=False,
    row_colors=tok_row_colors,
    col_colors=col_colors,
    cmap="magma",
    vmin=0,
    figsize=(tok_fig_w, tok_fig_h),
    rasterized=True,
    xticklabels=disease_names,
    yticklabels=tok_row_names,
)

# Chapter boundary ticks on x-axis
x_tick_coords2 = get_tick_coords(col_chapters)
g2.ax_heatmap.set_xticks(x_tick_coords2)
g2.ax_heatmap.set_xticklabels([])

# Chapter boundary ticks on y-axis (row tokens grouped by chapter)
y_tick_coords2 = get_tick_coords(tok_row_chapters)
g2.ax_heatmap.set_yticks(y_tick_coords2)
g2.ax_heatmap.set_yticklabels([])

g2.ax_heatmap.tick_params(
    length=0,
    width=0.5,
    labelsize=8,
    grid_alpha=0.6,
    grid_linewidth=0.35,
    grid_color="gray",
)
g2.ax_cbar.tick_params(length=0.5, width=0.6, labelsize=8)

# Chapter labels along x-axis
for ch, color in chapter_color_pairs.values:
    positions = np.where(col_chapters == ch)[0]
    if len(positions) > 0:
        g2.ax_heatmap.text(
            positions.mean(), -2, ch, va="bottom", rotation=90, ha="center", size=7
        )

# Chapter labels along y-axis
for ch, color in chapter_color_pairs.values:
    positions = np.where(tok_row_chapters == ch)[0]
    if len(positions) > 0:
        g2.ax_heatmap.text(
            -2, positions.mean(), ch, va="center", ha="right", size=7, rotation=0
        )

# Legend
chapter_handles2 = [
    Patch(facecolor=color, label=ch) for ch, color in chapter_color_pairs.values
]
g2.ax_heatmap.legend(
    handles=chapter_handles2,
    title="ICD-10 Chapter",
    bbox_to_anchor=(1.3, 1),
    loc="upper left",
    fontsize=7,
    title_fontsize=8,
)

plt.suptitle(
    f"Mean |SHAP| per Token (input) x Target (output) ({n_tok} tokens, {n_dis_tok} targets, n={len(shap_pickle):,})",
    y=1.02,
    size=10,
)

if args.output:
    out_path = out_dir / f"{args.output}_token.png"
    with out_path.open("wb") as f:
        g2.savefig(f, format="png", dpi=300, bbox_inches="tight")
    print(f"Saved to {out_path}")

plt.show()

# %%
