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
# # Compare C-index vs offset across runs
#
# Side-by-side version of plot/cindex_offset.py: overlays several offset-curve
# JSONs (one per checkpoint / config) in the same 3-sex panel. At each offset,
# one box per file is dodged apart and colour-coded, with a per-file median trend
# line so the downward drift of each run is legible.
#
# Inputs are the `cindex_offset.json` files written by plot/cindex_offset.py
# `--write` (flat per-disease c-index arrays per sex/offset), given as `--files`
# paths relative to DELPHI_RESULTS_DIR. Boxes are drawn from each file's arrays
# AS-IS — the per-file disease sets are NOT re-intersected, so two files' boxes at
# the same offset may summarise different disease sets (each file's own n is in
# its provenance line). The x-axis is the sorted UNION of every file's offsets; a
# file missing an offset simply has no box there.

# %%
import json

# %%
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cloudpathlib import AnyPath
from matplotlib.patches import Patch

from delphi.env import DELPHI_RESULTS_DIR
from delphi.experiment import CliConfig

mpl.rcParams["figure.dpi"] = 150


@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    # cindex_offset.json files to overlay, as paths relative to DELPHI_RESULTS_DIR.
    files: None | list = None
    # Optional legend labels, one per file (default: each file's parent dir name).
    labels: None | list = None
    # If set, save the figure to results/<write>/compare_cindex_offset.png.
    write: None | str = None


args = TaskConfig.from_cli()
files = args.files or []

# %%
RESULTS = AnyPath(DELPHI_RESULTS_DIR)

OUT_DIR = None
if args.write is not None:
    OUT_DIR = RESULTS / args.write
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name):
    if OUT_DIR is None:
        return
    out_path = OUT_DIR / name
    with out_path.open("wb") as f:
        fig.savefig(f, format="png", bbox_inches="tight")
    print(f"Saved {out_path}")


# %%
def load_json(rel):
    with (RESULTS / rel).open() as f:
        return json.load(f)


runs = [load_json(f) for f in files]
labels = args.labels or [str(AnyPath(f).parent) for f in files]
assert len(labels) == len(runs), "--labels must have one entry per --files"
if not runs:
    print("pass files=[a.json,b.json] ... (relative to DELPHI_RESULTS_DIR)")
for f, r in zip(files, runs):
    print(f"{f}: ckpt={r['ckpt_dir']} min={r['min']} offsets={r['offsets']}")

# %%
# Shared categorical x-axis = sorted union of every file's offsets.
all_offsets = sorted({o for r in runs for o in r["offsets"]})
pos = {o: i + 1 for i, o in enumerate(all_offsets)}
xticklabels = [f"{o:g}" for o in all_offsets]

SEXES = [("either", "Either"), ("male", "Male"), ("female", "Female")]
n = len(runs)
box_w = 0.8 / max(n, 1)  # group width 0.8 split evenly across files
colors = [f"C{i}" for i in range(n)]

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
for ax, (sex_key, title) in zip(axes, SEXES):
    for i, r in enumerate(runs):
        dx = (i - (n - 1) / 2) * box_w  # dodge offset from the offset's center
        xs, boxes, medians = [], [], []
        for o, arr in zip(r["offsets"], r["cindex"][sex_key]):
            if not arr:  # file kept no diseases at this offset
                continue
            xs.append(pos[o] + dx)
            boxes.append(arr)
            medians.append(np.median(arr))
        if not boxes:
            continue
        bp = ax.boxplot(
            boxes, positions=xs, widths=box_w * 0.9, showfliers=False, patch_artist=True
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.5)
        for med in bp["medians"]:
            med.set_color("black")
        ax.plot(xs, medians, color=colors[i], marker="o", lw=1, zorder=3)
    ax.axhline(0.5, ls=":", c="gray", lw=1)
    ax.set_xticks(list(pos.values()), xticklabels)
    ax.set_xlabel("offset / prediction lead-time (years)")
    ax.set_title(title)
axes[0].set_ylabel("c-index")
handles = [Patch(facecolor=colors[i], alpha=0.5, label=labels[i]) for i in range(n)]
axes[-1].legend(handles=handles, fontsize=8, loc="best")
fig.suptitle("C-index vs offset — comparison")
fig.tight_layout(rect=(0, 0, 1, 0.96))
save_fig(fig, "compare_cindex_offset.png")
plt.show()
