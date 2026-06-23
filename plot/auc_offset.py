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
# # AUC vs offset (prediction lead-time)
#
# The discrimination sibling of plot/cindex_offset.py. One violin per offset,
# each the distribution of per-disease AUC across diseases, for the SAME
# checkpoint scored at different `--offset` values in apps/auc-fast-m4.py. A
# larger offset moves the scoring time to `offset` years before each disease's
# onset, so the model must rank on earlier, less-informative history — each
# violin should drift downward.
#
# Files are named explicitly: `ckpt_dir` + a list of `fnames` (JSON stems, no
# `.json`). Each file's offset is read from the run config embedded in the JSON
# (auc-fast-m4.py writes `{"config": ..., "logbook": ...}`). Parquet/JSON must
# live under DELPHI_CKPT_READ (copy from the WRITE root if the two differ),
# matching compare_auc.py.
#
# Reading caveats:
# - Each violin point is ONE disease's AUC, unweighted by event count — a
#   distribution over diseases, not a pooled AUC.
# - The age-stratified logbook is collapsed to one AUC per disease via
#   `--aggregate` (uniform = plain mean over age bins; weighted = dis_count
#   weighted); "either" pools the within-female/within-male AUCs (dis_count
#   weighted), not a cross-sex ranking.
# - Diseases are the intersection across offsets (>= --min events at EVERY
#   offset): a paired comparison of the same diseases as they get harder; a
#   disease falling below --min only at a large offset is pruned everywhere,
#   which can under-state the drift. Per-offset eligible counts are printed.

# %%
from dataclasses import dataclass

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cloudpathlib import AnyPath

from delphi.env import DELPHI_CKPT_READ as DELPHI_CKPT_DIR
from delphi.env import DELPHI_RESULTS_DIR
from delphi.experiment import CliConfig
from delphi.plot import load_auc_json, per_disease_auc

mpl.rcParams["figure.dpi"] = 150


@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    # Checkpoint directory under DELPHI_CKPT_DIR (the dir holding the JSONs).
    ckpt_dir: str = "delphi-m4/delphi-m4"
    # AUC JSON stems (filename without the `.json` suffix), one violin each.
    # They should be the same checkpoint scored at different offsets.
    fnames: None | list = None
    # Drop diseases with fewer than `min` case events in ANY listed offset.
    min: int = 50
    # Collapse a disease's per-age-bin AUCs into one number: "uniform" (plain
    # mean over bins) or "weighted" (dis_count-weighted).
    aggregate: str = "uniform"
    # If set, save the figure to results/<write>/auc_offset.png (repo root).
    write: None | str = None


args = TaskConfig.from_cli()
fnames = args.fnames or ["auc"]

# %%
ckpt_dir = AnyPath(DELPHI_CKPT_DIR) / args.ckpt_dir

OUT_DIR = None
if args.write is not None:
    OUT_DIR = AnyPath(DELPHI_RESULTS_DIR) / args.write
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name):
    if OUT_DIR is None:
        return
    out_path = OUT_DIR / name
    with out_path.open("wb") as f:
        fig.savefig(f, format="png", bbox_inches="tight")
    print(f"Saved {out_path}")


# %%
def load_run(stem):
    """(offset, logbook) for one JSON stem; offset from the embedded config."""
    path = ckpt_dir / f"{stem}.json"
    config, logbook = load_auc_json(path)
    if "offset" not in config:
        raise ValueError(
            f"{path} has no config.offset; was it written by auc-fast-m4.py?"
        )
    return float(config["offset"]), logbook


runs = sorted((load_run(s) for s in fnames), key=lambda r: r[0])
offsets = [o for o, _ in runs]
logbooks = [lb for _, lb in runs]
print(f"offsets: {offsets}")


# %%
def violins_for_sex(sex_key):
    """One AUC array per offset over the diseases kept at EVERY offset:
    >= `min` events and a defined AUC in all runs (intersection). Prints the
    per-offset eligible count vs the kept count so cross-offset attrition shows."""
    per = [per_disease_auc(lb, sex_key, args.aggregate) for lb in logbooks]
    eligible = [
        set(p.index[(p["n_events"] >= args.min) & p["auc"].notna()]) for p in per
    ]
    keep = sorted(set.intersection(*eligible)) if eligible else []
    print(
        f"[{sex_key}] keeps {len(keep)} diseases (intersection); "
        f"per-offset eligible: {dict(zip(offsets, [len(e) for e in eligible]))}"
    )
    return [p.loc[keep, "auc"].to_numpy() for p in per], keep


# %%
SEXES = [("either", "Either"), ("male", "Male"), ("female", "Female")]
positions = np.arange(len(offsets)) + 1  # categorical: even spacing
xticklabels = [f"{o:g}" for o in offsets]

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
for ax, (sex_key, title) in zip(axes, SEXES):
    arrays, keep = violins_for_sex(sex_key)
    if keep:
        ax.violinplot(arrays, positions=positions, showmedians=True)
        medians = [np.median(a) for a in arrays]
        ax.plot(positions, medians, color="C3", marker="o", lw=1, zorder=3)
    else:
        ax.text(0.5, 0.5, f"no diseases with >= {args.min} events", ha="center")
    ax.axhline(0.5, ls=":", c="gray", lw=1)
    ax.set_xticks(positions, xticklabels)
    ax.set_xlabel("offset / prediction lead-time (years)")
    ax.set_title(f"{title} (n={len(keep)} diseases)")
axes[0].set_ylabel(f"AUC ({args.aggregate} over age bins)")
fig.suptitle(f"AUC vs offset — {args.ckpt_dir} (>= {args.min} events)")
fig.tight_layout(rect=(0, 0, 1, 0.96))
save_fig(fig, "auc_offset.png")
plt.show()
