"""Reference-chapter vs. other-chapter intensity correlation from a forward dump.

Loads an ``apps/forward.py`` ``.npz`` (last-position per-token intensities over
the full vocab), sums each participant's intensities by ICD-10 chapter, and
draws per-sex grids of log-log scatter (or hexbin) panels with a chosen
REFERENCE chapter's summed intensity on the y-axis (``--reference_chapter``,
default ``neoplasm`` = cancer). Panel 0 is the reference vs. the summed total of
all other disease chapters + Death; the rest are the reference vs. each
individual chapter (+ Death), ordered by descending Spearman rho.

Output is always split by sex into two figures (female, then male). For the
default cancer reference this matters because cancer profiles are sex-specific
(prostate in the male sum; breast and gynaecological cancers in the female sum),
so the joint distribution differs by sex and pooling would conflate two
populations; splitting is a sensible default for any reference. (Intensities are
model predictions over the full vocab, so no panel is empty.) Each figure is
ordered independently by that sex's rho.

Chapter membership comes from ``UKBReader.labels()`` (its ``index`` column is the
vocab token id, so it aligns 1:1 with the npz ``token_ids``). The three
non-disease chapters (``Technical`` = Padding/No-event, ``Sex``, and
``Smoking, Alcohol and BMI``) are excluded so the sums reflect disease burden
rather than the large No-event intensity.
"""

import pprint
import warnings
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cloudpathlib import AnyPath
from scipy.stats import pearsonr, spearmanr

from delphi.data.ukb import UKBReader
from delphi.env import DELPHI_CKPT_WRITE
from delphi.experiment import CliConfig

mpl.rcParams["figure.dpi"] = 300

# Chapters in labels() that are not diseases; kept out of the "other" axis.
NON_DISEASE_CHAPTERS = {"Technical", "Sex", "Smoking, Alcohol and BMI"}


@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    # path to the forward.py .npz, relative to DELPHI_CKPT_WRITE (or absolute)
    npz: str = "delphi-m4/delphi-m4/forward.npz"
    # chapter on the y-axis: case-insensitive substring of the (short or full)
    # chapter name — a keyword ("neoplasm", "circulatory", "death") or a roman
    # prefix ("ix"). Must match exactly one chapter.
    reference_chapter: str = "neoplasm"
    n_cols: int = 6  # columns in the per-chapter scatter grid
    winsor: float = 5.0  # clip log tails to [winsor, 100-winsor] pct for robust rw
    hexbin: bool = False  # render density hexbins instead of overplotted points
    gridsize: int = 40  # hexbins across each panel (only used when hexbin=True)


args = TaskConfig.from_cli()
args.print()
assert 0 <= args.winsor < 50, f"winsor must be in [0, 50); got {args.winsor}"


# +
npz_path = AnyPath(args.npz)
if not npz_path.exists():
    npz_path = AnyPath(DELPHI_CKPT_WRITE) / args.npz
with npz_path.open("rb") as f:
    d = np.load(f)
    intensities = d["intensities"]  # (N, V)
    token_ids = d["token_ids"]  # (V,)
    participant_ids = d["participant_ids"]  # (N,)
print(f"loaded {intensities.shape} intensities from {npz_path}")
# -

# +
# Map each vocab column to its ICD-10 chapter via labels()['index'] == token id.
# reindex(token_ids) keeps positional order, so col_chapter[i] is column i's
# chapter (NaN for any token id absent from labels).
labels = UKBReader.labels()
chapter_by_idx = labels.set_index("index")["ICD-10 Chapter"]
col_chapter = chapter_by_idx.reindex(token_ids).to_numpy()
valid = pd.notna(col_chapter)

# per-chapter short name + hex color for panel titles and point color
chapter_meta = labels.drop_duplicates("ICD-10 Chapter").set_index("ICD-10 Chapter")[
    ["ICD-10 Chapter (short)", "color"]
]

# Sum intensities by chapter, per participant: {chapter -> (N,)}.
chapter_sums = {}
for chapter in pd.unique(col_chapter[valid]):
    cols = np.flatnonzero(col_chapter == chapter)
    chapter_sums[chapter] = intensities[:, cols].sum(axis=1)

# Resolve the reference chapter (y-axis) by case-insensitive substring of the
# full or short chapter name. Must match exactly one chapter, else error with the
# candidate list so an ambiguous query (e.g. "i") can't silently pick wrong.
ref_q = args.reference_chapter.lower()
ref_matches = [
    c
    for c in chapter_sums
    if ref_q in str(c).lower()
    or ref_q in str(chapter_meta.loc[c, "ICD-10 Chapter (short)"]).lower()
]
if len(ref_matches) != 1:
    raise SystemExit(
        f"reference_chapter={args.reference_chapter!r} matched {len(ref_matches)}: "
        f"{sorted(ref_matches)} — be more specific"
    )
(ref_chapter,) = ref_matches
ref_short = chapter_meta.loc[ref_chapter, "ICD-10 Chapter (short)"]
print(f"reference chapter (y-axis): {ref_chapter!r}")

reference = chapter_sums[ref_chapter]
other_chapters = [
    c for c in chapter_sums if c != ref_chapter and c not in NON_DISEASE_CHAPTERS
]
other = np.sum([chapter_sums[c] for c in other_chapters], axis=0)
print(f"x-axis: {len(other_chapters)} other chapters (disease + Death)")
pprint.pp(sorted(other_chapters))
# -


# +
# log-log needs strictly positive, finite values; mask per panel because each
# chapter has its own zero/underflow pattern (cancer stays full-length so every
# panel masks against the same y values).
def masked_positive(x, y):
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    return x[m], y[m]


def corr_stats(xs, ys):
    """Three correlation views for strictly-positive (xs, ys).

    rho = Spearman (rank; scale-free, tail-robust).
    r   = Pearson on log values = correlation in the displayed log-log space
          (inflated by the low-intensity tail's leverage).
    rw  = Pearson on winsorized log values (tails clipped to
          [winsor, 100-winsor] pct), so it reflects the bulk, not the tail.
    """
    if xs.size <= 1:
        return float("nan"), float("nan"), float("nan")
    lx, ly = np.log(xs), np.log(ys)
    lxw = np.clip(lx, *np.percentile(lx, [args.winsor, 100 - args.winsor]))
    lyw = np.clip(ly, *np.percentile(ly, [args.winsor, 100 - args.winsor]))
    with warnings.catch_warnings():  # constant-input subsets -> NaN, quietly
        warnings.simplefilter("ignore")
        return spearmanr(xs, ys)[0], pearsonr(lx, ly)[0], pearsonr(lxw, lyw)[0]


def plot_grid(mask, label):
    """One 18-panel reference-vs-chapter grid for the participant subset ``mask``."""
    ref_m = reference[mask]
    x_agg, y_agg = masked_positive(other[mask], ref_m)
    rho_a, r_a, rw_a = corr_stats(x_agg, y_agg)
    print(
        f"[{label}] n={ref_m.size}  aggregate Spearman={rho_a:+.3f}  "
        f"Pearson(log)={r_a:+.3f}  Pearson(winsor p{args.winsor:g})={rw_a:+.3f}"
    )

    # rank chapters by this subset's correlation with the reference (NaN rho last)
    chapter_rho = {}
    for ch in other_chapters:
        xs, ys = masked_positive(chapter_sums[ch][mask], ref_m)
        chapter_rho[ch] = spearmanr(xs, ys)[0] if xs.size > 1 else np.nan
    ordered = sorted(
        other_chapters,
        key=lambda c: (
            np.isnan(chapter_rho[c]),
            -np.nan_to_num(chapter_rho[c], nan=-np.inf),
        ),
    )

    # panel 0 = aggregate (neutral gray); panels 1.. = per-chapter, chapter-colored
    panels = [("all other (disease + Death)", other[mask], "0.4")]
    panels += [
        (
            chapter_meta.loc[ch, "ICD-10 Chapter (short)"],
            chapter_sums[ch][mask],
            chapter_meta.loc[ch, "color"],
        )
        for ch in ordered
    ]

    n_rows = int(np.ceil(len(panels) / args.n_cols))
    fig, axes = plt.subplots(
        n_rows,
        args.n_cols,
        figsize=(3.2 * args.n_cols, 3.2 * n_rows),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )
    for idx, (title, xvals, color) in enumerate(panels):
        ax = axes[divmod(idx, args.n_cols)]
        xs, ys = masked_positive(xvals, ref_m)
        if args.hexbin and xs.size:
            # bin in log space (xscale/yscale) so hexagons aren't distorted on
            # log axes; bins="log" colors by log10(count) for the heavy density.
            ax.hexbin(
                xs,
                ys,
                gridsize=args.gridsize,
                xscale="log",
                yscale="log",
                bins="log",
                mincnt=1,
                cmap="viridis",
            )
        elif xs.size:
            ax.scatter(
                xs, ys, s=4, alpha=0.01, edgecolors="none", color=color, rasterized=True
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        rho, r, rw = corr_stats(xs, ys)
        ax.set_title(
            f"{title}\nρ={rho:.2f}  r={r:.2f}  rw={rw:.2f}  n={xs.size}", fontsize=8
        )

    # hide unused axes in the final row
    for idx in range(len(panels), n_rows * args.n_cols):
        axes[divmod(idx, args.n_cols)].set_visible(False)

    fig.supxlabel("log(chapter summed intensity)")
    fig.supylabel(f"log({ref_short} summed intensity)")
    fig.suptitle(
        f"{npz_path.name} [{label}] — {ref_short} vs. chapter, ordered by Spearman ρ"
    )


# -

# +
# Always split by sex: cancers and one "other" chapter are sex-specific, so a
# pooled grid mixes two populations. Sex is the actual label, not inferred.
print(f"resolving sex for {participant_ids.size} participants via UKBReader...")
try:
    is_female = UKBReader().is_female(participant_ids)
except KeyError as e:
    raise SystemExit(f"sex split needs UKB participant ids; pid {e} not in reader")
print(f"female fraction = {is_female.mean():.3f}")

plot_grid(is_female, "female")
plot_grid(~is_female, "male")
plt.show()
# -
