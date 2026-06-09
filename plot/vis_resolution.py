"""Visualize the generation-vs-ground-truth resolution metrics (apps/resolution.py).

Loads a ``resolution.py`` ``.npz`` — per prompt, the conditional log-likelihood
of the real continuation and of K sampled generations, plus two best-of-K
trajectory metrics (EventFlow sequence distance and mark-overlap recall) — and
draws four figures relevant to memorization:

1. **Per-token log-likelihood, generations vs ground truth** (joint / marks /
   times, three overlay panels). ``gen_ll`` / ``ll`` are *summed* conditional LLs,
   so we divide each by its token count (``gen_n_events`` / ``n_events``) to get a
   per-token mean comparable across trajectory lengths. All N x K generations are
   flattened and overlaid (density-normalized) against the N ground-truth values;
   a model that memorized places the ground truth to the *right* (less surprising
   per token) of its own samples.
2. **Best-of-K sequence distance** — ``min`` over the K generations of the
   self-anchored sequence distance (a mean per-event "when" error), shown as the
   SAME histogram twice side-by-side: stacked by GT continuation-event count
   (``gt_n_real``) and by the comparison horizon (``gt_horizon`` = prompt cutoff
   to last GT event), exposing how trajectory length and window length drive it.
3. **Best-of-K mark overlap** — ``max`` over the K generations of the recall of
   ground-truth continuation marks (the "what"), stacked by ``gt_n_real``.
4. **Rank of the ground-truth per-token LL among its K generations** (stacked by
   ``gt_n_real``) — per prompt, how many of the K sampled trajectories *out-score*
   the ground truth (0 = GT most likely, beats all K; K = GT least likely). A
   uniform histogram means the GT is exchangeable with the model's own samples; a
   left-skew (mass near 0) means the GT is more likely than its own generations —
   a memorization signal.

Mirrors plot/ridge_profile.py's conventions (CliConfig, AnyPath npz loader).
"""

import warnings
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cloudpathlib import AnyPath

from delphi.env import DELPHI_CKPT_WRITE
from delphi.experiment import CliConfig

mpl.rcParams["figure.dpi"] = 300

GEN_COLOR = "#4477aa"  # generations (N x K)
GT_COLOR = "#cc6677"  # ground truth (N)


@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    # path to the resolution.py .npz, relative to DELPHI_CKPT_WRITE (or absolute)
    npz: str = "cross-cohort/baseline/resolution_train_K16.npz"
    bins: int = 60
    clip_quantile: float = 0.99  # robust upper x-limit for the seq-distance tail


args = TaskConfig.from_cli()
args.print()


# +
npz_path = AnyPath(args.npz)
if not npz_path.exists():
    npz_path = AnyPath(DELPHI_CKPT_WRITE) / args.npz
with npz_path.open("rb") as f:
    d = np.load(f)
    # per-generation (N, K): summed conditional LL + the token count behind it
    gen_ll = d["gen_ll"]
    gen_ll_m = d["gen_ll_m"]
    gen_ll_t = d["gen_ll_t"]
    gen_n_events = d["gen_n_events"]
    seq_dist = d[
        "seq_dist"
    ]  # mean per-event EventFlow distance >=0; NaN if gen is empty
    overlap = d["overlap"]  # mark recall in [0, 1], NaN if GT has no cont. marks
    # per-prompt ground truth (N,)
    ll = d["ll"]
    ll_m = d["ll_m"]
    ll_t = d["ll_t"]
    n_events = d["n_events"]
    # GT continuation event count for stacking (added by a later resolution.py;
    # absent in older npz -> the bottom panels fall back to plain histograms).
    # First-occurrence data: this == # distinct marks == both metric denominators.
    gt_n_real = d["gt_n_real"] if "gt_n_real" in d.files else None
    # comparison horizon = days from the prompt cutoff to the GT's last event
    gt_horizon = d["gt_horizon"] if "gt_horizon" in d.files else None
N, K = gen_ll.shape
print(f"loaded {N} prompts x {K} generations from {npz_path}")
# -


# +
def per_token(ll_sum, n):
    """Summed conditional LL -> per-token mean; NaN where there are no tokens."""
    n = np.asarray(n, dtype=float)
    return np.where(
        n > 0, np.asarray(ll_sum, dtype=float) / np.where(n > 0, n, 1.0), np.nan
    )


def finite(x):
    """Flatten to 1-D and drop non-finite entries (NaN/inf)."""
    x = np.asarray(x, dtype=float).ravel()
    return x[np.isfinite(x)]


def overlay(ax, gen_vals, gt_vals, xlabel, title):
    """Density-overlaid per-token LL: flattened generations vs ground truth."""
    g, t = finite(gen_vals), finite(gt_vals)
    lo = min(g.min(), t.min())
    hi = max(g.max(), t.max())
    if hi <= lo:  # all values identical: widen so density bins aren't zero-width
        lo, hi = lo - 0.5, hi + 0.5
    edges = np.linspace(lo, hi, args.bins + 1)
    ax.hist(
        g,
        bins=edges,
        density=True,
        alpha=0.5,
        color=GEN_COLOR,
        label=f"generations (n={g.size})",
    )
    ax.hist(
        t,
        bins=edges,
        density=True,
        alpha=0.5,
        color=GT_COLOR,
        label=f"ground truth (n={t.size})",
    )
    ax.axvline(np.median(g), color=GEN_COLOR, ls="--", lw=1)
    ax.axvline(np.median(t), color=GT_COLOR, ls="--", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(fontsize=8)


def stacked_hist(
    ax, values, group_idx, labels, edges, xlabel, title, legend_title, ncol=2
):
    """Stacked histogram of ``values`` over ``edges``, grouped by integer ``group_idx``.

    ``group_idx`` is 0..len(labels)-1; entries with ``group_idx < 0`` or non-finite
    ``values`` are dropped. Groups are colored along viridis in label order.
    """
    values = np.asarray(values, dtype=float)
    group_idx = np.asarray(group_idx)
    keep = np.isfinite(values) & (group_idx >= 0)
    values, group_idx = values[keep], group_idx[keep]
    groups = [values[group_idx == g] for g in range(len(labels))]
    colors = mpl.colormaps["viridis"](np.linspace(0.1, 0.9, len(labels)))
    ax.hist(groups, bins=edges, stacked=True, color=colors, label=labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("prompts")
    ax.set_title(title)
    ax.legend(title=legend_title, fontsize=7, title_fontsize=7, ncol=ncol)


def count_groups(counts, max_bucket=5):
    """Map integer counts to buckets 1, 2, ..., max_bucket-1, ``max_bucket``+ (else -1)."""
    counts = np.asarray(counts)
    idx = np.where(counts >= 1, np.clip(counts, 1, max_bucket).astype(int) - 1, -1)
    labels = [str(c) for c in range(1, max_bucket)] + [f"{max_bucket}+"]
    return idx, labels


_YR = 365.25
# fixed horizon bands (days): <1mo, 1-6mo, 6-12mo, 1-5y, 5-10y, >10y
HORIZON_CUTS = np.array([_YR / 12, _YR / 2, _YR, 5 * _YR, 10 * _YR])
HORIZON_LABELS = ["<1mo", "1-6mo", "6-12mo", "1-5y", "5-10y", ">10y"]


def horizon_groups(horizon):
    """Bin the comparison horizon (days) into the fixed bands above (non-finite -> -1)."""
    horizon = np.asarray(horizon, dtype=float)
    idx = np.digitize(horizon, HORIZON_CUTS)
    return np.where(np.isfinite(horizon), idx, -1), HORIZON_LABELS


# per-token LL (divide summed LL by its token count); generations flattened N x K
gen_j, gen_m, gen_t = (per_token(x, gen_n_events) for x in (gen_ll, gen_ll_m, gen_ll_t))
gt_j, gt_m, gt_t = (per_token(x, n_events) for x in (ll, ll_m, ll_t))

# best-of-K trajectory metrics (one value per prompt); suppress all-NaN-row warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    min_seq = np.nanmin(seq_dist, axis=1)  # closest generation's "when" error
    max_overlap = np.nanmax(overlap, axis=1)  # best generation's "what" recall
# -


# +
# Figure 1 — per-token LL overlays (the memorization contrast).
fig1, (ax_j, ax_m, ax_t) = plt.subplots(
    1, 3, figsize=(15, 4.5), constrained_layout=True
)
overlay(ax_j, gen_j, gt_j, "per-token joint log-likelihood", "Joint LL")
overlay(ax_m, gen_m, gt_m, "per-token mark log-likelihood", "Marks LL ('what')")
overlay(ax_t, gen_t, gt_t, "per-token time log-likelihood", "Times LL ('when')")
fig1.suptitle(
    f"{npz_path.name} — per-token log-likelihood: generations vs ground "
    f"truth (N={N} prompts x K={K})"
)
plt.show()
# -


# +
# Figure 2 — best-of-K sequence distance: the SAME histogram stacked two ways
# (by GT continuation-event count, and by the comparison horizon).
if gt_n_real is None or gt_horizon is None:
    print(
        "note: gt_n_real/gt_horizon absent from this npz — re-run resolution.py to "
        "enable the stacked histograms; using plain histograms where missing."
    )

xlab_sd = "min sequence distance over K (best-of-K)"
v = finite(min_seq)
hi = float(np.quantile(v, args.clip_quantile)) or 1.0  # `or` guards an all-zero tail
frac_over = float(np.mean(v > hi))
edges = np.linspace(0, hi, args.bins + 1)
vals = np.minimum(
    min_seq, hi
)  # clip the tail into the last bin; empty-gen NaN stays NaN

fig2, (ax_c, ax_h) = plt.subplots(
    1, 2, figsize=(14, 5), sharey=True, constrained_layout=True
)
if gt_n_real is not None:
    gi, gl = count_groups(gt_n_real)
    stacked_hist(
        ax_c, vals, gi, gl, edges, xlab_sd, "stacked by GT events", "GT events"
    )
else:
    ax_c.hist(finite(vals), bins=edges, color="#228833")
    ax_c.set(xlabel=xlab_sd, ylabel="prompts", title="(gt_n_real absent)")
if gt_horizon is not None:
    hi_idx, hl = horizon_groups(gt_horizon)
    stacked_hist(
        ax_h, vals, hi_idx, hl, edges, xlab_sd, "stacked by horizon", "horizon", ncol=2
    )
else:
    ax_h.hist(finite(vals), bins=edges, color="#228833")
    ax_h.set(xlabel=xlab_sd, ylabel="prompts", title="(gt_horizon absent)")
fig2.suptitle(f"Best-of-K sequence distance — median={np.median(v):.2f}")
plt.show()
# -


# +
# Figure 3 — best-of-K mark recall, stacked by GT continuation-event count.
xlab_ov = "max mark overlap over K (best-of-K recall)"
v = finite(max_overlap)
edges = np.linspace(0, 1, args.bins + 1)
title_ov = (
    "Best-of-K mark recall — no prompts with continuation marks"
    if v.size == 0
    else f"Best-of-K mark recall — median={np.median(v):.2f}"
)
fig3, ax_ov = plt.subplots(figsize=(7, 5), constrained_layout=True)
if gt_n_real is not None:
    gi, gl = count_groups(gt_n_real)
    stacked_hist(ax_ov, max_overlap, gi, gl, edges, xlab_ov, title_ov, "GT events")
else:
    ax_ov.hist(v, bins=edges, color="#aa3377")
    ax_ov.set(xlabel=xlab_ov, ylabel="prompts", title=title_ov)
plt.show()
# -


# +
# Figure 4 — rank of the GT's per-token (joint) LL among its K generations, stacked
# by GT continuation-event count. rank = # generations that out-score the GT
# (0 = GT highest / beats all K; K = GT lowest). Uniform (dashed line) => GT is
# exchangeable with the model's own samples; left-skew (mass at 0) => GT more
# likely than its generations, a memorization signal. Per-token LL is finite for
# every generation (>= 1 continuation token each), so no prompt is dropped.
rank = (gen_j > gt_j[:, None]).sum(1)  # (N,); 0 = GT highest, K = GT lowest
top = float(np.mean(rank == 0))  # GT beats all K generations (rank 0)
edges_r = np.arange(-0.5, K + 1.5)  # one bin per integer rank 0..K
xlab_r = (
    f"rank of ground truth (GT) per-token LL among K={K} generations "
    f"(0 = GT highest, {K} = GT lowest)"
)
title_r = (
    f"ground truth (GT)-vs-generation per-token LL rank "
    f"(GT > all {K} gens in {top:.1%} of prompts)"
)
fig4, ax_r = plt.subplots(figsize=(8, 5), constrained_layout=True)
if gt_n_real is not None:
    gi, gl = count_groups(gt_n_real)
    stacked_hist(ax_r, rank, gi, gl, edges_r, xlab_r, title_r, "GT events")
else:
    ax_r.hist(rank, bins=edges_r, color="#882255")
    ax_r.set(xlabel=xlab_r, ylabel="prompts", title=title_r)
ax_r.axhline(N / (K + 1), color="0.4", ls="--", lw=1)  # uniform reference
plt.show()
# -
