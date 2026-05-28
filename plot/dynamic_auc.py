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
# # Dynamic I/D AUC visualization
#
# Saha-Chaudhuri & Heagerty (2013) WMR estimator with a uniform rectangular kernel.
# Reads per-case time-series from a parquet produced by `apps/c-index-m4.py`,
# and plots the smoothed AUC(t) curve for a given disease, both sexes on one axes.

# %%
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cloudpathlib import AnyPath

from delphi.data.ukb import UKBReader
from delphi.env import DELPHI_CKPT_READ, DELPHI_CKPT_WRITE
from delphi.experiment import CliConfig

mpl.rcParams["figure.dpi"] = 300


@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    parquet: str
    disease: str
    # Saha-Chaudhuri & Heagerty h_n: half-width of uniform kernel in days.
    # 365.25 -> 1 year on either side, 2-year smoothing window.
    bandwidth: float = 365.25
    # If set, switch from fixed-time-window to adaptive (fixed-K) neighborhood:
    # average over the K nearest unique event times at each query. Takes
    # precedence over `bandwidth` when non-None.
    n_neighbors: None | int = None
    # Weighting within the neighborhood: "uniform" (equal) or "tricube"
    # (lowess-style, weight ~ (1 - (d/d_max)^3)^3).
    kernel: str = "uniform"
    n_grid: int = 200
    age_since_recruit: bool = False


args = TaskConfig.from_cli()

# %%
# Resolve disease: integer -> ICD via UKBReader detokenizer; string -> as-is.
reader = UKBReader()
try:
    icd = reader.detokenizer[int(args.disease)]
except (ValueError, KeyError):
    icd = args.disease
print(f"Plotting dynamic AUC for disease: {icd}")

# %%
parquet_path = AnyPath(DELPHI_CKPT_READ) / args.parquet
with parquet_path.open("rb") as f:
    df = pd.read_parquet(f, engine="pyarrow")
df = df[df["icd"] == icd]
if len(df) == 0:
    raise ValueError(f"No rows for disease '{icd}' in {parquet_path}")

# %%
# Re-express case_time as days since each participant's recruitment, if requested.
if args.age_since_recruit:
    pids = df["participant_id"].unique()
    recruit_days = reader.recruitment_times(pids)
    pid_to_recruit = dict(zip(pids, recruit_days))
    df = df.assign(
        recruit_time=df["participant_id"].map(pid_to_recruit).astype("float32"),
    )
    n_before = len(df)
    df = df.dropna(subset=["recruit_time"])
    df = df.assign(case_time=df["case_time"] - df["recruit_time"])
    df = df[df["case_time"] >= 0]
    print(
        f"age_since_recruit: kept {len(df)}/{n_before} rows after dropping NaN "
        "recruitment and pre-recruit events"
    )
    if len(df) == 0:
        raise ValueError("No rows remain after age_since_recruit filtering")


# %%
def _kernel_weights(d, d_max, kernel):
    if kernel == "uniform":
        return np.ones_like(d)
    if kernel == "tricube":
        if d_max <= 0:
            return np.ones_like(d)
        u = np.clip(d / d_max, 0.0, 1.0)
        return (1 - u**3) ** 3
    raise ValueError(f"unknown kernel: {kernel!r}")


def saha_chaudhuri_heagerty(
    case_times,
    concordant,
    total_pairs,
    query_times,
    h_days=None,
    n_neighbors=None,
    kernel="uniform",
):
    """WMR smoother.

    Neighborhood: fixed time window (`h_days`) or adaptive K nearest unique
    event times (`n_neighbors`). Weighting within the neighborhood is
    controlled by `kernel` ("uniform" or "tricube"); the two axes are
    independent.
    """
    assert (h_days is None) != (
        n_neighbors is None
    ), "specify exactly one of h_days, n_neighbors"
    agg = (
        pd.DataFrame(
            {
                "case_time": case_times,
                "concordant": concordant,
                "total_pairs": total_pairs,
            }
        )
        .groupby("case_time", as_index=False)
        .agg({"concordant": "sum", "total_pairs": "sum"})
    )
    t_k = agg["case_time"].to_numpy()
    A_k = (agg["concordant"] / agg["total_pairs"]).to_numpy()

    out = np.full_like(query_times, np.nan, dtype=float)
    for i, q in enumerate(query_times):
        d = np.abs(t_k - q)
        if n_neighbors is not None:
            k_eff = min(n_neighbors, len(t_k))
            if k_eff == 0:
                continue
            idx = np.argpartition(d, k_eff - 1)[:k_eff]
            d_local = d[idx]
            d_max = d_local.max()
            A_local = A_k[idx]
        else:
            mask = d < h_days
            if not mask.any():
                continue
            d_local = d[mask]
            d_max = h_days
            A_local = A_k[mask]
        w = _kernel_weights(d_local, d_max, kernel)
        out[i] = (w * A_local).sum() / w.sum()
    return out


# %%
query_times = np.linspace(df["case_time"].min(), df["case_time"].max(), args.n_grid)

fig, ax = plt.subplots(figsize=(8, 5))
for sex, color in [("female", "tab:red"), ("male", "tab:blue")]:
    sub = df[df["sex"] == sex]
    if len(sub) == 0:
        continue
    auc_smooth = saha_chaudhuri_heagerty(
        case_times=sub["case_time"].to_numpy(),
        concordant=sub["concordant"].to_numpy(),
        total_pairs=sub["total_pairs"].to_numpy(),
        query_times=query_times,
        h_days=args.bandwidth if args.n_neighbors is None else None,
        n_neighbors=args.n_neighbors,
        kernel=args.kernel,
    )
    ax.plot(
        query_times / 365.25, auc_smooth, color=color, label=f"{sex} (n={len(sub)})"
    )

ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
xlabel = "Years since recruitment" if args.age_since_recruit else "Age at event (years)"
ax.set_xlabel(xlabel)
ax.set_ylabel("I/D AUC")
mode_str = (
    f"k={args.n_neighbors} events"
    if args.n_neighbors is not None
    else f"h={args.bandwidth:.0f} days"
)
ax.set_title(f"Dynamic AUC — {icd} ({args.kernel} kernel, {mode_str})")
ax.set_ylim(0.4, 1.0)
ax.legend()

# %%
out_dir = AnyPath(str(parquet_path.parent).replace(DELPHI_CKPT_READ, DELPHI_CKPT_WRITE))
out_dir.mkdir(parents=True, exist_ok=True)
suffix = "_since_recruit" if args.age_since_recruit else ""
out_path = out_dir / f"dynamic_auc_{icd}{suffix}.png"
fig.tight_layout()
with out_path.open("wb") as f:
    fig.savefig(f, format="png", bbox_inches="tight")
print(f"Saved to {out_path}")
plt.show()
