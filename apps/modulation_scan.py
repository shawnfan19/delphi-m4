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
# # Exhaustive biomarker→disease saliency modulation scan
#
# For every (biomarker feature, disease, binary covariate), measure how much a
# covariate modulates the biomarker's saliency, *controlling for the biomarker's
# own value*:
#
# 1. per (feature, disease): residualize saliency on a value spline (FWL),
#    `r = saliency - cr(value, df)` — strips out the value's own effect, once;
# 2. per covariate token k: `effect = mean(r | k present) - mean(r | k absent)`
#    — a directional, value-controlled shift (the token's single-covariate OLS
#    coefficient on the residual). NOT a ΔR²; it keeps the sign.
#
# GATING (information redundancy: saliency collapses to ~0 once a related
# diagnosis is recorded) is flagged when the present group's raw mean|saliency|
# is a small fraction of the absent group's AND the value-controlled `effect` is
# substantial (the latter rules out value-distribution confounds, whose residual
# shift is ~0). Reads the self-describing saliency-*.npz; no model load.

# %%
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import patsy
from cloudpathlib import AnyPath

from delphi.env import DELPHI_CKPT_DIR
from delphi.experiment import CliConfig, flexi_list


# %%
@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    # directory (relative to DELPHI_CKPT_DIR) holding the saliency-<panel>.npz files
    ckpt: str = "cross-cohort/blood+urine"
    # diseases to scan: a token, an inline list, or a .yaml path (flexi_list)
    diseases: Any = "results/blood/improved.yaml"
    # biomarker panels (npz stems): a name/list/.yaml; None = all panels in the dir
    biomarkers: Any = None
    df: int = 5  # natural-cubic-spline df for the value control
    min_count: int = 150  # min present (and absent) participants per token
    abs_floor: float = 0.05  # absent-group mean|saliency| floor (marker must matter)
    gating_ratio: float = 0.45  # gating: present mean|sal| / absent below this ...
    gating_effect: float = 0.05  # ... AND |value-controlled effect| at least this
    top_k: int = 30  # rows printed per ranking
    # if set, write the full sorted table to results/<write>/modulation_scan.csv
    write: None | str = None

    def __post_init__(self):
        self.diseases = flexi_list(self.diseases)
        if self.biomarkers is not None:
            self.biomarkers = flexi_list(self.biomarkers)


args = TaskConfig.from_cli()
args.print()

ckpt_dir = AnyPath(DELPHI_CKPT_DIR) / args.ckpt
if args.biomarkers is None:
    panels = sorted(
        p.name[len("saliency-") : -len(".npz")] for p in ckpt_dir.glob("saliency-*.npz")
    )
else:
    panels = list(args.biomarkers)
assert panels, f"no saliency-*.npz panels found under {ckpt_dir}"
print(f"panels: {panels}\ndiseases: {len(args.diseases)}")


# %%
def kind_of(token: str) -> str:
    if token == "female":
        return "sex"
    if token.startswith(("bmi_", "smoking_", "alcohol_")):
        return "lifestyle"
    return "history"


def short(d: str) -> str:
    return d.split("_(")[0].split("_mal")[0]


def spline(values, df):
    return np.asarray(patsy.dmatrix(f"cr(v, df={df})", {"v": values}), dtype=np.float64)


# %%
# Accumulate one panel (file) at a time — panel jacobians are multi-GB.
rows = []
for panel in panels:
    z = np.load(ckpt_dir / f"saliency-{panel}.npz", allow_pickle=True)
    jac = z["jacobians"]  # (N, n_feat, n_targets)
    feats = z["feature_names"].tolist()
    target_names = z["target_names"].tolist()
    bio_names = z["bio_names"].tolist()
    bio_values = z["bio_values"]
    token_matrix = z["token_matrix"]
    token_names = z["token_names"].tolist()

    missing = [d for d in args.diseases if d not in target_names]
    assert not missing, f"diseases not in {panel} target_names: {missing}"

    # per-feature value spline (depends only on the value, not the disease)
    feat_spline = {
        f: spline(bio_values[:, bio_names.index(f)].astype(np.float64), args.df)
        for f in feats
    }

    for d in args.diseases:
        di = target_names.index(d)
        valid = ~np.isnan(jac[:, 0, di])  # extinguishment is feature-independent
        Mv = token_matrix[valid].astype(np.float32)
        cnt = Mv.sum(0)
        nv = Mv.shape[0]
        keep = (cnt >= args.min_count) & (cnt <= nv - args.min_count)
        if not keep.any():
            continue
        Mk = Mv[:, keep]
        cntk = cnt[keep]
        toks = [token_names[i] for i in np.flatnonzero(keep)]
        for fi, f in enumerate(feats):
            y = jac[valid, fi, di].astype(np.float64)
            S = feat_spline[f][valid]
            beta, *_ = np.linalg.lstsq(S, y, rcond=None)
            r = (y - S @ beta).astype(np.float32)
            absy = np.abs(y).astype(np.float32)
            psum_r, psum_a = Mk.T @ r, Mk.T @ absy
            pres_r = psum_r / cntk
            abs_r = (r.sum() - psum_r) / (nv - cntk)
            pres_a = psum_a / cntk
            absn_a = (absy.sum() - psum_a) / (nv - cntk)
            effect = pres_r - abs_r
            ratio = pres_a / np.maximum(absn_a, 1e-9)
            for k, tkn in enumerate(toks):
                if absn_a[k] < args.abs_floor:
                    continue
                rows.append(
                    (
                        f.split(":", 1)[1],
                        short(d),
                        tkn,
                        kind_of(tkn),
                        int(cntk[k]),
                        float(effect[k]),
                        float(pres_a[k]),
                        float(absn_a[k]),
                        float(ratio[k]),
                    )
                )
    del jac
    print(f"  scanned {panel}")

df = pd.DataFrame(
    rows,
    columns=[
        "feature",
        "disease",
        "token",
        "kind",
        "n_present",
        "effect",
        "present_abs_sal",
        "absent_abs_sal",
        "ratio",
    ],
)
df["abs_effect"] = df["effect"].abs()
df["gating"] = (
    (df["ratio"] < args.gating_ratio)
    & (df["abs_effect"] >= args.gating_effect)
    & (df["kind"] == "history")
)
print(f"\nscanned {len(df)} (feature, disease, token) cells")

# %%
k = args.top_k
gating = df[df["gating"]].sort_values("ratio")
print(
    f"\n===== GATING (present mean|sal| / absent < {args.gating_ratio}, "
    f"|value-controlled effect| >= {args.gating_effect}); top {k} ====="
)
print(
    gating.head(k)[
        [
            "feature",
            "disease",
            "token",
            "n_present",
            "effect",
            "present_abs_sal",
            "absent_abs_sal",
            "ratio",
        ]
    ].to_string(index=False)
)

sex = df[df["kind"] == "sex"].sort_values("abs_effect", ascending=False)
print(f"\n===== SEX (female) value-controlled effect; top {k} by |effect| =====")
print(
    sex.head(k)[["feature", "disease", "n_present", "effect", "ratio"]].to_string(
        index=False
    )
)

mod = df[df["kind"].isin(["lifestyle", "history"])].sort_values(
    "abs_effect", ascending=False
)
print(
    f"\n===== STRONGEST value-controlled modulation (lifestyle+history); top {k} ====="
)
print(
    mod.head(k)[
        ["feature", "disease", "token", "kind", "n_present", "effect", "ratio"]
    ].to_string(index=False)
)

# %%
if args.write is not None:
    out_dir = AnyPath(__file__).resolve().parents[1] / "results" / args.write
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "modulation_scan.csv"
    with out_path.open("w") as fh:
        df.sort_values("abs_effect", ascending=False).to_csv(fh, index=False)
    print(f"\nsaved {out_path}  ({len(df)} rows)")
