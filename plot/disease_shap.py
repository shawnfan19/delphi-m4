"""Feature dependence of a single disease, from token-only SHAP.

For one disease (selected by name substring), rank the input tokens that most
drive its next-event prediction and draw a signed horizontal bar chart: bar =
mean SHAP of that token onto the disease (logit space), whisker = SD across
occurrences, red = raises risk / blue = lowers risk.

Consumes the per-patient gzip pickle written by ``apps/run_shap.py``
(``{pid: {"x", "t", "shap"}, "tokenizer", "model"}``) — token-only models only.
Biomarker SHAP is a different format/script.
"""

import gzip
import pickle
from dataclasses import dataclass
from typing import cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cloudpathlib import AnyPath

from delphi.env import DELPHI_CKPT_READ, DELPHI_RESULTS_DIR
from delphi.experiment import CliConfig, match_unique

mpl.rcParams["figure.dpi"] = 300


@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    # shap.pickle.gz produced by apps/run_shap.py, relative to DELPHI_CKPT_READ
    shap: str = "delphi-m4/delphi-m4/shap.pickle.gz"
    disease: str = "death"  # name substring, resolved against the tokenizer
    top_k: int = 20
    min_samples: int = 10  # min token occurrences to include a feature
    write: str = "disease_shap"  # output filename prefix


args = TaskConfig.from_cli()
args.print()

# +
shap_path = AnyPath(DELPHI_CKPT_READ) / args.shap
with shap_path.open("rb") as raw, gzip.open(raw, "rb") as f:
    shap_data = pickle.load(f)
tokenizer = shap_data.pop("tokenizer")  # name -> id
shap_data.pop("model", None)
detok = {v: k for k, v in tokenizer.items()}

disease_name = match_unique(args.disease, tokenizer.keys(), label="disease")
disease_idx = tokenizer[disease_name]
# -

# +
# flatten every (patient, token occurrence) into (feature token id, SHAP onto disease)
feats, vals = [], []
for pid, d in shap_data.items():
    feats.append(np.asarray(d["x"]))
    vals.append(np.asarray(d["shap"][:, disease_idx], dtype=np.float32))
feats = np.concatenate(feats)
vals = np.concatenate(vals)

g = cast(
    pd.DataFrame,
    pd.DataFrame({"feat": feats, "shap": vals})
    .groupby("feat")["shap"]
    .agg(["mean", "std", "count"]),
)
g = cast(pd.DataFrame, g[g["count"] >= args.min_samples])
if g.empty:
    raise SystemExit(
        f"no token appears in >= {args.min_samples} occurrences for "
        f"'{disease_name}'; lower min_samples or run run_shap.py on more patients"
    )
g["std"] = g["std"].fillna(0.0)  # count==1 -> NaN SD
g = g.assign(absmean=g["mean"].abs())
# rank by |mean|, keep top_k, reverse so the largest sits at the top of the barh
g = g.sort_values("absmean", ascending=False).head(args.top_k).iloc[::-1]  # type: ignore[call-overload]
# -


# +
def _label(fid: int) -> str:
    name = detok.get(int(fid), str(fid))
    return name if len(name) <= 42 else name[:39] + "..."


colors = ["#c0392b" if m > 0 else "#2980b9" for m in g["mean"]]
fig, ax = plt.subplots(figsize=(8, 0.34 * len(g) + 1.6))
ax.barh(
    range(len(g)),
    g["mean"],
    xerr=g["std"],
    color=colors,
    error_kw=dict(elinewidth=0.8, capsize=2, ecolor="0.4"),
)
ax.set_yticks(range(len(g)))
ax.set_yticklabels([_label(f) for f in g.index], fontsize=8)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("mean SHAP (logit)   → raises risk    ← lowers risk")
ax.set_title(
    f"{disease_name}: top {len(g)} predictive tokens "
    f"(n≥{args.min_samples} occurrences)",
    fontsize=10,
)

short = disease_name.split("_(")[0]
out_dir = AnyPath(DELPHI_RESULTS_DIR)
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"{args.write}_{short}.png"
fig.savefig(out_path, bbox_inches="tight")
print(f"Saved {out_path}")

plt.show()
# -
