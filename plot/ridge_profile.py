"""Profile low-disease-risk ("ridge") participants vs the rest from a forward dump.

A scatter of summed per-chapter intensities (see ``plot/chapter_correlate.py``)
shows a tight "stick" of highly-correlated chapters at low predicted risk that
fans into a decorrelated "blob" at high risk. This script characterizes *who*
the low-risk (ridge) participants are.

It loads an ``apps/forward.py`` ``.npz`` (per-token intensities at the prompt
cutoff), defines each participant's predicted disease hazard ``risk`` = the sum
of intensities over disease tokens (all chapters except Technical/Sex/lifestyle),
and pulls per-participant history covariates from ``UKBReader``: number of prompt
tokens, disease events, and distinct ICD chapters before the cutoff, plus sex.
It then compares the ridge (bottom ``--ridge_quantile`` of risk) against the rest
and visualizes the shared-baseline mechanism (low-risk log-intensity vectors are
near-degenerate — they differ mainly by a single shared scalar, so all chapters
co-scale, producing the stick).

Finding (UKB, age-60 prompt): the ridge is overwhelmingly people with ~no disease
history (median 0 disease events; ~78% have zero), and predicted risk rises
monotonically with disease-event count. The relationship is real but not a
tautology: disease-event count alone explains only ~40% of log-risk variance —
*which* disease and *when* drives the rest (e.g. at zero events risk still spans
orders of magnitude). Requires UKB participant ids (uses ``UKBReader``).
"""

from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cloudpathlib import AnyPath
from scipy.stats import rankdata
from tqdm import tqdm

from delphi.data.ukb import MultimodalUKBReader
from delphi.env import DELPHI_CKPT_WRITE
from delphi.experiment import CliConfig

mpl.rcParams["figure.dpi"] = 300

# Chapters in labels() that are not diseases; excluded from the disease-hazard sum.
NON_DISEASE_CHAPTERS = {"Technical", "Sex", "Smoking, Alcohol and BMI"}


@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    # path to the forward.py .npz, relative to DELPHI_CKPT_WRITE (or absolute)
    npz: str = "delphi-m4/delphi-m4/forward.npz"
    ridge_quantile: float = 0.10  # bottom quantile of risk = "ridge"
    blob_quantile: float = 0.90  # top quantile of risk = "blob"
    pca_sample: int = 6000  # participants sampled for the PCA panels
    seed: int = 0


args = TaskConfig.from_cli()
args.print()


# +
npz_path = AnyPath(args.npz)
if not npz_path.exists():
    npz_path = AnyPath(DELPHI_CKPT_WRITE) / args.npz
with npz_path.open("rb") as f:
    d = np.load(f)
    intensities = d["intensities"]  # (N, V)
    token_ids = d["token_ids"]  # (V,)
    participant_ids = d["participant_ids"]  # (N,)
    prompt_age = d["prompt_age"].astype(float)  # (N,) days
print(f"loaded {intensities.shape} intensities from {npz_path}")

# disease-token columns = every chapter except the non-disease ones (keeps Death
# and Neoplasms); risk = total predicted disease hazard at the prompt.
labels = MultimodalUKBReader.labels()
col_chapter = labels.set_index("index")["ICD-10 Chapter"].reindex(token_ids).to_numpy()
disease_col = (
    pd.notna(col_chapter)
    & ~pd.Series(col_chapter).isin(NON_DISEASE_CHAPTERS).to_numpy()
)
risk = intensities[:, disease_col].sum(axis=1)
# -

# +
# Per-participant history covariates from the token reader: count tokens before
# each participant's prompt cutoff, splitting disease events from the structural
# (sex/lifestyle/no_event/padding) scaffold.
reader = MultimodalUKBReader()
female_tok = reader.tokenizer["female"]
whitelist = np.array(
    sorted(
        {
            reader.tokenizer["padding"],
            reader.tokenizer["no_event"],
            *[reader.tokenizer[k] for k in reader.sex_keys],
            *[reader.tokenizer[k] for k in reader.lifestyle_keys],
        }
    )
)
# token id -> integer chapter code (disease tokens only; -1 otherwise)
chap_code = pd.Series(col_chapter).astype("category").cat.codes.to_numpy().copy()
chap_code[~disease_col] = -1

n = participant_ids.size
n_prompt = np.zeros(n, np.int32)
n_disease = np.zeros(n, np.int32)
n_chapters = np.zeros(n, np.int32)
seq_len_total = np.zeros(n, np.int32)
is_female = np.zeros(n, bool)
for i, pid in enumerate(tqdm(participant_ids, desc="covariates", mininterval=5)):
    toks, times = reader.token_reader[int(pid)]
    seq_len_total[i] = toks.size
    is_female[i] = (toks == female_tok).any()
    pt = toks[times <= prompt_age[i]]  # prompt = events before the cutoff
    n_prompt[i] = pt.size
    dt = pt[~np.isin(pt, whitelist)]  # disease events only
    n_disease[i] = dt.size
    if dt.size:
        codes = chap_code[dt]
        n_chapters[i] = np.unique(codes[codes >= 0]).size

df = pd.DataFrame(
    {
        "risk": risk,
        "log_risk": np.log(risk),
        "n_prompt_tokens": n_prompt,
        "n_disease_tokens": n_disease,
        "n_chapters": n_chapters,
        "seq_len_total": seq_len_total,
        "is_female": is_female,
    }
)
# -

# +
# Ridge / blob / rest by risk quantiles.
ridge_thr = df.risk.quantile(args.ridge_quantile)
blob_thr = df.risk.quantile(args.blob_quantile)
ridge = (df.risk <= ridge_thr).to_numpy()
blob = (df.risk >= blob_thr).to_numpy()
rest = ~ridge
df["decile"] = pd.qcut(df.risk, 10, labels=False)


def auc_score(score, positive):
    """AUC of `score` separating `positive` (bool) from the rest, via ranks."""
    r = rankdata(score)
    npos = int(positive.sum())
    nneg = positive.size - npos
    return (
        (r[positive].mean() - (npos + 1) / 2) / nneg if npos and nneg else float("nan")
    )


def pc12_fraction(mask, rng):
    """Fraction of log-intensity variance in PC1-2 within a participant subset."""
    idx = np.flatnonzero(mask)
    if idx.size > args.pca_sample:
        idx = rng.choice(idx, args.pca_sample, replace=False)
    lx = np.log(intensities[idx][:, disease_col])
    lx = lx - lx.mean(axis=0)
    s = np.linalg.svd(lx, full_matrices=False, compute_uv=False)
    v = s**2 / (s**2).sum()
    return float(v[:2].sum())


rng = np.random.default_rng(args.seed)
ridge_pc = pc12_fraction(ridge, rng)
blob_pc = pc12_fraction(blob, rng)

print("\n=== ridge vs rest (covariate comparison) ===")
print(
    f"risk q{args.ridge_quantile:g}={ridge_thr:.3f}  q{args.blob_quantile:g}={blob_thr:.3f}"
)
print(f"ridge n={ridge.sum()}  rest n={rest.sum()}  blob n={blob.sum()}")
for col in ["n_disease_tokens", "n_prompt_tokens", "n_chapters", "seq_len_total"]:
    print(
        f"  {col:18s} ridge med/mean={df[col][ridge].median():.2f}/{df[col][ridge].mean():.2f}"
        f"   rest med/mean={df[col][rest].median():.2f}/{df[col][rest].mean():.2f}"
    )
print(
    f"  zero-disease frac   ridge={np.mean(df.n_disease_tokens[ridge] == 0):.3f}   rest={np.mean(df.n_disease_tokens[rest] == 0):.3f}"
)
print(
    f"  female fraction     ridge={df.is_female[ridge].mean():.3f}   rest={df.is_female[rest].mean():.3f}"
)
print(
    f"  AUC(ridge | -n_disease_tokens) = {auc_score(-df.n_disease_tokens.values, ridge):.3f}"
)
print(f"  PC1-2 var: ridge={ridge_pc:.3f} (near-degenerate) vs blob={blob_pc:.3f}")
zero = df.n_disease_tokens == 0
print(
    f"  risk spread at 0 disease events: p90/p10="
    f"{df.risk[zero].quantile(0.9) / df.risk[zero].quantile(0.1):.0f}x "
    "(low risk is NOT a deterministic function of count)"
)
# -

# +
fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)

# A: mean disease-event count per risk decile (the monotonic curve)
ax = axes[0, 0]
dec = df.groupby("decile")["n_disease_tokens"].mean()
ax.plot(dec.index, dec.values, "o-", color="#34495e")
ax.set_xlabel("risk decile (0 = lowest predicted risk)")
ax.set_ylabel("mean disease events before cutoff")
ax.set_title("Disease-event count rises monotonically with predicted risk")

# B: disease-event distribution, ridge vs rest
ax = axes[0, 1]
ax.boxplot(
    [df.n_disease_tokens[ridge], df.n_disease_tokens[rest]],
    labels=[f"ridge\n(bottom {args.ridge_quantile:.0%})", "rest"],
    showfliers=False,
)
ax.set_ylabel("disease events before cutoff")
ax.set_title(
    f"Ridge median={df.n_disease_tokens[ridge].median():.0f} events "
    f"({np.mean(df.n_disease_tokens[ridge] == 0):.0%} have zero)"
)

# C: the "stalk" — log risk vs disease-event count
ax = axes[1, 0]
hb = ax.hexbin(
    df.n_disease_tokens, df.log_risk, gridsize=50, bins="log", cmap="viridis", mincnt=1
)
fig.colorbar(hb, ax=ax, label="log10(count)")
ax.axhline(np.log(ridge_thr), color="red", ls="--", lw=1, label=f"ridge threshold")
ax.set_xlabel("disease events before cutoff")
ax.set_ylabel("log(total predicted disease hazard)")
ax.set_title("Low risk is a near-empty history (note the vertical stalk at 0 events)")
ax.legend(loc="lower right")

# D: shared-baseline mechanism — log-intensity PCs, stick fans into blob
ax = axes[1, 1]
samp = rng.choice(n, size=min(args.pca_sample, n), replace=False)
lx = np.log(intensities[samp][:, disease_col])
lx = lx - lx.mean(axis=0)
u, s, _ = np.linalg.svd(lx, full_matrices=False)
pcs = u[:, :2] * s[:2]
var = s**2 / (s**2).sum()
sc = ax.scatter(
    pcs[:, 0],
    pcs[:, 1],
    c=df.log_risk.values[samp],
    s=5,
    cmap="viridis",
    rasterized=True,
)
fig.colorbar(sc, ax=ax, label="log risk")
ax.set_xlabel(f"PC1 ({var[0]:.0%} var)")
ax.set_ylabel(f"PC2 ({var[1]:.0%} var)")
ax.set_title(
    f"log-intensity PCs: ridge=tight stick, blob=cloud\n"
    f"within-group PC1-2 var: ridge {ridge_pc:.0%} vs blob {blob_pc:.0%}"
)

fig.suptitle(f"{npz_path.name} — low-risk 'ridge' profile (n={n})")
plt.show()
# -
