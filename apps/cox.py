"""Fit a ridge Cox model on extracted hidden states for ONE disease, scored like forecast-m4.

"Cox probing" of the forecasting representation: load the embed.py bundles, fit a
ridge Cox for a single disease on the train-split hidden states (standardized; the
clock re-zeroed at the anchor and prevalent participants dropped), predict a risk
score on the eval split, and grade it with the shared ``windowed_auc`` so the
numbers are directly comparable to forecast-m4's baselines.

One disease per run (parallelize over diseases with a SLURM array). Reads only the
.npz bundles (no checkpoint / model). Each run writes ``<fname>.<target>.json``
next to the eval bundle; afterwards a single ``merge=true`` run unions the
per-disease files into ``<fname>.json``:

    # fit one disease (per SLURM array task)
    python apps/cox.py \
        train_npz=.../embed_train_recruitment.npz \
        eval_npz=.../embed_val_recruitment.npz target=1269 alpha=1.0

    # merge the per-disease files once the array finishes
    python apps/cox.py eval_npz=.../embed_val_recruitment.npz merge=true
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from delphi.eval.auc import windowed_auc
from delphi.eval.cox import CoxRidge
from delphi.experiment import CliConfig

EPS = 0.5  # day floor so an event/censor exactly at the anchor has positive time


@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    train_npz: str = ""
    eval_npz: str = ""
    target: int = -1  # disease token to fit (one per run; the SLURM array fans out)
    horizons: list = field(default_factory=lambda: [1, 3, 5, 10])
    alpha: float = 1.0
    ties: str = "efron"
    fname: str = "cox"
    merge: bool = (
        False  # if set: union the per-disease files into <fname>.json and exit
    )

    def __post_init__(self):
        assert self.eval_npz, "eval_npz is required (it locates the output directory)"
        if not self.merge:
            assert self.train_npz, "train_npz is required for fitting"
            assert self.target >= 0, "target (a disease token) is required for fitting"


args = TaskConfig.from_cli()
args.print()
out_dir = Path(args.eval_npz).parent

if args.merge:
    shards = sorted(
        out_dir.glob(f"{args.fname}.*.json")
    )  # excludes merged <fname>.json
    if not shards:
        raise SystemExit(f"no shards matching {args.fname}.*.json in {out_dir}")
    merged = defaultdict(dict)
    for shard in shards:
        with open(shard) as f:
            logbook = json.load(f)
        for horizon, diseases in logbook.items():
            merged[horizon].update(diseases)
    out = out_dir / f"{args.fname}.json"
    with open(out, "w") as f:
        json.dump(dict(merged), f)
    print(f"merged {len(shards)} shards -> {out}")
    raise SystemExit(0)


train = np.load(args.train_npz, allow_pickle=True)
ev = np.load(args.eval_npz, allow_pickle=True)
assert np.array_equal(
    train["target_tokens"], ev["target_tokens"]
), "train/eval bundles describe different disease columns"
target_tokens = ev["target_tokens"]
if not (target_tokens == args.target).any():
    raise SystemExit(f"target {args.target} not in bundle target_tokens")
col = int(np.where(target_tokens == args.target)[0][0])
name = str(ev["target_names"][col])

# standardize hidden states: scaler fit on train, applied to train + eval
mu = train["h"].mean(axis=0)
sd = train["h"].std(axis=0)
sd[sd == 0] = 1.0
Xtr = (train["h"] - mu) / sd
Xev = (ev["h"] - mu) / sd

# train labels, clock re-zeroed at the anchor (sksurv has no delayed entry);
# drop prevalent participants (diseased before the anchor).
ev_age = train["event_times"][:, col]
anchor = train["prompt_age"]
keep = ~(~np.isnan(ev_age) & (ev_age < anchor))
occ = np.maximum(ev_age[keep] - anchor[keep], EPS)  # NaN (no event) stays NaN
cens = np.maximum(train["exit_time"][keep] - anchor[keep], EPS)
n_event = int(np.isfinite(occ).sum())
if n_event == 0:
    print(f"no events after anchor for {args.target} ({name}) in train; skipping")
    raise SystemExit(0)

model = CoxRidge(alpha=args.alpha, ties=args.ties).fit(Xtr[keep], occ, cens)
risk = model.predict(Xev)  # (N_eval,) linear-predictor risk score

# score on eval against absolute event/censor times, matching forecast-m4
ev_anchor = ev["prompt_age"]
ev_censor = np.where(ev["died"], np.inf, ev["exit_time"])  # death = complete follow-up
ev_age_eval = ev["event_times"][:, col]

logbook = defaultdict(dict)
for horizon in args.horizons:
    t1 = ev_anchor + horizon * 365.25
    for gender_key, is_gender in {
        "female": ev["is_female"],
        "male": ~ev["is_female"],
    }.items():
        n_ctl, n_case, auc = windowed_auc(
            risk[is_gender],
            ev_age_eval[is_gender],
            ev_censor[is_gender],
            (ev_anchor[is_gender], t1[is_gender]),
        )
        logbook[horizon].setdefault(name, {})[gender_key] = {
            "auc": float(auc),
            "ctl_count": int(n_ctl),
            "dis_count": int(n_case),
        }

path = out_dir / f"{args.fname}.{args.target}.json"
with open(path, "w") as f:
    json.dump(dict(logbook), f)
print(f"{args.target} ({name}): {n_event} train events -> {path}")
