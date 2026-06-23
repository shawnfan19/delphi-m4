"""Fit a ridge Cox model on extracted hidden states for ONE disease, scored like forecast-m4.

"Cox probing" of the forecasting representation: load the embed.py bundles, fit a
ridge Cox for a single disease on the train-split hidden states (standardized; the
clock re-zeroed at the anchor and prevalent participants dropped), predict a risk
score on the eval split, and grade it with the shared ``windowed_auc`` so the
numbers are directly comparable to forecast-m4's baselines.

One disease per run (parallelize over diseases with a SLURM array). Reads only the
.npz bundles (no checkpoint / model); train_npz and eval_npz are resolved relative
to DELPHI_CKPT_DIR (where embed.py writes them). Each run writes its shard to a
``<fname>/<target>.json`` subdirectory next to the eval bundle (keeping thousands
of shards off the top level); afterwards a single ``merge=true`` run unions them
into ``<fname>.json``:

    # fit one disease (per SLURM array task)
    python apps/cox.py \
        train_npz=<run>/embed_train_recruitment.npz \
        eval_npz=<run>/embed_val_recruitment.npz target=1269 alpha=1.0

    # merge the per-disease files once the array finishes
    python apps/cox.py eval_npz=<run>/embed_val_recruitment.npz merge=true
"""

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from delphi.env import DELPHI_CKPT_DIR
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
    # death censors -> cause-specific risk among survivors. False = "death is
    # complete follow-up" (a death-without-event counts as a control).
    death_as_censor: bool = True
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
ckpt_dir = Path(DELPHI_CKPT_DIR)
eval_path = ckpt_dir / args.eval_npz
out_dir = eval_path.parent
shard_dir = out_dir / args.fname  # per-disease shards live here, off the top level

if args.merge:
    shards = sorted(shard_dir.glob("*.json"))
    if not shards:
        raise SystemExit(f"no shards in {shard_dir}")
    merged = defaultdict(dict)
    for shard in shards:
        try:
            with open(shard) as f:
                logbook = json.load(f)
        except json.JSONDecodeError:
            print(f"[warn] skipping unparsable shard {shard.name}")
            continue
        for horizon, diseases in logbook.items():
            merged[horizon].update(diseases)
    out = out_dir / f"{args.fname}.json"
    with open(out, "w") as f:
        json.dump(dict(merged), f)
    # completeness check: which expected diseases have no shard (failed/unrun jobs)?
    expected = {str(n) for n in np.load(eval_path, allow_pickle=True)["target_names"]}
    found = {d for diseases in merged.values() for d in diseases}
    missing = sorted(expected - found)
    if missing:
        shown = ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else "")
        print(
            f"[warn] {len(missing)}/{len(expected)} diseases missing from "
            f"{out.name} (failed/unrun jobs): {shown}"
        )
    print(f"merged {len(shards)} shards -> {out}")
    raise SystemExit(0)


train = np.load(ckpt_dir / args.train_npz, allow_pickle=True)
ev = np.load(eval_path, allow_pickle=True)
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
    # rare disease with no incident train cases: a Cox can't be fit. Still emit a
    # shard (NaN AUC, real eval counts) so the disease set matches forecast-m4 and
    # a *missing* shard unambiguously means a failed job, not a no-event skip.
    print(f"no train events after anchor for {args.target} ({name}); NaN-AUC shard")
    risk = None
else:
    model = CoxRidge(alpha=args.alpha, ties=args.ties).fit(Xtr[keep], occ, cens)
    risk = model.predict(Xev)  # (N_eval,) linear-predictor risk score

# score on eval against absolute event/censor times, matching forecast-m4
ev_anchor = ev["prompt_age"]
ev_censor = (
    ev["exit_time"]
    if args.death_as_censor
    else np.where(ev["died"], np.inf, ev["exit_time"])
)
ev_age_eval = ev["event_times"][:, col]
# zeros give the real ctl/case counts when unfit; the AUC itself is NaN then.
score = np.zeros(len(ev_age_eval)) if risk is None else risk

logbook = defaultdict(dict)
for horizon in args.horizons:
    t1 = ev_anchor + horizon * 365.25
    for gender_key, is_gender in {
        "female": ev["is_female"],
        "male": ~ev["is_female"],
    }.items():
        n_ctl, n_case, auc = windowed_auc(
            score[is_gender],
            ev_age_eval[is_gender],
            ev_censor[is_gender],
            (ev_anchor[is_gender], t1[is_gender]),
        )
        logbook[horizon].setdefault(name, {})[gender_key] = {
            "auc": float("nan") if risk is None else float(auc),
            "ctl_count": int(n_ctl),
            "dis_count": int(n_case),
        }

shard_dir.mkdir(parents=True, exist_ok=True)
path = shard_dir / f"{args.target}.json"
tmp = path.with_name(
    path.name + ".tmp"
)  # atomic write: a killed job leaves no half-shard
with open(tmp, "w") as f:
    json.dump(dict(logbook), f)
os.replace(tmp, path)
print(f"{args.target} ({name}): {n_event} train events -> {path}")
