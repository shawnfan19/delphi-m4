"""Instant (frozen-history) reliability curves, per sex and 5-year age bin.

The calibration sibling of ``apps/auc-fast-m4.py`` — same single-pass, frozen-
history scoring, but instead of a per-(sex, age-bin) AUC it emits the data for
reliability (calibration) curves. Reproduces the legacy ``plot_calibration``
(gerstung-lab/Delphi ``evaluate_delphi.ipynb``) in the current pipeline:

- CASE score: the model's predicted rate at the input position immediately
  before the disease event (``DiseaseRatesCollator``; ``offset`` rewinds it).
- CONTROL score: one randomly-sampled position per 5-year age bin per
  participant (``AgeStratRatesCollator``).
- Rate -> probability over a window ``W = age_gap`` years (mirrors the legacy
  ``x = 1 - exp(-rate * age_step)``). ``tpp.intensity`` already returns a
  per-year rate (the TPP carries ``time_unit = 365.25``), so no extra scaling.
- Within each (sex, age bin, disease), predicted probabilities are binned on a
  fixed power-law grid; per bin we report the mean predicted probability,
  the observed case fraction, and the count.

Output is a bare JSON logbook consumed directly by ``plot/compare_calibration.py``
(its outer key is generic — here it is the age bracket rather than a horizon):
``logbook[age_bracket][token][sex] = {"pred": [...], "obs": [...], "counts": [...]}``.
"""

# +
import json
import math
import pprint
from dataclasses import asdict, dataclass

import numpy as np
import torch
from cloudpathlib import AnyPath
from tqdm import tqdm

from delphi.env import DELPHI_CKPT_READ, DELPHI_CKPT_WRITE
from delphi.eval import AgeStratRatesCollator, DiseaseRatesCollator
from delphi.experiment import (
    EvalConfig,
    eval_iter,
    load_ckpt,
    move_batch_to_device,
    setup_eval_dataset,
)
from delphi.model.tpp import tpp_dispatch

# Power-law predicted-probability bins (identical to the legacy plot_calibration
# and the deleted apps/forecast.py calibration): 14 bins over (1e-6, ~31].
PROB_BINS = 10.0 ** np.arange(-6.0, 1.5, 0.5)


@dataclass(kw_only=True)
class TaskConfig(EvalConfig):
    fname_prefix = "instant_calibration"
    age_start: int = 40
    age_end: int = 85
    age_gap: int = 5  # also the probability window W (years), mirroring age_step


args = TaskConfig.from_cli()
print("args:")
pprint.pp(args)


# +
model, ckpt_dict = load_ckpt(AnyPath(DELPHI_CKPT_READ) / args.ckpt)
device = "cuda" if torch.cuda.is_available() else "cpu"
reader, ds, val_pids = setup_eval_dataset(
    ckpt_dict,
    fold=args.fold,
    override_biomarkers=args.biomarkers,
    override_expansion_packs=args.expansion_packs,
)
# -

# +
offset_days = args.offset * 365.25
model_targets = model.targets.to(device)
model_targets = model_targets[model_targets != 1]

# Age-bin edges in days; AgeStratRatesCollator makes len(edges) - 1 bins.
age_group_edges = (
    np.arange(args.age_start, args.age_end + args.age_gap, args.age_gap) * 365.25
)
n_bins = len(age_group_edges) - 1

ctl_collator = AgeStratRatesCollator(
    age_groups=torch.from_numpy(age_group_edges).float().to(device)
)
dis_collator = DiseaseRatesCollator(targets=model_targets)

it = tqdm(
    eval_iter(total_size=len(ds), batch_size=args.batch_size),
    total=math.ceil(len(ds) / args.batch_size),
    leave=False,
)
with torch.no_grad():
    for batch_idx in it:
        batch_input = ds.get_batch(batch_idx)
        batch_input = move_batch_to_device(batch_input, device=device)
        x0, t0, _, _, _, x1, t1 = batch_input

        out_dict, _, _ = model(*batch_input[:5])
        tpp = tpp_dispatch(model, out_dict)

        # Intensity at the input position strictly before each target's
        # (t1 - offset); nearest_t0 is that position's age. offset=0 -> the
        # position immediately preceding the event (the "instant" anchor).
        intensity, nearest_t0 = tpp.intensity(t1 - offset_days)
        intensity = intensity.half()

        ctl_collator.step(timesteps=nearest_t0, logits=intensity)
        dis_collator.step(tokens=x1, timesteps=nearest_t0, logits=intensity)

ctl_rates, _ = ctl_collator.finalize()  # (N, n_bins, V)
dis_rates, dis_times = dis_collator.finalize()  # (N, V), (N, V)

ctl_rates = ctl_rates.numpy().astype(np.float32)
dis_rates = dis_rates.numpy().astype(np.float32)
dis_times = dis_times.numpy()
is_female = reader.is_female(val_pids)  # (N,) bool
# -

# +
# Rate (per year) -> probability over a W = age_gap year window, assuming a
# constant rate (mirrors legacy x = 1 - exp(-rate * age_step)). NaNs (non-case
# entries / empty control bins) propagate and are excluded downstream.
window = float(args.age_gap)
with np.errstate(over="ignore", invalid="ignore"):
    prob_dis = 1.0 - np.exp(-dis_rates * window)  # (N, V): case probabilities
# ponytail: control probs computed per age-bin in the loop below, not as one
# (N, n_bins, V) array upfront — that duplicate OOMs on large (e.g. AoU) val.

# Bin each case by the age of its prediction position, matching the control
# binning. is_case marks (participant, token) pairs the participant developed.
dis_time_bin = np.searchsorted(age_group_edges, dis_times, side="right") - 1  # (N, V)
is_case = ~np.isnan(dis_rates)  # (N, V)

age_group_keys = [
    f"{int(start / 365.25)}-{int(end / 365.25)}"
    for start, end in zip(age_group_edges[:-1], age_group_edges[1:])
]


def reliability(p_ctl, p_case):
    """(pred, obs, counts) over PROB_BINS for one (disease, sex, age bin).

    pred = mean predicted probability in bin; obs = observed case fraction
    (#cases / #total) in bin; counts = #participants in bin. Empty bins are
    NaN (pred/obs) / 0 (count). Bins are right-closed (bins[b-1], bins[b]],
    matching the legacy ``(xa > bins[b-1]) & (xa <= bins[b])``.
    """
    p = np.concatenate([p_ctl, p_case])
    y = np.concatenate([np.zeros(len(p_ctl)), np.ones(len(p_case))])
    idx = np.digitize(p, PROB_BINS, right=True)
    pred, obs, counts = [], [], []
    for b in range(1, len(PROB_BINS)):
        m = idx == b
        c = int(m.sum())
        counts.append(c)
        pred.append(float(p[m].mean()) if c else float("nan"))
        obs.append(float(y[m].mean()) if c else float("nan"))
    return pred, obs, counts


targets = model_targets.detach().cpu().numpy().tolist()
logbook = {}
for i, bracket in enumerate(tqdm(age_group_keys, desc="age bins")):
    logbook[bracket] = {}
    ctl_here = (~is_case) & ~np.isnan(ctl_rates[:, i, :])  # (N, V)
    case_here = is_case & (dis_time_bin == i)  # (N, V)
    with np.errstate(invalid="ignore"):
        prob_ctl_i = 1.0 - np.exp(-ctl_rates[:, i, :] * window)  # (N, V)
    for d in targets:
        token = reader.detokenizer[int(d)]
        entry = {}
        for sex_label, is_g in [("female", is_female), ("male", ~is_female)]:
            cmask = ctl_here[:, d] & is_g
            kmask = case_here[:, d] & is_g
            pred, obs, counts = reliability(prob_ctl_i[cmask, d], prob_dis[kmask, d])
            entry[sex_label] = {"pred": pred, "obs": obs, "counts": counts}
        logbook[bracket][token] = entry
# -

# Spot-check: death token (1269) in the first age bracket.
pprint.pp(logbook[age_group_keys[0]].get(reader.detokenizer.get(1269), {}))
print("config:", asdict(args))

out_dir = (AnyPath(DELPHI_CKPT_WRITE) / args.ckpt).parent
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"{args.fname}.json"
# Bare logbook (no config envelope) so plot/compare_calibration.py reads it
# directly — it iterates the top-level keys as the series to overlay.
with out_path.open("w") as f:
    json.dump(logbook, f)
print(f"Saved to {out_path}")
