"""Forecast AUC evaluation with a choice of predictor.

Predictors (``--method``):
- ``hazards``: one forward pass on the prompt, use the last prompt position's
  logits as the per-horizon risk score. Valid when intensity is time-homogeneous
  between events (∫ λ dτ = λ · H with H shared across participants → λ alone
  ranks the same as the integral). For a time-inhomogeneous model, reintroduce
  a probe grid over [t0, t0 + H] and integrate per horizon.
- ``sampling_hazards``: sample trajectories and integrate λ dt over
  [t0, t0 + H] per horizon using the TPP module.
- ``nelson_aalen``: sample trajectories and use a Nelson-Aalen estimator to
  approximate the cumulative hazard, then 1 - exp(-ΔH) as the per-horizon risk.
"""

import json
import math
import pprint
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from delphi.data import MultimodalDataset
from delphi.data.transform import BiomarkerTransform, MultimodalPrompt, TokenTransform
from delphi.data.ukb import MultimodalUKBReader
from delphi.env import DELPHI_CKPT_READ, DELPHI_CKPT_WRITE
from delphi.eval.auc import batched_mann_whitney_auc
from delphi.eval.survival import NelsonAalenEstimator
from delphi.experiment import GenerateConfig, eval_iter, load_ckpt, move_batch_to_device
from delphi.model.tpp import tpp_dispatch
from delphi.model.transformer import generate


@dataclass(kw_only=True)
class TaskConfig(GenerateConfig):
    horizons: list = field(default_factory=lambda: [1, 3, 5, 10])
    method: str = "hazards"  # "hazards", "sampling_hazards", "nelson_aalen"
    prompt_age: int | str = "recruitment"  # fixed anchor age (years), or "recruitment"
    fname: None | str = None

    def __post_init__(self):

        assert self.method in {"hazards", "sampling_hazards", "nelson_aalen"}
        if self.prompt_age != "recruitment":
            self.prompt_age = int(self.prompt_age)  # fixed anchor age in years

        if not self.fname:
            if self.method == "hazards":
                suffix = "hazards"
            else:
                suffix = f"{self.method}_n{self.n_repeats}"
            self.fname = f"forecast_{suffix}"


args = TaskConfig.from_cli()
args.print()

ckpt = Path(DELPHI_CKPT_READ) / args.ckpt
model, ckpt_dict = load_ckpt(ckpt)
device = "cuda" if torch.cuda.is_available() else "cpu"
assert model.config.block_size is None


reader_args = ckpt_dict["reader_args"]
pprint.pp(
    {
        "reader_args": reader_args,
    }
)

# pass dict (not list) so reader uses the checkpoint's index assignments
# instead of re-deriving them from sorted order
reader = MultimodalUKBReader(
    biomarkers=model.config.biomarker2idx or None,
    expansion_packs=reader_args["expansion_packs"],
)

val_pids = MultimodalUKBReader.participants("val")
total_val = val_pids.size

# anchor: recruitment age (when blood biomarkers were measured) or a fixed age.
# recruitment_times yields NaN where undefined -> drop those participants.
if args.prompt_age == "recruitment":
    prompt_age = reader.recruitment_times(val_pids)
else:
    prompt_age = np.full(len(val_pids), int(args.prompt_age) * 365.25, dtype=np.float32)
keep = ~np.isnan(prompt_age)
val_pids, prompt_age = val_pids[keep], prompt_age[keep]
print(f"{val_pids.size} / {total_val} val pids (valid anchor)")
token_transform = TokenTransform.from_ckpt(ckpt_dict)
token_transform.describe()
biomarker_transform = BiomarkerTransform.from_ckpt(ckpt_dict)
if biomarker_transform is not None:
    biomarker_transform = biomarker_transform.replace(dropout=None)
    biomarker_transform.describe()

prompt_transform = MultimodalPrompt(
    prompt_age={pid: age for pid, age in zip(val_pids, prompt_age.tolist())},
    biomarker2idx=reader.biomarker2idx,
    append_no_event=args.method != "hazards",
)


ds = MultimodalDataset(
    reader=reader,
    pids=val_pids,
    token_transform=token_transform,
    biomarker_transform=biomarker_transform,
    prompt_transform=prompt_transform,
)


if args.method == "hazards":
    predictor = list()
    it = eval_iter(total_size=len(ds), batch_size=args.batch_size)
    pbar = tqdm(it, total=math.ceil(len(ds) / args.batch_size))
    for batch_idx in pbar:

        pmt_idx, pmt_age, pmt_bio_x_dict, pmt_bio_t, pmt_bio_m, X1, T1 = (
            move_batch_to_device(ds.get_batch(batch_idx), device=device)
        )

        with torch.no_grad():
            outputs, _, _ = model(
                pmt_idx,
                pmt_age,
                biomarker=pmt_bio_x_dict,
                mod_age=pmt_bio_t,
                mod_idx=pmt_bio_m,
            )

        predictor.append(outputs["logits"][:, -1, :].detach().cpu().numpy())

    predictor = np.concatenate(predictor, axis=0)
    predictor = {horizon: predictor for horizon in args.horizons}

else:
    predictor = {horizon: list() for horizon in args.horizons}
    horizons_tensor = torch.tensor(args.horizons, device=device) * 365.25

    # sampling_hazards / nelson_aalen are implemented for the homogeneous-Poisson
    # head only (HomoPoissonTPP.integral + sliceable TPP); fail fast otherwise.
    assert model.config.loss == "homo_poisson"
    assert args.batch_size % args.n_repeats == 0
    eff_batch_size = int(args.batch_size / args.n_repeats)
    it = eval_iter(total_size=len(ds), batch_size=eff_batch_size)
    pbar = tqdm(it, total=math.ceil(len(ds) / eff_batch_size))
    for batch_idx in pbar:

        batch_idx = np.repeat(batch_idx, args.n_repeats)
        pmt_idx, pmt_age, pmt_bio_x_dict, pmt_bio_t, pmt_bio_m, X1, T1 = (
            move_batch_to_device(ds.get_batch(batch_idx), device=device)
        )
        t0 = torch.as_tensor(
            prompt_age[batch_idx], dtype=pmt_age.dtype, device=pmt_age.device
        )

        idx, age, stats = generate(
            model=model,
            idx=pmt_idx,
            age=pmt_age,
            max_age=t0 + max(args.horizons) * 365.25,
            stop_at_block_size=False,
            termination_tokens=[1269],
            cached=True,
            biomarker=pmt_bio_x_dict,
            mod_age=pmt_bio_t,
            mod_idx=pmt_bio_m,
        )
        pbar.set_postfix(
            {"n_gen": stats["n_gen"].mean(), "n_pmt": stats["n_prompt"].mean()}
        )

        with torch.no_grad():
            outputs, _, _ = model(
                idx,
                age,
                biomarker=pmt_bio_x_dict,
                mod_age=pmt_bio_t,
                mod_idx=pmt_bio_m,
            )

        tpp = tpp_dispatch(model, outputs)
        if args.method == "sampling_hazards":
            for horizon in predictor.keys():
                cap_lambda, _ = tpp.integral(t0=t0, t1=t0 + horizon * 365.25)
                cap_lambda = torch.reshape(
                    cap_lambda, shape=(-1, args.n_repeats, cap_lambda.shape[-1])
                ).mean(dim=1)
                predictor[horizon].append(cap_lambda.detach().cpu().numpy())

        else:  # nelson_aalen
            n_cuts = len(batch_idx) // args.n_repeats
            batch_predictor = {horizon: list() for horizon in args.horizons}
            for i in range(n_cuts):
                cut = slice(i * args.n_repeats, (i + 1) * args.n_repeats)
                pid_idx = batch_idx[i * args.n_repeats]
                estimator = NelsonAalenEstimator(tpp[cut])
                hazard_at_pmt = estimator(prompt_age[pid_idx])
                target_times = prompt_age[pid_idx] + horizons_tensor
                hazard_at_horizon = estimator(target_times)
                diff_hazard = hazard_at_horizon - hazard_at_pmt.unsqueeze(0)
                probs = 1.0 - torch.exp(-diff_hazard)
                probs = probs.detach().cpu().numpy()
                for h_idx, horizon in enumerate(args.horizons):
                    batch_predictor[horizon].append(probs[h_idx])

            for horizon in args.horizons:
                predictor[horizon].append(np.stack(batch_predictor[horizon], axis=0))

    for horizon in predictor.keys():
        predictor[horizon] = np.concatenate(predictor[horizon], axis=0)


is_female = reader.is_female(pids=val_pids)
event_timesteps = reader.event_times(pids=val_pids)
exit_time = reader.exit_times(pids=val_pids)
died = ~np.isnan(event_timesteps[:, 1269])

logbook = defaultdict(dict)
for horizon in predictor.keys():

    window_end = prompt_age + horizon * 365.25
    incomplete = (exit_time < window_end) & ~died

    positive = (event_timesteps >= prompt_age[:, None]) & (
        event_timesteps < window_end[:, None]
    )
    prevalent = event_timesteps < prompt_age[:, None]

    for gender_key, is_gender in {"female": is_female, "male": ~is_female}.items():

        gender_col = is_gender[:, None]
        ctl_mask = gender_col & ~positive & ~prevalent & ~incomplete[:, None]
        case_mask = gender_col & positive

        n_ctl, n_case, auc = batched_mann_whitney_auc(
            predictor[horizon], ctl_mask, case_mask
        )

        for dis_token in model.targets.detach().cpu().numpy():
            logbook[horizon].setdefault(reader.detokenizer[dis_token], {})[
                gender_key
            ] = {
                "auc": float(auc[dis_token]),
                "ctl_count": int(n_ctl[dis_token]),
                "dis_count": int(n_case[dis_token]),
            }

print(logbook[args.horizons[0]][reader.detokenizer[1269]])

path = Path(DELPHI_CKPT_WRITE) / Path(args.ckpt).parent / f"{args.fname}.json"
with open(path, "w") as f:
    json.dump(logbook, f)
