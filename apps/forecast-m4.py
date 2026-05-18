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
from delphi.data.ukb import (
    Biomarker,
    ExpansionPack,
    MultimodalUKBReader,
    filter_participants_with_biomarkers,
)
from delphi.env import DELPHI_CKPT_READ, DELPHI_CKPT_WRITE
from delphi.eval.auc import batched_mann_whitney_auc
from delphi.eval.survival import NelsonAalenEstimator
from delphi.experiment import GenerateConfig, eval_iter, load_ckpt, move_batch_to_device
from delphi.model.tpp import HomoPoissonTPP
from delphi.model.transformer import generate
from delphi.model.utils import self_terminate
from delphi.multimodal import Modality, parse_panel


def first_modality_timestep(pids, biomarkers, expansion_packs):

    cutoff = np.full(len(pids), -np.inf, dtype=np.float32)
    for mod_name in biomarkers or []:
        first = Biomarker.first_occurrence_times(mod_name, pids)
        cutoff = np.fmin(cutoff, first)
    for pack_name in expansion_packs or []:
        first = ExpansionPack.first_occurrence_times(pack_name, pids)
        cutoff = np.fmin(cutoff, first)

    return cutoff


@dataclass(kw_only=True)
class TaskConfig(GenerateConfig):
    panel: None | str = None
    biomarkers: None | list = None
    expansion_packs: None | list = None
    horizons: list = field(default_factory=lambda: [1, 3, 5, 10])
    method: str = "hazards"  # "hazards", "sampling_hazards", "nelson_aalen"
    fname: None | str = None

    def __post_init__(self):

        assert args.method in {"hazards", "sampling_hazards", "nelson_aalen"}

        if self.panel:
            self.biomarkers, self.expansion_packs, self.panel_name = parse_panel(
                self.panel
            )
        if not self.fname:
            if args.method == "hazards":
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


val_pids = MultimodalUKBReader.participants("val")
total_val = val_pids.size
val_pids = filter_participants_with_biomarkers(
    val_pids, biomarkers=args.biomarkers, any=False
)
print(f"{val_pids.size} / {total_val} val pids (biomarker filter)")

prompt_age = first_modality_timestep(
    pids=val_pids, biomarkers=args.biomarkers, expansion_packs=args.expansion_packs
)
prompt_transform = MultimodalPrompt(
    prompt_age={pid: age for pid, age in zip(val_pids, prompt_age.tolist())},
    append_no_event=args.method != "hazards",
)


reader_args = ckpt_dict["reader_args"]
token_transform_args = ckpt_dict["token_transform_args"]
biomarker_transform_args = ckpt_dict.get("biomarker_transform_args")
biomarker_stats = ckpt_dict.get("biomarker_stats")
pprint.pp(
    {
        "reader_args": reader_args,
        "token_transform_args": token_transform_args,
        "biomarker_transform_args": biomarker_transform_args,
    }
)

reader = MultimodalUKBReader(**reader_args)
token_transform = TokenTransform(**token_transform_args)
if biomarker_transform_args is not None:
    mean = biomarker_stats["mean"] if biomarker_stats else None
    std = biomarker_stats["std"] if biomarker_stats else None
    if mean is not None:
        mean = {Modality[k.upper()]: v for k, v in mean.items()}
    if std is not None:
        std = {Modality[k.upper()]: v for k, v in std.items()}
    biomarker_transform = BiomarkerTransform(
        **biomarker_transform_args, mean=mean, std=std
    )
    biomarker_transform.describe()
else:
    biomarker_transform = None


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

        logits, termin_mask = self_terminate(
            outputs["idx"],
            outputs["logits"],
            terminate_except=torch.tensor(model.config.self_terminate_except).to(
                idx.device
            ),
        )

        if args.method == "sampling_hazards":
            tpp = TPP(timesteps=outputs["age"], logits=logits)
            t0_col = t0.unsqueeze(1)
            for horizon in predictor.keys():
                cap_lambda, _ = tpp.integral(t0=t0_col, t1=t0_col + horizon * 365.25)
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
                estimator = NelsonAalenEstimator(
                    timesteps=outputs["age"][cut],
                    intensities=torch.exp(logits)[cut],
                    at_risk=~termin_mask.bool()[cut],
                )
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
