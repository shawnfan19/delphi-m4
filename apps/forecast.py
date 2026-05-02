# +
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from delphi.data.dataset import Dataset
from delphi.data.transform import Prompt, TokenTransform
from delphi.data.ukb import UKBReader
from delphi.env import DELPHI_CKPT_READ, DELPHI_CKPT_WRITE
from delphi.eval.auc import batched_mann_whitney_auc
from delphi.eval.survival import KaplanMeierEstimator, NelsonAalenEstimator
from delphi.experiment import (
    CliConfig,
    eval_iter,
    load_ckpt,
    move_batch_to_device,
)
from delphi.model.tpp import HomoPoissonTPP, tpp_dispatch
from delphi.model.transformer import generate

# -


@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    ckpt: str
    prompt_age: None | int = None
    prompt_no_event: bool = False
    batch_size: int = 128
    n_repeats: int = 1
    horizons: list = field(default_factory=lambda: [1, 3, 5, 10])
    method: str = "hazards"  # "hazards", "sampling_hazards", "nelson_aalen"
    n_grid: int = 20
    suffix: None | str = None
    calibrate_only: bool = False

    def __post_init__(self):

        assert self.method in {"hazards", "sampling_hazards"}

        if not self.suffix:
            self.suffix = self.method
            if self.method != "hazards":
                self.suffix += f"_n{self.n_repeats}"

            if self.prompt_age:
                self.suffix += f"_prompt{self.prompt_age}"


args = TaskConfig.from_cli()
args.print()

ckpt = Path(DELPHI_CKPT_READ) / args.ckpt
model, ckpt_dict = load_ckpt(ckpt)
device = "cuda" if torch.cuda.is_available() else "cpu"
assert model.config.block_size is None

token_transform_args = ckpt_dict["token_transform_args"]
val_pids = UKBReader.participants("val")
# -

reader = UKBReader()
token_transform = TokenTransform(**token_transform_args)
token_transform.describe()

if args.prompt_age:
    prompt_age = np.full(val_pids.shape, fill_value=args.prompt_age * 365.25)
    prompt_age = prompt_age.astype(float)
else:
    recruitment_times = reader.recruitment_times(val_pids)
    val_pids = val_pids[~np.isnan(recruitment_times)]
    prompt_age = recruitment_times[~np.isnan(recruitment_times)]
exit_times = reader.exit_times(val_pids)
val_pids = val_pids[exit_times > prompt_age]
prompt_age = prompt_age[exit_times > prompt_age]
prompt_age_dict = dict(zip(val_pids, prompt_age))
prompt_transform = Prompt(
    prompt_age=prompt_age_dict, append_no_event=args.prompt_no_event
)

ds = Dataset(
    reader=reader,
    pids=val_pids,
    token_transform=token_transform,
    prompt_transform=prompt_transform,
)


# +
if args.method == "hazards":
    it = eval_iter(total_size=len(ds), batch_size=args.batch_size)
    pbar = tqdm(it, total=math.ceil(len(ds) / args.batch_size))
    predictor_tally = defaultdict(list)
    for batch_idx in pbar:

        pmt_idx, pmt_age, X1, T1 = move_batch_to_device(
            ds.get_batch(batch_idx), device=device
        )

        with torch.no_grad():
            outputs, _, _ = model(pmt_idx, pmt_age)

        tpp = tpp_dispatch(model, outputs, device, time_unit=model.config.time_unit)

        t0 = torch.tensor(prompt_age[batch_idx], device=device).to(torch.float32)
        for horizon in args.horizons:
            hazards, _ = tpp.integral(
                t0=t0, t1=t0 + horizon * 365.25, n_grid=args.n_grid
            )
            predictor_tally[horizon].append(hazards.detach().cpu().numpy())

    predictor = dict()
    for horizon in predictor_tally.keys():
        predictor[horizon] = np.concatenate(predictor_tally[horizon], axis=0)
else:
    predictor_tally = {horizon: list() for horizon in args.horizons}
    horizons_tensor = torch.tensor(args.horizons, device=device) * 365.25

    assert args.batch_size % args.n_repeats == 0
    eff_batch_size = int(args.batch_size / args.n_repeats)
    it = eval_iter(total_size=len(ds), batch_size=eff_batch_size)
    pbar = tqdm(it, total=math.ceil(len(ds) / eff_batch_size))
    for batch_idx in pbar:

        batch_idx = np.repeat(batch_idx, args.n_repeats)
        pmt_idx, pmt_age, X1, T1 = move_batch_to_device(
            ds.get_batch(batch_idx), device=device
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
        )
        pbar.set_postfix(
            {"n_gen": stats["n_gen"].mean(), "n_pmt": stats["n_prompt"].mean()}
        )

        with torch.no_grad():
            out_dict, _, _ = model(idx, age)

        assert model.config.loss == "homo_poisson"

        n_cuts = len(batch_idx) // args.n_repeats
        batch_predictor = {horizon: list() for horizon in args.horizons}
        for i in range(n_cuts):
            cut = slice(i * args.n_repeats, (i + 1) * args.n_repeats)
            pid_idx = batch_idx[i * args.n_repeats]
            tpp = HomoPoissonTPP(
                logits=out_dict["logits"][cut],
                tokens=out_dict["idx"][cut],
                timesteps=out_dict["age"][cut],
                terminate_except=torch.tensor(
                    model.config.self_terminate_except, device=device
                ),
                time_unit=model.config.time_unit,
            )
            estimator = NelsonAalenEstimator(tpp=tpp)
            hazard_at_pmt = estimator(prompt_age[pid_idx])
            target_times = prompt_age[pid_idx] + horizons_tensor
            hazard_at_horizon = estimator(target_times)
            diff_hazard = hazard_at_horizon - hazard_at_pmt.unsqueeze(0)
            probs = 1.0 - torch.exp(-diff_hazard)
            probs = probs.detach().cpu().numpy()
            for h_idx, horizon in enumerate(args.horizons):
                batch_predictor[horizon].append(probs[h_idx])

        for horizon in args.horizons:
            predictor_tally[horizon].append(np.stack(batch_predictor[horizon], axis=0))

    predictor = dict()
    for horizon in predictor_tally.keys():
        predictor[horizon] = np.concatenate(predictor_tally[horizon], axis=0)


is_female = reader.is_female(pids=val_pids)
event_timesteps = reader.event_times(pids=val_pids)
exit_timesteps = reader.exit_times(pids=val_pids)
died = ~np.isnan(event_timesteps[:, 1269])

if not args.calibrate_only:
    logbook = defaultdict(dict)
    for horizon in predictor.keys():

        window_end = prompt_age + horizon * 365.25
        incomplete = (exit_timesteps < window_end) & ~died

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

    path = (
        Path(DELPHI_CKPT_WRITE)
        / Path(args.ckpt).parent
        / f"forecast_{args.suffix}.json"
    )
    with open(path, "w") as f:
        json.dump(logbook, f)


calibrate_logbook = defaultdict(dict)
bins = 10 ** np.arange(-6.0, 1.5, 0.5)
for horizon in tqdm(predictor.keys(), leave=False):
    for i in model.targets.detach().cpu().numpy():
        for gender_key, is_gender in {"female": is_female, "male": ~is_female}.items():

            prob_hat = 1 - np.exp(-predictor[horizon][:, i])
            bin_masks = [
                np.logical_and(prob_hat > bins[b - 1], prob_hat <= bins[b])
                for b in range(1, len(bins))
            ]
            bin_masks = [np.logical_and(bin_mask, is_gender) for bin_mask in bin_masks]

            bin_counts = [int(bin_mask.sum()) for bin_mask in bin_masks]
            avg_pred = [float(np.nanmean(prob_hat[bin_mask])) for bin_mask in bin_masks]
            avg_true = list()
            for bin_mask in bin_masks:

                disease_timesteps = event_timesteps[bin_mask, i]
                occur = ~np.isnan(disease_timesteps)
                surv_timesteps = np.where(
                    occur, disease_timesteps, exit_timesteps[bin_mask]
                )
                prompt_timesteps = prompt_age[bin_mask]
                km_estimator = KaplanMeierEstimator(
                    surv_timesteps=surv_timesteps - prompt_timesteps, occur=occur
                )
                avg_true.append(
                    float(km_estimator.incidence(start_age=0, end_age=horizon * 365.25))
                )

            calibrate_logbook[horizon].setdefault(reader.detokenizer[i], {})[
                gender_key
            ] = {"pred": avg_pred, "obs": avg_true, "counts": bin_counts}

print(calibrate_logbook[args.horizons[0]][reader.detokenizer[1269]])
path = (
    Path(DELPHI_CKPT_WRITE)
    / Path(args.ckpt).parent
    / f"forecast_calibrate_{args.suffix}.json"
)
with open(path, "w") as f:
    json.dump(calibrate_logbook, f)
