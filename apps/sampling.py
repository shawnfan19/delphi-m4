# +
import math
import pprint
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from delphi.data.ukb import UKBDataset, cut_prompt
from delphi.env import DELPHI_CKPT_DIR
from delphi.eval import (
    EventTimeCollator,
    IntervalKaplanMeierCollator,
    SamplingProbCollator,
    mann_whitney_auc,
)
from delphi.experiment import GenerateConfig, eval_iter, load_ckpt
from delphi.model.transformer import generate
from delphi.model.utils import self_terminate

# -


@dataclass
class TaskConfig(GenerateConfig):
    prompt_age: int = 60
    prompt_no_event: bool = True

    time_horizon: list[int] = field(default_factory=lambda: [1, 3, 5, 10, 15])
    km_interval: float = 365.25
    km_start: int = 0
    km_end: int = 81


args = TaskConfig.auto(
    ckpt="cluster/2026-02-02-115354/ckpt.pt",
    batch_size=256,
    n_repeats=1,
)
pprint.pp(args)

time_horizon = np.array(args.time_horizon) * 365.25
start_age = args.prompt_age * 365.25
time_intervals = np.arange(args.km_start, args.km_end * 365.25, args.km_interval)
print(start_age)
print(time_horizon)
print(time_intervals)

model, ckpt_dict = load_ckpt(Path(DELPHI_CKPT_DIR) / args.ckpt)

device = "cuda" if torch.cuda.is_available() else "cpu"
data_args = ckpt_dict["data_args"].copy()
data_args["subject_list"] = "participants/val_fold.bin"
data_args["perturb"] = False
data_args["deterministic"] = True
data_args["additional_dx_token"] = ckpt_dict["data_args"].get(
    "additional_dx_token", False
)
pprint.pp(data_args)

ds = UKBDataset(**data_args)
ds.subset_participants_for_prompt(prompt_age=start_age, prompt_tokens=None)


# +
event_time_collator = EventTimeCollator(vocab_size=ds.vocab_size)
risk_collator = IntervalKaplanMeierCollator(
    start_age=start_age,
    time_horizon=time_horizon,
    time_intervals=time_intervals,
    n_repeats=args.n_repeats,
)
prob_collator = SamplingProbCollator(
    vocab_size=ds.vocab_size,
    time_horizon=time_horizon,
    start_age=start_age,
    n_repeats=args.n_repeats,
)

assert args.batch_size % args.n_repeats == 0
eff_batch_size = int(args.batch_size / args.n_repeats)
it = eval_iter(total_size=len(ds), batch_size=eff_batch_size)
pbar = tqdm(it, total=math.ceil(len(ds) / eff_batch_size))

logits_at_prompt = list()
for batch_idx in pbar:

    X0, T0, X1, T1 = ds.get_batch(batch_idx)
    event_time_collator.step(X1, T1)
    pmt_idx, pmt_age, _ = cut_prompt(
        X0,
        T0,
        prompt_age=start_age,
        prompt_token=None,
        append_no_event=args.prompt_no_event,
    )
    pmt_idx, pmt_age = pmt_idx.to(device), pmt_age.to(device)

    output, _, _ = model.forward(pmt_idx, pmt_age)
    logits_at_prompt.append(output["logits"][:, -1, :].detach().cpu().numpy())

    pmt_idx = torch.repeat_interleave(pmt_idx, args.n_repeats, dim=0)
    pmt_age = torch.repeat_interleave(pmt_age, args.n_repeats, dim=0)
    max_age = 85 * 365.25
    tokens, timestep, gen_stats = generate(
        model=model,
        idx=pmt_idx,
        age=pmt_age,
        max_age=max_age,
        max_new_tokens=args.max_new_tokens,
        termination_tokens=[1269],
        stop_at_block_size=args.stop_at_block_size,
    )

    output, _, _ = model.forward(tokens, timestep)
    logits = output["logits"]
    logits = self_terminate(
        tokens,
        logits,
        terminate_except=torch.tensor(model.config.self_terminate_except).to(
            tokens.device
        ),
    )

    n_gen = gen_stats["n_gen"].mean() - gen_stats["n_prompt"].mean()
    pbar.set_postfix({"n_gen": n_gen})

    risk_collator.step(tokens, timestep, logits)
    prob_collator.step(tokens, timestep)

occur_time, exit_time = event_time_collator.finalize()
prob_by_horizon = prob_collator.finalize()
km_risk = risk_collator.finalize()
# -


logits_at_prompt = np.concatenate(logits_at_prompt, axis=0)
base_incidence = dict()
for horizon in time_horizon:
    base_incidence[horizon] = 1 - np.exp(-np.exp(logits_at_prompt) * horizon)

occur = dict()
for horizon in time_horizon:

    end = start_age + horizon

    _occur = np.zeros((len(ds), ds.vocab_size))
    _occur[np.logical_and(occur_time > start_age, occur_time <= end)] = 1
    _occur[occur_time <= start_age] = float("nan")

    ended_early = exit_time < end
    ended_early = ended_early[:, None]
    _occur[np.logical_and(ended_early, _occur == 0)] = float("nan")

    occur[horizon] = _occur

# +
calc_aucs, base_aucs, prob_aucs = (
    defaultdict(list),
    defaultdict(list),
    defaultdict(list),
)
ctl_ct, dis_ct = defaultdict(list), defaultdict(list)

for i, horizon in enumerate(time_horizon):

    end = start_age + horizon

    for token in tqdm(range(ds.vocab_size)):
        is_ctl = occur[horizon][:, token] == 0
        is_dis = occur[horizon][:, token] == 1
        ctl_ct[horizon].append(is_ctl.sum())
        dis_ct[horizon].append(is_dis.sum())

        ctl_logits = logits_at_prompt[is_ctl, token]
        dis_logits = logits_at_prompt[is_dis, token]
        base_aucs[horizon].append(mann_whitney_auc(ctl_logits, dis_logits))

        ctl_incid = km_risk[horizon][is_ctl, token]
        dis_incid = km_risk[horizon][is_dis, token]
        calc_aucs[horizon].append(mann_whitney_auc(ctl_incid, dis_incid))

        ctl_prob = prob_by_horizon[horizon][is_ctl, token]
        dis_prob = prob_by_horizon[horizon][is_dis, token]
        prob_aucs[horizon].append(mann_whitney_auc(ctl_prob, dis_prob))


# +
ckpt = Path(DELPHI_CKPT_DIR) / args.ckpt
odir = ckpt.parent


def remove_nan_for_violin(aucs: dict):
    nan_free_aucs = list()
    for key, vals in aucs.items():
        vals = np.array(vals)
        notna = ~np.isnan(vals)
        if notna.sum() > 0:
            nan_free_aucs.append(vals[~np.isnan(vals)])
    assert len(nan_free_aucs) > 0
    return nan_free_aucs


fig, ax = plt.subplots()
parts = ax.violinplot(remove_nan_for_violin(base_aucs))
parts = ax.violinplot(remove_nan_for_violin(calc_aucs))
parts = ax.violinplot(remove_nan_for_violin(prob_aucs))
ax.set_xticks(np.arange(len(base_aucs)) + 1, time_horizon / 365.25)
ax.set_xlabel("time horizon of prediction (year)")
ax.set_title(f"{args.n_repeats} repetitions")
ax.set_ylabel("Mann-Whitney AUC")
plt.savefig(odir / f"sampling_auc_nreps{args.n_repeats}.png", dpi=300)


# +
def binned_average(
    pred: np.ndarray,
    obs: np.ndarray,
    bins: np.ndarray = 10 ** np.arange(-6.0, 1.5, 0.5),
):

    bin_masks = [
        np.logical_and(pred > bins[b - 1], pred <= bins[b]) for b in range(1, len(bins))
    ]
    avg_pred = [np.nanmean(pred[bin_mask]) for bin_mask in bin_masks]
    avg_obs = [np.nanmean(obs[bin_mask]) for bin_mask in bin_masks]

    return avg_pred, avg_obs, np.nanmean(pred), np.nanmean(obs)


def plot_calibration(
    doi: list[int],
    doi_titles: list[str],
    time_horizon: np.ndarray,
    y_hat_dict: dict[str, np.ndarray],
    y_dict: dict[str, np.ndarray],
    title: str,
):

    colors = plt.cm.tab20(np.linspace(0, 1, len(time_horizon)))
    n_cols = 5
    n_rows = math.ceil(len(doi) / n_cols)
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(20, 4 * n_rows), sharex=True, sharey=True
    )
    axs = axs.ravel()

    for i, token in enumerate(doi):
        axs[i].plot([0, 1], [0, 1], color="k")
        axs[i].set_title(doi_titles[i], fontsize=8)
        for j, horizon in enumerate(time_horizon):
            pred, obs, pred_mu, obs_mu = binned_average(
                pred=y_hat_dict[horizon][:, token], obs=y_dict[horizon][:, token]
            )
            axs[i].scatter(pred, obs, color=colors[j], marker="o", alpha=0.7)
            axs[i].scatter(pred_mu, obs_mu, color=colors[j], marker="X")

    legend_lines = list()
    for j, horizon in enumerate(time_horizon):
        line = mlines.Line2D([], [], color=colors[j], label=f"{horizon / 365.25} years")
        legend_lines.append(line)

    axs[-1].set_yscale("log")
    axs[-1].set_xscale("log")
    axs[-1].set_ylim(1e-5, 1)
    axs[-1].set_xlim(1e-5, 1)
    fig.supxlabel("predicted rates")
    fig.supylabel("observed rates", x=0.08)
    fig.legend(handles=legend_lines)
    fig.suptitle(title)


# -
doi = [46, 95, 1168, 1188, 374, 214, 305, 505, 603, 1269]
suffix = f"_nreps{args.n_repeats}_int{args.interval}"
token2icd = {v: k for k, v in ds.tokenizer.items()}
doi_titles = [token2icd[token] for token in doi]


plot_calibration(
    doi,
    doi_titles,
    time_horizon,
    base_incidence,
    occur,
    title="disease rates at age 60",
)
plt.savefig(odir / f"baseline_calibrate{suffix}.png", dpi=300)

plot_calibration(
    doi,
    doi_titles,
    time_horizon,
    prob_by_horizon,
    occur,
    title="k / N sampled incidence",
)
plt.savefig(odir / f"simulate_calibrate{suffix}.png", dpi=300)

plot_calibration(
    doi, doi_titles, time_horizon, km_risk, occur, title="calculated risks"
)
plt.savefig(odir / f"integrate_calibrate{suffix}.png", dpi=300)


k = 365.25 * 1
tok = 1188
pred = km_risk[k][:, tok]
obs = occur[k][:, tok]
print((obs == 1).sum(), np.isnan(pred[obs == 1]).sum())
# binned_average(pred=pred, obs=obs)
bins = 10 ** np.arange(-6.0, 1.5, 0.5)
for b in range(1, len(bins)):
    bin_mask = np.logical_and(pred > bins[b - 1], pred <= bins[b])
    print(
        bins[b - 1],
        bins[b],
        bin_mask.sum(),
        np.nanmean(pred[bin_mask]),
        np.nanmean(obs[bin_mask]),
    )

(pred == 1).sum()
