import os

os.chdir("/hps/nobackup/birney/users/sfan/Delphi")

import argparse
import math
import pprint

# +
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from delphi.data.ukb import UKBDataset, cut_batch_for_prompt
from delphi.env import DELPHI_CKPT_DIR
from delphi.eval import (
    EventTimeCollator,
    IntervalKaplanMeierCollator,
    SamplingProbCollator,
    mann_whitney_auc,
)
from delphi.experiment import eval_iter
from delphi.model.transformer import Delphi2M, Delphi2MConfig, generate

# +
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="delphi-2m-og/ckpt.pt")
parser.add_argument("--age", type=int, default=60)
parser.add_argument("--interval", type=float, default=1.0)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--subsample", type=int, default=None)
parser.add_argument("--n_repeats", type=int, default=1)
parser.add_argument("--stop_at_block_size", type=bool, default=True)
parser.add_argument("--max_new_tokens", type=int, default=128)
parser.add_argument("--prompt_no_event", type=bool, default=True)
parser.add_argument("--must_have_lifestyle", type=bool, default=False)

if "ipykernel" in sys.modules:
    print(f"running in jupyter notebook")
    args = parser.parse_args([])
    # args.ckpt = "delphi-2m-ablation/delphi-2m-long-ctx/ckpt.pt"
    args.stop_at_block_size = True
    args.prompt_no_event = True
    args.n_repeats = 1
    args.interval = 365.25
else:
    args = parser.parse_args()

print("args:")
pprint.pp(vars(args))
# -
time_horizon = np.array([1, 3, 5, 10, 15]) * 365.25
start_age = args.age * 365.25
time_intervals = np.arange(0, 81 * 365.25, args.interval)

# +
ckpt = Path(DELPHI_CKPT_DIR) / args.ckpt
ckpt_dict = torch.load(
    ckpt,
    map_location=torch.device("cpu") if not torch.cuda.is_available() else None,
)
model = Delphi2M(Delphi2MConfig(**ckpt_dict["model_args"]))
pprint.pp(ckpt_dict["model_args"])
model.load_state_dict(ckpt_dict["model"])
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

exclude_lifestyle = ckpt_dict["config"].get("exclude_lifestyle", False)
no_event_mode = ckpt_dict["config"].get("no_event_mode", "legacy-random")
# -
ds = UKBDataset(
    data_dir="ukb_real_data",
    subject_list="participants/val_fold.bin",
    perturb=False,
    no_event_mode=no_event_mode,
    exclude=exclude_lifestyle,
    block_size=model.config.block_size,
)
ds.subset_participants_for_prompt(prompt_age=args.age * 365.25)


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
    pmt_idx, pmt_age = cut_batch_for_prompt(
        X0, T0, prompt_age=args.age * 365.25, append_no_event=args.prompt_no_event
    )
    pmt_idx, pmt_age = pmt_idx.to(device), pmt_age.to(device)

    output, _, _ = model.forward(pmt_idx, pmt_age)
    logits_at_prompt.append(output["logits"][:, -1, :].detach().cpu().numpy())

    pmt_idx = torch.repeat_interleave(pmt_idx, args.n_repeats, dim=0)
    pmt_age = torch.repeat_interleave(pmt_age, args.n_repeats, dim=0)
    tokens, timestep, logits = generate(
        model=model,
        idx=pmt_idx,
        age=pmt_age,
        max_age=85 * 365.25,
        no_repeat=True,
        max_new_tokens=args.max_new_tokens,
        termination_tokens=[1269],
        stop_at_block_size=args.stop_at_block_size,
    )

    pbar.set_postfix(
        {"prompt block size": pmt_idx.shape[1], "total block size": tokens.shape[1]}
    )

    risk_collator.step(tokens, timestep, logits)
    prob_collator.step(tokens, timestep)

occur_time, exit_time = event_time_collator.finalize()
prob_by_horizon = prob_collator.finalize()
km_risk = risk_collator.finalize()
# -


risk_collator.time_intervals[1:], risk_collator.time_horizon[
    0
] + risk_collator.start_age

km_risk[365.25][:, -1]

logits_at_prompt = np.concatenate(logits_at_prompt, axis=0)
base_incidence = dict()
for horizon in time_horizon:
    base_incidence[horizon] = 1 - np.exp(-np.exp(logits_at_prompt) * horizon)

occur = dict()
for horizon in time_horizon:

    start = args.age * 365.25
    end = start + horizon

    _occur = np.zeros((len(ds), ds.vocab_size))
    _occur[np.logical_and(occur_time > start, occur_time <= end)] = 1
    _occur[occur_time <= start] = float("nan")

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

    start = args.age * 365.25
    end = start + horizon

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
plt.savefig(ckpt.parent / f"sampling_auc_nreps{args.n_repeats}.png", dpi=300)


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
plt.savefig(ckpt.parent / f"baseline_calibrate{suffix}.png", dpi=300)

plot_calibration(
    doi,
    doi_titles,
    time_horizon,
    prob_by_horizon,
    occur,
    title="k / N sampled incidence",
)
plt.savefig(ckpt.parent / f"simulate_calibrate{suffix}.png", dpi=300)

plot_calibration(
    doi, doi_titles, time_horizon, km_risk, occur, title="calculated risks"
)
plt.savefig(ckpt.parent / f"integrate_calibrate{suffix}.png", dpi=300)
