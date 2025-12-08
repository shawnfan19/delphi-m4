import os

os.chdir("/hps/nobackup/birney/users/sfan/Delphi")

import argparse
import math
import pprint

# +
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from delphi.data.ukb import UKBDataset, cut_batch_for_prompt
from delphi.data.utils import collate_batches
from delphi.env import DELPHI_CKPT_DIR
from delphi.eval import (
    IntervalRiskCollator,
    KaplanMeierEstimator,
    kaplan_meier_incidence,
)
from delphi.experiment import eval_iter
from delphi.model.transformer import Delphi2M, Delphi2MConfig, generate

delphi_labels = pd.read_csv("notebook/delphi_labels_chapters_colours_icd.csv")

DAYS_PER_YEAR = 365.25


# +
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="delphi-2m-og/ckpt.pt")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--age", type=int, default=60)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--subsample", type=int, default=None)
parser.add_argument("--n_repeats", type=int, default=1)
parser.add_argument("--stop_at_block_size", type=bool, default=False)
parser.add_argument("--max_new_tokens", type=int, default=128)
parser.add_argument("--prompt_no_event", type=bool, default=True)
parser.add_argument("--must_have_lifestyle", type=bool, default=False)


if "ipykernel" in sys.modules:
    print(f"running in jupyter notebook")
    args = parser.parse_args([])
    args.ckpt = "time/zlpr/ckpt.pt"
    args.age = 60
    args.stop_at_block_size = True
else:
    args = parser.parse_args()

print("args:")
pprint.pp(vars(args))
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
ds.subset_participants_for_prompt(
    prompt_age=args.age * 365.25, must_have_lifestyle=args.must_have_lifestyle
)

# +
# outputs, _, _ = model(pmt_idx, pmt_age)
# logits = outputs["logits"][:, -1, :]
# thresh_logits = outputs["aux_rates"][:, -1]
# clamp_min = 0.0
# clamp_max = 365.25 * 80.0

# batch_size = logits.shape[0]
# assert thresh_logits.shape == (batch_size, )
# thresh_logits = thresh_logits.unsqueeze(-1)
# device = logits.device

# # t_next = torch.clamp(
# #     -torch.exp(-logits) * torch.rand(logits.shape, device=device).log(),
# #     min=clamp_min,
# #     max=clamp_max,
# # )

# # t_nod_next = torch.clamp(
# #     -torch.exp(-thresh_logits) * torch.rand(thresh_logits.shape, device=device).log(),
# #     min=clamp_min,
# #     max=clamp_max,
# # )
# # sample_mask = t_next <= t_nod_next

# t_nod_next = torch.clamp(
#     -torch.exp(-thresh_logits) * torch.rand(thresh_logits.shape, device=device).log(),
#     min=clamp_min,
#     max=clamp_max,
# )
# n = sample_mask.sum(dim=1)
# max_n = n.max().item()
# subject_idx, token_idx = torch.nonzero(sample_mask, as_tuple=True)
# pseudo_idx = sample_mask.cumsum(1) - 1
# pseudo_idx = pseudo_idx[sample_mask]

# next_token = torch.zeros((batch_size, int(max_n)), device=device).long()
# next_token[subject_idx, pseudo_idx] = token_idx

# time_til_next = t_nod_next.expand(-1, int(max_n)).clone()
# time_til_next[next_token == 0] = -1e4

# +
# (logits > thresh_logits).sum(dim=1)

# +
# a = torch.bernoulli(torch.clamp(torch.exp(logits - thresh_logits), max=1.0))

# +
# a.sum(dim=-1)

# +
# plt.hist(sample_mask.sum(dim=1).detach().cpu().numpy(), bins=np.arange(25))

# +
time_intervals = np.arange(0, 81 * 365.25, 365.25)
risk_collator = IntervalRiskCollator(
    time_intervals=time_intervals,
)
syn_idx, syn_age = list(), list()
real_idx, real_age = list(), list()

it = eval_iter(total_size=len(ds), batch_size=args.batch_size)
pbar = tqdm(it, total=math.ceil(len(ds) / args.batch_size))
for batch_idx in pbar:

    X0, T0, X1, T1 = ds.get_batch(batch_idx)
    real_idx.append(X1.detach().cpu().numpy())
    real_age.append(T1.detach().cpu().numpy())

    pmt_idx, pmt_age = cut_batch_for_prompt(
        X0, T0, prompt_age=args.age * 365.25, append_no_event=args.prompt_no_event
    )
    pmt_idx, pmt_age = pmt_idx.to(device), pmt_age.to(device)

    idx, age, logits = generate(
        model=model,
        idx=pmt_idx,
        age=pmt_age,
        max_age=85 * 365.25,
        no_repeat=True,
        max_new_tokens=args.max_new_tokens,
        termination_tokens=[1269],
        stop_at_block_size=args.stop_at_block_size,
    )
    syn_idx.append(idx.detach().cpu().numpy())
    syn_age.append(age.detach().cpu().numpy())

    pbar.set_postfix(
        {"prompt block size": pmt_idx.shape[1], "total block size": idx.shape[1]}
    )
    risk_collator.step(tokens=idx, timestep=age, logits=logits)

# -
risk_per_interval = risk_collator.finalize()
risk_per_interval = risk_per_interval.nanmean(dim=0)
risk_per_interval = risk_per_interval[None, :, :]
risk_per_interval.shape

# +
syn_idx = collate_batches(syn_idx)
syn_age = collate_batches(syn_age, fill_value=-1e4)
real_idx = collate_batches(real_idx)
real_age = collate_batches(real_age, fill_value=-1e4)

syn_estimator = KaplanMeierEstimator.from_population(
    timestep=syn_age, tokens=syn_idx, vocab_size=ds.vocab_size
)
real_estimator = KaplanMeierEstimator.from_population(
    timestep=real_age, tokens=real_idx, vocab_size=ds.vocab_size
)
# -


surv_prob = torch.cumprod(1 - risk_per_interval, dim=-1).numpy()


start_age = 60
end_age = 70
surv_time = time_intervals[1:]
calc = kaplan_meier_incidence(
    surv_prob, surv_time, start_age * 365.25, end_age * 365.25
).ravel()
real = real_estimator.incidence(start_age * 365.25, end_age * 365.25)
syn = syn_estimator.incidence(start_age * 365.25, end_age * 365.25)


# +
plt.figure()
plt.scatter(calc[13:], syn[13:], marker=".", c=delphi_labels["color"][13:])
plt.plot([0, 1], [0, 1], c="k", ls=":")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("calculated")
plt.ylabel("simulated")
plt.title(f"probability of disease between age {start_age} and {end_age}")
plt.xlim(1e-5, 1)
plt.ylim(1e-5, 1)

plt.figure()
plt.scatter(syn[13:], real[13:], marker=".", c=delphi_labels["color"][13:])
plt.plot([0, 1], [0, 1], c="k", ls=":")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("simulated")
plt.ylabel("real")
plt.title(f"probability of disease between age {start_age} and {end_age}")
plt.xlim(1e-5, 1)
plt.ylim(1e-5, 1)
# -


plt.figure()
token = -1
plt.plot(
    real_estimator.surv_time[token] / 365.25,
    real_estimator.surv_percent[token],
    label="observed",
    alpha=0.7,
)
plt.plot(
    syn_estimator.surv_time[token] / 365.25,
    syn_estimator.surv_percent[token],
    label="simulated",
    alpha=0.7,
)
plt.plot(surv_time / 365.25, surv_prob[0, token, :], label="calculated", alpha=0.7)
plt.legend()
plt.xlim(60, None)
plt.xlabel("age (years)")
plt.ylabel("S(t)")
