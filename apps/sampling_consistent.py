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

from delphi.data.ukb import UKBDataset, cut_prompt
from delphi.data.utils import collate_batches
from delphi.env import DELPHI_CKPT_DIR
from delphi.eval import (
    KaplanMeierEstimator,
    OnlineSurvivalEstimator,
    kaplan_meier_incidence,
)
from delphi.experiment import eval_iter
from delphi.model.transformer import Delphi2M, Delphi2MConfig, generate

delphi_labels = pd.read_csv("notebook/delphi_labels_chapters_colours_icd.csv")


# +
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="delphi-2m-og/ckpt.pt")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--age", type=int, default=60)
parser.add_argument("--interval", type=float, default=365.25)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--subsample", type=int, default=None)
parser.add_argument("--n_repeats", type=int, default=1)
parser.add_argument("--stop_at_block_size", type=bool, default=False)
parser.add_argument("--max_new_tokens", type=int, default=128)
parser.add_argument("--prompt_no_event", type=bool, default=False)
parser.add_argument("--must_have_lifestyle", type=bool, default=False)


if "ipykernel" in sys.modules:
    print(f"running in jupyter notebook")
    args = parser.parse_args([])
    args.ckpt = "cluster/homo_poisson/ckpt.pt"
    args.age = 60
    args.stop_at_block_size = False
    args.interval = 365.25
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
missing, unexpected = model.load_state_dict(ckpt_dict["model"], strict=False)
print("missing:", missing)
print("unexpected:", unexpected)
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
time_intervals = np.arange(0, 85 * 365.25, args.interval)
risk_collator = OnlineSurvivalEstimator(
    time_intervals=time_intervals, vocab_size=model.config.vocab_size
)
syn_idx, syn_age = list(), list()

it = eval_iter(total_size=len(ds), batch_size=args.batch_size)
pbar = tqdm(it, total=math.ceil(len(ds) / args.batch_size))
for batch_idx in pbar:

    X0, T0, X1, T1 = ds.get_batch(batch_idx)

    pmt_idx, pmt_age = cut_prompt(
        X0,
        T0,
        prompt_age=args.age * 365.25,
        prompt_token=None,
        append_no_event=args.prompt_no_event,
    )
    pmt_idx, pmt_age = pmt_idx.to(device), pmt_age.to(device)

    idx, age, logits, stats = generate(
        model=model,
        idx=pmt_idx,
        age=pmt_age,
        max_age=85 * 365.25,
        # max_age=T0.max(dim=1)[0].to(pmt_idx.device),
        no_repeat=True,
        max_new_tokens=args.max_new_tokens,
        termination_tokens=[1269],
        stop_at_block_size=args.stop_at_block_size,
    )
    syn_idx.append(idx.detach().cpu().numpy())
    syn_age.append(age.detach().cpu().numpy())
    risk_collator.step(tokens=idx, timestep=age, logits=logits)

    pbar.set_postfix(
        {
            "n_gen": stats["n_gen"].mean() - stats["n_prompt"].mean(),
        }
    )
# -


surv_prob, surv_time = risk_collator.finalize()
surv_time = surv_time[1:]

syn_idx = collate_batches(syn_idx)
syn_age = collate_batches(syn_age, fill_value=-1e4)


syn_estimator = KaplanMeierEstimator(
    timestep=syn_age, tokens=syn_idx, vocab_size=model.config.vocab_size
)

# +
# bins = np.arange(60, 85)*365.25
# plt.hist(syn_age.max(axis=1), bins=bins);
# plt.xticks(bins, (bins / 365.25).astype(int));

# +
start_age = 60
end_age = 80
calc = kaplan_meier_incidence(
    surv_prob[None, ...], surv_time, start_age * 365.25, end_age * 365.25
).ravel()
syn = syn_estimator.incidence(start_age * 365.25, end_age * 365.25)

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
# -
plt.figure()
token = -1
plt.plot(
    syn_estimator.surv_time[token] / 365.25,
    syn_estimator.surv_percent[token],
    label="simulated",
    alpha=0.7,
)
plt.plot(surv_time / 365.25, surv_prob[token, :], label="calculated", alpha=0.7)
plt.legend()
plt.xlim(60, None)
plt.xlabel("age (years)")
plt.ylabel("S(t)")
