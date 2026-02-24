# +
import math
import os
import pprint
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from delphi.data.ukb import UKBDataset, cut_prompt
from delphi.data.utils import collate_batches, pack_clusters
from delphi.env import DELPHI_CKPT_DIR
from delphi.eval import KaplanMeierEstimator
from delphi.experiment import GenerateConfig, eval_iter, load_ckpt
from delphi.model.transformer import Delphi2M, Delphi2MConfig, generate

# -


args = GenerateConfig.auto(ckpt="cluster/freectx/ckpt.pt")
print("args:")
pprint.pp(args)

model, ckpt_dict = load_ckpt(Path(DELPHI_CKPT_DIR) / args.ckpt)
data_args = ckpt_dict["data_args"]
data_args["subject_list"] = "participants/val_fold.bin"
data_args["perturb"] = False
data_args["deterministic"] = True
data_args["crop_mode"] = "left"
data_args["additional_dx_token"] = ckpt_dict["data_args"].get(
    "additional_dx_token", False
)
pprint.pp(data_args)

ds = UKBDataset(**data_args)
prompt_age = args.prompt_age * 365.25 if args.prompt_age is not None else None
prompt_tokens = ds.lifestyle_tokens if args.prompt_lifestyle else None
ds.subset_participants_for_prompt(prompt_age=prompt_age, prompt_tokens=prompt_tokens)


if data_args["additional_dx_token"]:
    model.config.self_terminate_except.append(ds.dx_token)
model.config.self_terminate_except

break_clusters = data_args.get("break_clusters", False)
break_clusters

whitelist = np.concatenate(
    (np.array([0, 1]), np.array([ds.dx_token]), ds.sex_tokens, ds.lifestyle_tokens)
)
whitelist

# +
syn_idx, syn_age = list(), list()
real_idx, real_age = list(), list()

it = eval_iter(total_size=len(ds), batch_size=args.batch_size)
device = "cuda" if torch.cuda.is_available() else "cpu"
pbar = tqdm(it, total=math.ceil(len(ds) / args.batch_size))
for batch_idx in pbar:

    X0, T0, X1, T1 = ds.get_batch(batch_idx)

    X1_np = torch.cat((X0, X1[:, [-1]]), dim=1).detach().cpu().numpy()
    T1_np = torch.cat((T0, T1[:, [-1]]), dim=1).detach().cpu().numpy()

    if break_clusters:
        X1_np, T1_np = pack_clusters(X1_np, T1_np, whitelist, dx_token=ds.dx_token)

    real_idx.append(X1_np)
    real_age.append(T1_np)

    pmt_idx, pmt_age, _ = cut_prompt(
        X0,
        T0,
        prompt_age=prompt_age,
        prompt_token=torch.Tensor(prompt_tokens) if prompt_tokens is not None else None,
        append_no_event=args.prompt_no_event,
    )
    pmt_idx, pmt_age = pmt_idx.to(device), pmt_age.to(device)

    idx, age, stats = generate(
        model=model,
        idx=pmt_idx,
        age=pmt_age,
        max_age=T1.max(dim=1)[0].to(pmt_idx.device),
        # no_repeat=True,
        # no_repeat_except=torch.Tensor([1, ds.dx_token]),
        max_new_tokens=args.max_new_tokens,
        termination_tokens=[1269],
        stop_at_block_size=True,
    )
    idx = idx.detach().cpu().numpy()
    age = age.detach().cpu().numpy()
    if break_clusters:
        idx, age = pack_clusters(idx, age, whitelist, dx_token=ds.dx_token)

    syn_idx.append(idx)
    syn_age.append(age)

    pbar.set_postfix(
        {
            "n_gen": stats["n_gen"].mean(),
            "n_pmt": stats["n_prompt"].mean(),
        }
    )


# +
syn_idx = collate_batches(syn_idx)
syn_age = collate_batches(syn_age, fill_value=-1e4)
real_idx = collate_batches(real_idx)
real_age = collate_batches(real_age, fill_value=-1e4)

syn_estimator = KaplanMeierEstimator(
    timestep=syn_age, tokens=syn_idx, vocab_size=model.config.vocab_size
)
real_estimator = KaplanMeierEstimator(
    timestep=real_age, tokens=real_idx, vocab_size=model.config.vocab_size
)


# +
start_age = 60
end_age = 80

real = real_estimator.incidence(start_age * 365.25, end_age * 365.25)
syn = syn_estimator.incidence(start_age * 365.25, end_age * 365.25)

plt.figure()
plt.scatter(
    syn[13:],
    real[13:],
    marker=".",
    # c=delphi_labels["color"][13:]
)
plt.plot([0, 1], [0, 1], c="k", ls=":")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("simulated")
plt.ylabel("real")
plt.title(f"probability of disease between age {start_age} and {end_age}")
plt.xlim(1e-5, 1)
plt.ylim(1e-5, 1)
# -

plt.figure(figsize=(15, 5))
bins = np.arange(30, 85) * 365.25
plt.hist(real_age.max(axis=1), bins=bins, alpha=0.3, label="real")
plt.hist(syn_age.max(axis=1), bins=bins, alpha=0.3, label="generated")
plt.xticks(bins, (bins / 365.25).astype(int))
plt.xlabel("age of final token")
plt.ylabel("# participants")
plt.legend()
