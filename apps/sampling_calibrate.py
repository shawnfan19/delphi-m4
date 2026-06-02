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

from delphi.data import Dataset
from delphi.data.transform import Prompt, TokenTransform
from delphi.data.ukb import UKBReader
from delphi.data.utils import collate_batches, pack_clusters
from delphi.env import DELPHI_CKPT_DIR
from delphi.eval import KaplanMeierEstimator
from delphi.experiment import GenerateConfig, eval_iter, load_ckpt
from delphi.model.transformer import generate

# -


args = GenerateConfig.from_cli()
print("args:")
pprint.pp(args)

model, ckpt_dict = load_ckpt(Path(DELPHI_CKPT_DIR) / args.ckpt)
device = "cuda" if torch.cuda.is_available() else "cpu"

reader = UKBReader()
pids = UKBReader.participants("val")

token_transform_args = ckpt_dict["token_transform_args"]
token_transform = TokenTransform(**token_transform_args)
token_transform.describe()

lifestyle_tokens = [reader.tokenizer[k] for k in reader.lifestyle_keys]
sex_tokens = [reader.tokenizer[k] for k in reader.sex_keys]
dx_token = token_transform_args.get("dx_token") or 1
break_clusters = token_transform_args.get("break_clusters", False)
whitelist = np.array([0, 1, dx_token] + sex_tokens + lifestyle_tokens)

if dx_token != 1:
    model.config.self_terminate_except = list(
        set(model.config.self_terminate_except).union({dx_token})
    )

prompt_age_arg = args.prompt_age if args.prompt_age is not None else "recruitment"
if prompt_age_arg == "recruitment":
    rec = reader.recruitment_times(pids)
    has_rec = ~np.isnan(rec)
    pids = pids[has_rec]
    prompt_age = {int(p): float(a) for p, a in zip(pids, rec[has_rec])}
else:
    prompt_age = float(prompt_age_arg) * 365.25
prompt_transform = Prompt(prompt_age=prompt_age, append_no_event=args.prompt_no_event)
ds = Dataset(
    reader=reader,
    pids=pids,
    token_transform=token_transform,
    prompt_transform=prompt_transform,
)

# +
syn_idx, syn_age = list(), list()
real_idx, real_age = list(), list()

it = eval_iter(total_size=len(ds), batch_size=args.batch_size)
device = "cuda" if torch.cuda.is_available() else "cpu"
pbar = tqdm(it, total=math.ceil(len(ds) / args.batch_size))
for batch_idx in pbar:

    pmt_idx, pmt_age, X1, T1 = ds.get_batch(batch_idx)

    X1_np = X1.detach().cpu().numpy()
    T1_np = T1.detach().cpu().numpy()
    if break_clusters:
        X1_np, T1_np = pack_clusters(X1_np, T1_np, whitelist, dx_token=dx_token)

    real_idx.append(X1_np)
    real_age.append(T1_np)

    pmt_idx, pmt_age = pmt_idx.to(device), pmt_age.to(device)

    idx, age, stats = generate(
        model=model,
        idx=pmt_idx,
        age=pmt_age,
        max_age=T1.max(dim=1)[0].to(device),
        termination_tokens=[1269],
        stop_at_block_size=True,
        cached=True,
    )
    idx = idx.detach().cpu().numpy()
    age = age.detach().cpu().numpy()
    if break_clusters:
        idx, age = pack_clusters(idx, age, whitelist, dx_token=dx_token)

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
