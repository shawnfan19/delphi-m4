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
from delphi.eval import KaplanMeierEstimator
from delphi.experiment import eval_iter
from delphi.model.transformer import Delphi2M, Delphi2MConfig, generate

delphi_labels = pd.read_csv("notebook/delphi_labels_chapters_colours_icd.csv")


# +
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="delphi-2m-og/ckpt.pt")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--age", type=int, default=60)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--subsample", type=int, default=None)
parser.add_argument("--n_repeats", type=int, default=1)
parser.add_argument("--stop_at_block_size", type=bool, default=True)
parser.add_argument("--max_new_tokens", type=int, default=128)
parser.add_argument("--prompt_no_event", type=bool, default=False)
parser.add_argument("--must_have_lifestyle", type=bool, default=False)


if "ipykernel" in sys.modules:
    print(f"running in jupyter notebook")
    args = parser.parse_args([])
    args.ckpt = "cluster/dx_token/ckpt.pt"
    args.age = None
    args.must_have_lifestyle = True
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
# -
data_args = ckpt_dict["data_args"]
data_args["subject_list"] = "participants/val_fold.bin"
data_args["perturb"] = False
data_args["deterministic"] = True
data_args["crop_mode"] = "left"
data_args["additional_dx_token"] = ckpt_dict["data_args"].get(
    "additional_dx_token", False
)
pprint.pp(data_args)

# +
ds = UKBDataset(**data_args)

if args.age is not None:
    ds.subset_participants_for_prompt(
        prompt_age=args.age * 365.25, must_have_lifestyle=args.must_have_lifestyle
    )
else:
    ds.subset_by_tokens(ds.lifestyle_tokens)


# +
syn_idx, syn_age = list(), list()
real_idx, real_age = list(), list()
prompt_age = args.age * 365.25 if args.age is not None else None

it = eval_iter(total_size=len(ds), batch_size=args.batch_size)
pbar = tqdm(it, total=math.ceil(len(ds) / args.batch_size))
for batch_idx in pbar:

    X0, T0, X1, T1 = ds.get_batch(batch_idx)

    real_idx.append(torch.cat((X0, X1[:, [-1]]), dim=1).detach().cpu().numpy())
    real_age.append(torch.cat((T0, T1[:, [-1]]), dim=1).detach().cpu().numpy())

    pmt_idx, pmt_age, _ = cut_prompt(
        X0,
        T0,
        prompt_age=prompt_age,
        prompt_token=torch.Tensor(ds.lifestyle_tokens),
        append_no_event=args.prompt_no_event,
    )
    pmt_idx, pmt_age = pmt_idx.to(device), pmt_age.to(device)

    idx, age, logits, stats = generate(
        model=model,
        idx=pmt_idx,
        age=pmt_age,
        max_age=T1.max(dim=1)[0].to(pmt_idx.device),
        no_repeat=True,
        no_repeat_except=torch.Tensor([1, ds.dx_token]),
        max_new_tokens=args.max_new_tokens,
        termination_tokens=[1269],
        stop_at_block_size=True,
    )
    syn_idx.append(idx.detach().cpu().numpy())
    syn_age.append(age.detach().cpu().numpy())

    pbar.set_postfix(
        {
            "n_gen": stats["n_gen"].mean() - stats["n_prompt"].mean(),
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
