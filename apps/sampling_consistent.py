import math
import pprint
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from delphi.data import Dataset
from delphi.data.transform import Prompt, TokenTransform
from delphi.data.ukb import UKBReader
from delphi.data.utils import collate_batches
from delphi.env import DELPHI_CKPT_DIR
from delphi.eval import (
    KaplanMeierEstimator,
    OnlineSurvivalEstimator,
    kaplan_meier_incidence,
)
from delphi.experiment import GenerateConfig, eval_iter, load_ckpt
from delphi.model.transformer import generate
from delphi.model.utils import self_terminate

# +
args = GenerateConfig.auto(
    ckpt="cluster/homo_poisson/ckpt.pt",
    prompt_age=60,
    prompt_lifestyle=False,
    interval=365.25,
    stop_at_block_size=False,
)
print("args:")
pprint.pp(args)

# +
model, ckpt_dict = load_ckpt(Path(DELPHI_CKPT_DIR) / args.ckpt)
device = "cuda" if torch.cuda.is_available() else "cpu"

reader = UKBReader()
pids = UKBReader.participants("val")

token_transform = TokenTransform(block_size=None)
if args.prompt_age is not None:
    args.prompt_age = args.prompt_age * 365.25
prompt_transform = Prompt(
    prompt_age=args.prompt_age, append_no_event=args.prompt_no_event
)

ds = Dataset(
    reader=reader,
    pids=pids,
    token_transform=token_transform,
    prompt_transform=prompt_transform,
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

    pmt_idx, pmt_age, _, _ = ds.get_batch(batch_idx)
    pmt_idx = pmt_idx.to(device)
    pmt_age = pmt_age.to(device)

    idx, age, stats = generate(
        model=model,
        idx=pmt_idx,
        age=pmt_age,
        max_age=85 * 365.25,
        max_new_tokens=args.max_new_tokens,
        termination_tokens=[1269],
        stop_at_block_size=args.stop_at_block_size,
    )

    with torch.no_grad():
        outputs, _, _ = model(idx, age)
    logits = self_terminate(
        idx,
        outputs["logits"],
        terminate_except=torch.tensor(model.config.self_terminate_except).to(
            idx.device
        ),
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
start_age = 60
end_age = 80
calc = kaplan_meier_incidence(
    surv_prob[None, ...], surv_time, start_age * 365.25, end_age * 365.25
).ravel()
syn = syn_estimator.incidence(start_age * 365.25, end_age * 365.25)


labels = UKBReader.labels()

plt.figure()
plt.scatter(calc[13:], syn[13:], marker=".", c=labels["color"][13:])
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

plt.show()
