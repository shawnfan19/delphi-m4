# +
import math
import pprint
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from delphi.data import MultimodalDataset
from delphi.data.transform import BiomarkerTransform, MultimodalPrompt, TokenTransform
from delphi.data.ukb import MultimodalUKBReader
from delphi.data.utils import collate_batches
from delphi.env import DELPHI_CKPT_DIR
from delphi.eval import KaplanMeierEstimator
from delphi.experiment import GenerateConfig, eval_iter, load_ckpt
from delphi.model.transformer import generate

# -


args = GenerateConfig.from_cli()
# interactive debug: GenerateConfig(ckpt="delphi-m4/new_blood/ckpt.pt")
print("args:")
pprint.pp(args)

model, ckpt_dict = load_ckpt(Path(DELPHI_CKPT_DIR) / args.ckpt)
data_args = ckpt_dict["data_args"]
pprint.pp(data_args)

reader = MultimodalUKBReader(
    biomarkers=ckpt_dict["config"]["biomarkers"],
    expansion_packs=ckpt_dict["config"]["expansion_packs"],
)

pids = MultimodalUKBReader.participants("val")

biomarkers = ckpt_dict["config"]["biomarkers"]
pmt_bio_m = np.array([m.value if hasattr(m, "value") else m for m in biomarkers])
prompt_age = {}
for pid in pids:
    _, _, _, bio_t, bio_m = reader[pid]
    is_pmt = np.isin(bio_m, pmt_bio_m)
    assert is_pmt.any(), f"no prompt biomarkers for pid {pid}"
    prompt_age[pid] = bio_t[is_pmt].max()

token_transform = TokenTransform(block_size=None, crop_mode="left")
biomarker_transform = BiomarkerTransform()
prompt_transform = MultimodalPrompt(prompt_age=prompt_age)

ds = MultimodalDataset(
    reader=reader,
    pids=pids,
    token_transform=token_transform,
    biomarker_transform=biomarker_transform,
    prompt_transform=prompt_transform,
)

model.config.self_terminate_except

# +
syn_idx, syn_age = list(), list()
real_idx, real_age = list(), list()

it = eval_iter(total_size=len(ds), batch_size=args.batch_size)
device = "cuda" if torch.cuda.is_available() else "cpu"
pbar = tqdm(it, total=math.ceil(len(ds) / args.batch_size))
for batch_idx in pbar:

    pmt_idx, pmt_age, pmt_bio_x_dict, pmt_bio_t, pmt_bio_m, X1, T1 = ds.get_batch(
        batch_idx
    )

    X1_np = X1.detach().cpu().numpy()
    T1_np = T1.detach().cpu().numpy()

    real_idx.append(X1_np)
    real_age.append(T1_np)

    pmt_idx = pmt_idx.to(device)
    pmt_age = pmt_age.to(device)
    pmt_bio_x_dict = {k: v.to(device) for k, v in pmt_bio_x_dict.items()}
    pmt_bio_t = pmt_bio_t.to(device)
    pmt_bio_m = pmt_bio_m.to(device)

    idx, age, stats = generate(
        model=model,
        idx=pmt_idx,
        age=pmt_age,
        max_age=T1.max(dim=1)[0].to(pmt_idx.device),
        termination_tokens=[1269],
        stop_at_block_size=True,
        cached=True,
        biomarker=pmt_bio_x_dict,
        mod_age=pmt_bio_t,
        mod_idx=pmt_bio_m,
    )
    idx = idx.detach().cpu().numpy()
    age = age.detach().cpu().numpy()

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
