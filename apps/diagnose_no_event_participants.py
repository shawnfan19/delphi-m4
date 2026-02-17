# +
"""
Diagnostic script to check how many validation participants have no real-event
targets (i.e., all their target tokens are either padding, ignored, or no-event).

This explains the participant count discrepancy between models trained with and
without no-event tokens in eval_nll.py.
"""
import argparse
import pprint
import sys
from pathlib import Path

import numpy as np
from tqdm.autonotebook import tqdm

from delphi.data.ukb import NO_EVENT_TOKEN, UKBDataset
from delphi.env import DELPHI_CKPT_DIR
from delphi.experiment import load_ckpt

# -

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="path/to/ckpt.pt")

if "ipykernel" in sys.modules:
    print("running in jupyter notebook")
    args = parser.parse_args([])
    args.ckpt = "debug/ckpt.pt"
else:
    args = parser.parse_args()

print("args:")
pprint.pp(vars(args))

# +
ckpt_path = Path(DELPHI_CKPT_DIR) / args.ckpt
model, ckpt_dict = load_ckpt(ckpt_path)

data_args = ckpt_dict["data_args"].copy()
data_args["subject_list"] = "participants/val_fold.bin"
data_args["perturb"] = False
data_args["deterministic"] = True

ds = UKBDataset(**data_args)

has_no_event = data_args.get("no_event_interval") is not None

# tokens to ignore (matching training logic)
ignored_tokens = {0}
if model.config.ignore_tokens is not None:
    ignored_tokens.update(model.config.ignore_tokens)


# +
n_total = len(ds)
n_no_real_targets = 0  # participants with zero real-event targets
n_all_no_event = 0  # participants whose targets are all no-event (subset of above)
seq_len_no_real = []  # sequence lengths of participants with no real targets

for i in tqdm(range(n_total), leave=False):
    x0, t0, x1, t1 = ds[i]

    # build valid mask (same as eval_nll.py / training)
    valid = np.ones(len(x1), dtype=bool)
    for k in ignored_tokens:
        valid &= x1 != k

    is_no_event = x1 == NO_EVENT_TOKEN
    real_event_mask = valid & ~is_no_event

    if real_event_mask.sum() == 0:
        n_no_real_targets += 1
        seq_len_no_real.append(len(x1))

        if valid.sum() > 0 and is_no_event[valid].all():
            n_all_no_event += 1

# -

print(f"\n{'='*60}")
print(f"Total validation participants: {n_total}")
print(f"Participants with NO real-event targets: {n_no_real_targets}")
print(f"  of which all valid targets are no-event: {n_all_no_event}")
print(
    f"  of which have zero valid targets at all: {n_no_real_targets - n_all_no_event}"
)
print(
    f"Participants with at least one real-event target: {n_total - n_no_real_targets}"
)
print(f"has_no_event: {has_no_event}")
if seq_len_no_real:
    print(f"\nSequence lengths of no-real-target participants:")
    print(f"  mean: {np.mean(seq_len_no_real):.1f}")
    print(f"  median: {np.median(seq_len_no_real):.1f}")
    print(f"  min: {np.min(seq_len_no_real)}, max: {np.max(seq_len_no_real)}")
print(f"{'='*60}")
