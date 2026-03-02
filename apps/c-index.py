# +
import argparse
import json
import math
import pprint
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from delphi.data.ukb import MultimodalUKBDataset
from delphi.env import DELPHI_CKPT_DIR
from delphi.eval import DiseaseRatesCollator, SexCollator, correct_time_offset
from delphi.experiment import eval_iter, load_ckpt, move_batch_to_device

# -


class OnsetTimeCollator:
    """Collects raw onset times from (x1, t1) for the at-risk mask in Phase 2."""

    def __init__(self, targets):
        self.targets_set = set(targets.cpu().tolist())
        self._V = int(max(self.targets_set)) + 1
        self._onset = []

    def step(
        self, tokens, timestamps
    ):  # tokens=x1, timestamps=t1, both (B, L) on device
        tokens_cpu = tokens.cpu()
        timestamps_cpu = timestamps.cpu()
        B = tokens_cpu.shape[0]
        out = torch.full((B, self._V), float("nan"))
        for d in torch.unique(tokens_cpu).tolist():
            if d not in self.targets_set:
                continue
            mask = tokens_cpu == d  # (B, L)
            first = mask.long().argmax(dim=1)
            has = mask.any(dim=1)
            out[has, d] = timestamps_cpu[has, first[has]]
        self._onset.append(out)

    def finalize(self):
        return torch.cat(self._onset)  # (N, V)


# +
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="delphi-m4/delphi-m4/ckpt.pt")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--min_time_gap", type=float, default=0.01)
parser.add_argument(
    "--modalities",
    type=str,
    nargs="+",
    default=None,
    help="Only evaluate participants with ALL of these modalities",
)
parser.add_argument("--fname", type=str)

if "ipykernel" in sys.modules:
    print("running in jupyter notebook")
    args = parser.parse_args([])
    args.ckpt = "delphi-m4/baseline/ckpt.pt"
else:
    args = parser.parse_args()

print("args:")
pprint.pp(vars(args))


# +
ckpt = Path(DELPHI_CKPT_DIR) / args.ckpt
model, ckpt_dict = load_ckpt(ckpt)
device = "cuda" if torch.cuda.is_available() else "cpu"

data_args = ckpt_dict["data_args"].copy()
data_args["subject_list"] = "participants/val_fold.bin"
data_args["stats_subject_list"] = ckpt_dict["data_args"]["subject_list"]
if args.modalities is not None:
    data_args["must_have_biomarkers"] = args.modalities

pprint.pp(data_args)
# -

ds = MultimodalUKBDataset(**data_args)

# +
offset_days = args.min_time_gap * 365.25
model_targets = model.targets.to(device)
model_targets = model_targets[model_targets != 1]

dis_collator = DiseaseRatesCollator(targets=model_targets)
sex_collator = SexCollator()
onset_collator = OnsetTimeCollator(targets=model_targets)

it = tqdm(
    eval_iter(total_size=len(ds), batch_size=args.batch_size),
    total=math.ceil(len(ds) / args.batch_size),
    desc="Phase 1",
    leave=False,
)
with torch.no_grad():
    for batch_idx in it:
        batch_input = ds.get_batch(batch_idx)
        batch_input = move_batch_to_device(batch_input, device=device)
        x0, t0, _, _, _, x1, t1 = batch_input

        out_dict, _, _ = model(*batch_input)
        logits = out_dict["logits"].half()

        t0_off, logits_off = correct_time_offset(t0, t1, logits, offset=offset_days)
        dis_collator.step(tokens=x1, timesteps=t0_off, logits=logits_off)
        sex_collator.step(tokens=x0)
        onset_collator.step(tokens=x1, timestamps=t1)

dis_rates, dis_times = dis_collator.finalize()  # (N, V)
is_female = sex_collator.finalize()  # (N,)
onset_times = onset_collator.finalize()  # (N, V)


# +
# Flatten case events: each non-NaN entry in dis_rates is a case
event_p_idx, event_d_idx = (~torch.isnan(dis_rates)).nonzero(as_tuple=True)  # (E,)

# Case score at offset-corrected query time
event_case_scores = dis_rates[event_p_idx, event_d_idx].float()  # (E,)
# Offset-corrected t0 used to score the case — used as searchsorted query for controls
event_query_times = dis_times[event_p_idx, event_d_idx].float()  # (E,)
# Raw onset time — used for the at-risk check
event_actual_times = onset_times[event_p_idx, event_d_idx].float()  # (E,)
event_sex = is_female[event_p_idx]  # (E,) bool

print(f"Total events across all diseases: {len(event_p_idx)}")

# Convert to numpy for the Phase 2 inner loop
onset_times_np = onset_times.numpy()  # (N, V)
event_p_idx_np = event_p_idx.numpy()  # (E,)
event_d_idx_np = event_d_idx.numpy()  # (E,)
event_case_scores_np = event_case_scores.numpy()  # (E,)
event_query_times_np = event_query_times.numpy()  # (E,)
event_actual_times_np = event_actual_times.numpy()  # (E,)
event_sex_np = event_sex.numpy()  # (E,) bool


# +
# Phase 2: for every (participant, event) pair, check concordance
concordant = np.zeros(len(event_p_idx_np), dtype=np.float64)
total_pairs = np.zeros(len(event_p_idx_np), dtype=np.float64)

participant_offset = 0

it2 = tqdm(
    eval_iter(total_size=len(ds), batch_size=args.batch_size),
    total=math.ceil(len(ds) / args.batch_size),
    desc="Phase 2",
    leave=False,
)
with torch.no_grad():
    for batch_idx in it2:
        batch_input = ds.get_batch(batch_idx)
        batch_input = move_batch_to_device(batch_input, device=device)
        x0, t0, _, _, _, x1, t1 = batch_input

        # No correct_time_offset: raw t0 and logits for searchsorted lookup
        out_dict, _, _ = model(*batch_input)
        logits = out_dict["logits"].half()

        t0_np = t0.cpu().float().numpy()  # (B, L)
        logits_np = logits.cpu().float().numpy()  # (B, L, V)
        B, L, V = logits_np.shape

        for j in range(B):
            j_global = participant_offset + j
            t_j = t0_np[j]  # (L,) sorted ascending

            # Locate each event's query horizon in participant j's timeline
            idx = np.searchsorted(t_j, event_query_times_np, side="right") - 1  # (E,)
            idx_clipped = np.clip(idx, 0, L - 1)
            t_at_idx = t_j[idx_clipped]  # (E,)

            # Exclude out-of-range positions and padding (padding timestamps are -1e4)
            valid = (idx >= 0) & (t_at_idx > 0)

            # At-risk: j had not yet developed disease d when the case event occurred
            j_onset = onset_times_np[j_global, event_d_idx_np]  # (E,)
            valid &= np.isnan(j_onset) | (j_onset > event_actual_times_np)

            # Do not compare a case to itself
            valid &= j_global != event_p_idx_np

            # j's predicted score for each disease at the event's query time
            scores_j = logits_np[j, idx_clipped, event_d_idx_np]  # (E,)

            concordant += valid & (scores_j < event_case_scores_np)
            total_pairs += valid

        participant_offset += B


# +
# Aggregate C-index per disease per sex
reverse_tokenizer = {v: k for k, v in ds.tokenizer.items()}

result = {}
for d_int in np.unique(event_d_idx_np):
    d_mask = event_d_idx_np == d_int
    icd = reverse_tokenizer.get(int(d_int), str(d_int))
    result[icd] = {}
    for sex_label, sex_mask in [
        ("female", event_sex_np),
        ("male", ~event_sex_np),
        ("either", np.ones(len(event_sex_np), dtype=bool)),
    ]:
        mask = d_mask & sex_mask
        n_events = int(mask.sum())
        n_pairs = int(total_pairs[mask].sum())
        conc = concordant[mask].sum()
        c_index = round(float(conc / n_pairs), 4) if n_pairs > 0 else None
        result[icd][sex_label] = {
            "c_index": c_index,
            "n_events": n_events,
            "n_pairs": n_pairs,
        }
# -


pprint.pp(result.get("death", result.get(next(iter(result)), {})))

if args.fname is None:
    args.fname = f"cindex-min_time_gap-{args.min_time_gap}-ckpt-{ckpt.stem}"
    if args.modalities is not None:
        args.fname += f"-modalities_{'_'.join(args.modalities)}"

out_path = ckpt.parent / f"{args.fname}.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=4)
print(f"Saved to {out_path}")
