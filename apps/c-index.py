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

# Numpy arrays needed only for post-processing
event_d_idx_np = event_d_idx.numpy()  # (E,)
event_sex_np = event_sex.numpy()  # (E,) bool

# Move event arrays and onset_times to GPU once for Phase 2
chunk_size = 8192
onset_times_gpu = onset_times.to(device)  # (N, V)
event_query_times_t = event_query_times.to(device)  # (E,)
event_actual_times_t = event_actual_times.to(device)  # (E,)
event_case_scores_t = event_case_scores.to(device)  # (E,)
event_d_idx_t = event_d_idx.to(device)  # (E,) long
event_p_idx_t = event_p_idx.to(device)  # (E,) long


# +
# Phase 2: for every (participant, event) pair, check concordance — fully on GPU
E_total = len(event_p_idx)
concordant = np.zeros(E_total, dtype=np.float64)
total_pairs = np.zeros(E_total, dtype=np.float64)

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
        logits = out_dict["logits"].half()  # (B, L, V)

        t0_f = t0.float()  # (B, L) float32 required by searchsorted
        B, L, V = logits.shape
        j_globals = torch.arange(B, device=device) + participant_offset  # (B,)

        for e_start in range(0, E_total, chunk_size):
            e_end = min(e_start + chunk_size, E_total)
            E_c = e_end - e_start

            t_q = event_query_times_t[e_start:e_end]  # (E_c,)
            t_act = event_actual_times_t[e_start:e_end]  # (E_c,)
            d_idx = event_d_idx_t[e_start:e_end]  # (E_c,)
            p_idx = event_p_idx_t[e_start:e_end]  # (E_c,)
            c_sc = event_case_scores_t[e_start:e_end]  # (E_c,)

            # Batched searchsorted: (B, L) sorted × (B, E_c) queries → (B, E_c) indices
            idx_mat = (
                torch.searchsorted(
                    t0_f.contiguous(), t_q.unsqueeze(0).expand(B, -1), right=True
                )
                - 1
            )  # (B, E_c)
            idx_c = idx_mat.clamp(0, L - 1)

            # Timestamps and logit scores at each found position
            t_at = t0_f.gather(1, idx_c)  # (B, E_c)
            flat_b = (
                torch.arange(B, device=device).unsqueeze(1).expand(-1, E_c).reshape(-1)
            )
            scores = logits[
                flat_b,
                idx_c.reshape(-1),
                d_idx.unsqueeze(0).expand(B, -1).reshape(-1),
            ].reshape(
                B, E_c
            )  # (B, E_c) half

            # Validity: within timeline and not padding (padding ≈ -1e4)
            valid = (idx_mat >= 0) & (t_at > 0)
            # At-risk: j had not yet developed disease d at the case's event time
            j_onset = onset_times_gpu[
                j_globals.unsqueeze(1), d_idx.unsqueeze(0).expand(B, -1)
            ]  # (B, E_c)
            valid &= j_onset.isnan() | (j_onset > t_act.unsqueeze(0))
            # Do not compare a case to itself
            valid &= j_globals.unsqueeze(1) != p_idx.unsqueeze(0)

            concordant[e_start:e_end] += (
                (valid & (scores.float() < c_sc.unsqueeze(0))).sum(0).cpu().numpy()
            )
            total_pairs[e_start:e_end] += valid.sum(0).cpu().numpy()

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
