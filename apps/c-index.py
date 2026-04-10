# +
import argparse
import json
import math
import os
import pprint
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from delphi.data.ukb import MultimodalUKBDataset
from delphi.env import DELPHI_CKPT_READ, DELPHI_CKPT_WRITE
from delphi.eval import (
    DiseaseRatesCollator,
    EventTimeCollator,
    SexCollator,
    correct_time_offset,
)
from delphi.experiment import eval_iter, load_ckpt, move_batch_to_device


# +
class ConcordanceCollator:

    def __init__(
        self,
        dis_rates,
        onset_times,
        is_female,
        offset,
        chunk_size=8192,
        max_gap_days=1826.25,
        cutoff=None,
    ):
        # Flatten case events: each non-NaN entry in dis_rates is a case
        case_participants, case_tokens = (~torch.isnan(dis_rates)).nonzero(
            as_tuple=True
        )
        self.case_scores = dis_rates[case_participants, case_tokens].float()
        self.case_times = onset_times[case_participants, case_tokens].float()
        self.case_tokens = case_tokens
        self.case_participants = case_participants
        self.case_sex = is_female[case_participants].cpu().numpy()

        self.query_times = self.case_times - offset
        self.onset_times = onset_times
        self.chunk_size = chunk_size
        self.max_gap_days = max_gap_days
        self.cutoff = cutoff

        E = len(case_participants)
        self.concordant_pairs = np.zeros(E, dtype=np.float64)
        self.total_pairs = np.zeros(E, dtype=np.float64)
        self.participant_offset = 0

    def step(self, age, scores):
        B, L, V = scores.shape
        device = scores.device
        E_total = len(self.case_tokens)
        j_globals = torch.arange(B, device=device) + self.participant_offset

        for e_start in range(0, E_total, self.chunk_size):
            e_end = min(e_start + self.chunk_size, E_total)
            E_c = e_end - e_start

            chunk_query_times = self.query_times[e_start:e_end]
            chunk_case_times = self.case_times[e_start:e_end]
            chunk_tokens = self.case_tokens[e_start:e_end]
            chunk_participants = self.case_participants[e_start:e_end]
            chunk_scores = self.case_scores[e_start:e_end]

            # Batched searchsorted: (B, L) sorted × (B, E_c) queries → (B, E_c) indices
            idx_mat = (
                torch.searchsorted(
                    age.contiguous(),
                    chunk_query_times.unsqueeze(0).expand(B, -1).contiguous(),
                    right=True,
                )
                - 1
            )
            idx_c = idx_mat.clamp(0, L - 1)

            # Timestamps and scores at each found position
            t_at = age.gather(1, idx_c)
            flat_b = (
                torch.arange(B, device=device).unsqueeze(1).expand(-1, E_c).reshape(-1)
            )
            ctrl_scores = scores[
                flat_b,
                idx_c.reshape(-1),
                chunk_tokens.unsqueeze(0).expand(B, -1).reshape(-1),
            ].reshape(B, E_c)

            # Validity: within timeline and not padding
            valid = (idx_mat >= 0) & (t_at > 0)
            # Max gap: control score must be within max_gap of query time
            valid &= (chunk_query_times.unsqueeze(0) - t_at) < self.max_gap_days
            # Control score must be after control's biomarker cutoff
            if self.cutoff is not None:
                valid &= t_at >= self.cutoff[j_globals].unsqueeze(1)
            # At-risk: control had not yet developed disease at the case's event time
            j_onset = self.onset_times[
                j_globals.unsqueeze(1), chunk_tokens.unsqueeze(0).expand(B, -1)
            ]
            valid &= j_onset.isnan() | (j_onset > chunk_case_times.unsqueeze(0))
            # Do not compare a case to itself
            valid &= j_globals.unsqueeze(1) != chunk_participants.unsqueeze(0)

            self.concordant_pairs[e_start:e_end] += (
                (valid & (ctrl_scores.float() < chunk_scores.unsqueeze(0)))
                .sum(0)
                .cpu()
                .numpy()
            )
            self.total_pairs[e_start:e_end] += valid.sum(0).cpu().numpy()

        self.participant_offset += B

    def finalize(self):
        return (
            self.case_sex,
            self.case_tokens.cpu().numpy(),
            self.total_pairs,
            self.concordant_pairs,
        )


def parse_modalities(modalities):
    if modalities is None:
        return None, None
    if len(modalities) == 1 and modalities[0].endswith(".yaml"):
        path = Path(modalities[0])
        with open(path) as f:
            return yaml.safe_load(f), path.stem
    return modalities, None


# +
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="delphi-m4/delphi-m4/ckpt.pt")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--min_time_gap", type=float, default=0)
parser.add_argument(
    "--modalities",
    type=str,
    nargs="+",
    default=None,
    help="Only evaluate participants with ALL of these modalities",
)
parser.add_argument(
    "--max_gap",
    type=float,
    default=5,
    help="Maximum allowed gap in years between case query time and control score time",
)
parser.add_argument(
    "--after_modality",
    action="store_true",
    default=False,
    help="Only compute C-index at time points after first occurrence of specified modalities",
)
parser.add_argument("--fname", type=str)

if "ipykernel" in sys.modules:
    print("running in jupyter notebook")
    args = parser.parse_args([])
    args.ckpt = "delphi-m4/baseline/ckpt.pt"
    args.after_modality = True
    args.modalities = ["config/panel/blood.yaml"]
else:
    args = parser.parse_args()

args.modalities, args.panel_name = parse_modalities(args.modalities)
if args.fname is None:
    args.fname = "cindex"
    if args.panel_name is not None:
        args.fname += f"_{args.panel_name}"
    elif args.modalities is not None:
        args.fname += f"-modalities_{'_'.join(args.modalities)}"
print("args:")
pprint.pp(vars(args))


# +
ckpt = Path(DELPHI_CKPT_READ) / args.ckpt
model, ckpt_dict = load_ckpt(ckpt)
device = "cuda" if torch.cuda.is_available() else "cpu"

data_args = ckpt_dict["data_args"].copy()
data_args["subject_list"] = "participants/val_fold.bin"
data_args["stats_subject_list"] = ckpt_dict["data_args"]["subject_list"]
data_args["biomarker_require"] = "any"
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
onset_collator = EventTimeCollator(vocab_size=int(model_targets.max()) + 1)

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
        t0 = out_dict["age"]

        t0_off, logits_off = correct_time_offset(t0, t1, logits, offset=offset_days)
        dis_collator.step(tokens=x1, timesteps=t0_off, logits=logits_off)
        sex_collator.step(tokens=x0)
        onset_collator.step(tokens=x1.cpu(), timestep=t1.cpu())

dis_rates, dis_times = dis_collator.finalize()  # (N, V)
is_female = sex_collator.finalize()  # (N,)
onset_times, _ = onset_collator.finalize()
onset_times = torch.from_numpy(onset_times)  # (N, V)

# Restrict to time points after first occurrence of specified modalities
if args.after_modality and args.modalities is not None:
    bio_cutoff = np.full(len(ds), np.inf, dtype=np.float32)
    for mod_name in args.modalities:
        bio_first = ds.first_occurrence_times(mod_name)
        bio_cutoff = np.fmin(bio_cutoff, bio_first)
    # mask case events before the cutoff
    before_cutoff = dis_times.numpy() < bio_cutoff[:, None]
    dis_rates[torch.from_numpy(before_cutoff)] = torch.nan
    bio_cutoff = torch.from_numpy(bio_cutoff).to(device)
else:
    bio_cutoff = None

# Move tensors to device for Phase 2
dis_rates = dis_rates.to(device)
onset_times = onset_times.to(device)
is_female = is_female.to(device)

# +
concordance_collator = ConcordanceCollator(
    dis_rates=dis_rates,
    onset_times=onset_times,
    is_female=is_female,
    offset=offset_days,
    max_gap_days=args.max_gap * 365.25,
    cutoff=bio_cutoff,
)

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

        out_dict, _, _ = model(*batch_input)
        scores = out_dict["logits"].half()
        age = out_dict["age"]
        concordance_collator.step(age=age, scores=scores)

case_sex, case_tokens, total_pairs, concordant = concordance_collator.finalize()
# -

# Aggregate C-index per disease per sex
result = {}
for d_int in np.unique(case_tokens):
    d_mask = case_tokens == d_int
    icd = ds.detokenizer.get(int(d_int), str(d_int))
    result[icd] = {}
    for sex_label, sex_mask in [
        ("female", case_sex),
        ("male", ~case_sex),
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


pprint.pp(result.get("death", result.get(next(iter(result)), {})))


ckpt_write = Path(str(ckpt).replace(DELPHI_CKPT_READ, DELPHI_CKPT_WRITE))
os.makedirs(ckpt_write.parent, exist_ok=True)
out_path = ckpt_write.parent / f"{args.fname}.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=4)
print(f"Saved to {out_path}")
