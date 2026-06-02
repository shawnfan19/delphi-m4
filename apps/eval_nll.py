# +
import argparse
import json
import math
import pprint
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm.autonotebook import tqdm

from delphi.data import Dataset
from delphi.data.transform import TokenTransform
from delphi.data.ukb import NO_EVENT_TOKEN, UKBReader
from delphi.env import DELPHI_CKPT_DIR
from delphi.experiment import eval_iter, load_ckpt, move_batch_to_device

# -


class NLLCollator:
    """Accumulates NLL statistics for a single masking scope."""

    def __init__(self, suffix: str = ""):
        self.suffix = suffix
        self.global_sums = defaultdict(float)  # comp_key -> sum
        self.global_counts = defaultdict(int)  # comp_key -> count
        # participant_id -> {comp_key -> [sum, count]}
        self.per_participant = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))

    def step(
        self,
        loss_dict: dict[str, torch.Tensor],
        mask: torch.Tensor,
        batch_idx: np.ndarray,
    ):
        mask_cpu = mask.detach().cpu()
        loss_cpu = {k: v.detach().cpu() for k, v in loss_dict.items()}
        total_nll = sum(loss_cpu.values())

        for b_local, b_global in enumerate(batch_idx):
            pos_mask = mask_cpu[b_local]
            if pos_mask.sum() == 0:
                continue

            pid = int(b_global)

            # total NLL
            nll_vals = total_nll[b_local][pos_mask]
            s, c = nll_vals.sum().item(), nll_vals.numel()
            self.global_sums["total"] += s
            self.global_counts["total"] += c
            self.per_participant[pid]["total"][0] += s
            self.per_participant[pid]["total"][1] += c

            # per-component (only if >1 component)
            if len(loss_cpu) > 1:
                for comp_key, comp_tensor in loss_cpu.items():
                    comp_vals = comp_tensor[b_local][pos_mask]
                    cs, cc = comp_vals.sum().item(), comp_vals.numel()
                    self.global_sums[comp_key] += cs
                    self.global_counts[comp_key] += cc
                    self.per_participant[pid][comp_key][0] += cs
                    self.per_participant[pid][comp_key][1] += cc

    def finalize(self) -> dict:
        sfx = self.suffix
        metrics = {}

        # global mean NLL
        total_count = self.global_counts["total"]
        metrics[f"mean_nll{sfx}"] = (
            self.global_sums["total"] / total_count if total_count > 0 else None
        )
        metrics[f"n_valid_tokens{sfx}"] = total_count

        # per-participant stats
        per_participant_means = []
        for pid_data in self.per_participant.values():
            s, c = pid_data["total"]
            if c > 0:
                per_participant_means.append(s / c)

        per_participant_means = np.array(per_participant_means)
        if len(per_participant_means) > 0:
            metrics[f"mean_nll_per_participant{sfx}"] = float(
                np.mean(per_participant_means)
            )
            metrics[f"std_nll_per_participant{sfx}"] = float(
                np.std(per_participant_means)
            )
            metrics[f"median_nll_per_participant{sfx}"] = float(
                np.median(per_participant_means)
            )

        # per-component breakdown
        comp_keys = [k for k in self.global_sums if k != "total"]
        for comp_key in comp_keys:
            comp_name = comp_key.removeprefix("loss_")
            comp_count = self.global_counts[comp_key]
            if comp_count > 0:
                metrics[f"mean_nll_{comp_name}{sfx}"] = (
                    self.global_sums[comp_key] / comp_count
                )

            comp_per_participant = []
            for pid_data in self.per_participant.values():
                if comp_key in pid_data:
                    s, c = pid_data[comp_key]
                    if c > 0:
                        comp_per_participant.append(s / c)
            if len(comp_per_participant) > 0:
                metrics[f"mean_nll_{comp_name}_per_participant{sfx}"] = float(
                    np.mean(comp_per_participant)
                )

        return metrics


class PerTokenNLLCollator:
    """Accumulates per-token-type NLL breakdown."""

    def __init__(self, idx_to_event: dict[int, str]):
        self.idx_to_event = idx_to_event
        self.token_nll_sum = defaultdict(float)
        self.token_nll_count = defaultdict(int)

    def step(
        self,
        total_nll: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ):
        mask_cpu = mask.detach().cpu()
        targets_cpu = targets.detach().cpu()
        nll_cpu = total_nll.detach().cpu()

        flat_mask = mask_cpu.reshape(-1)
        flat_targets = targets_cpu.reshape(-1)[flat_mask]
        flat_nll = nll_cpu.reshape(-1)[flat_mask]

        for tok_id in flat_targets.unique().tolist():
            tok_sel = flat_targets == tok_id
            self.token_nll_sum[tok_id] += flat_nll[tok_sel].sum().item()
            self.token_nll_count[tok_id] += tok_sel.sum().item()

    def finalize(self) -> dict:
        per_token_nll = {}
        for tok_id, nll_sum in self.token_nll_sum.items():
            count = self.token_nll_count[tok_id]
            name = self.idx_to_event.get(tok_id, str(tok_id))
            per_token_nll[name] = round(nll_sum / count, 4) if count > 0 else None
        return {"per_token_nll": dict(sorted(per_token_nll.items()))}


# +
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="path/to/ckpt.pt")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--fname", type=str, default=None)

if "ipykernel" in sys.modules:
    print("running in jupyter notebook")
    args = parser.parse_args([])
    args.ckpt = "tpp/hawkes_no_pad_365.25/ckpt.pt"
else:
    args = parser.parse_args()

print("args:")
pprint.pp(vars(args))

# +
ckpt_path = Path(DELPHI_CKPT_DIR) / args.ckpt
model, ckpt_dict = load_ckpt(ckpt_path)
device = next(model.parameters()).device

token_transform_args = ckpt_dict["token_transform_args"]
val_pids = UKBReader.participants("val")

reader = UKBReader()
token_transform = TokenTransform(**token_transform_args)
token_transform.describe()

ds = Dataset(reader=reader, pids=val_pids, token_transform=token_transform)

has_no_event = token_transform_args.get("no_event_interval") is not None
print(f"has_no_event: {has_no_event}")

# build reverse tokenizer
idx_to_event = {v: k for k, v in reader.tokenizer.items()}

# determine which tokens are ignored (matching training logic in Delphi2M.forward)
ignored_tokens = {0}
if model.config.ignore_tokens is not None:
    ignored_tokens.update(model.config.ignore_tokens)


# +
# instantiate collators
nll_collator = NLLCollator(suffix="")
nll_collators = [nll_collator]
if has_no_event:
    nll_collators.append(NLLCollator(suffix="_no_event"))
    nll_collators.append(NLLCollator(suffix="_all"))
token_collator = PerTokenNLLCollator(idx_to_event)

total_size = len(ds)
batch_size = args.batch_size
it = tqdm(
    eval_iter(total_size=total_size, batch_size=batch_size),
    total=math.ceil(total_size / batch_size),
    leave=False,
)

with torch.no_grad():
    for batch_idx in it:
        batch_input = ds.get_batch(batch_idx)
        batch_input = move_batch_to_device(batch_input, device=device)
        x0, t0, x1, t1 = batch_input

        # forward pass without targets to get outputs
        outputs, _, _ = model(x0, t0)

        # compute per-position loss (invalid positions are NaN)
        loss_out = model.loss(outputs=outputs, targets=x1, targets_age=t1, reduce=False)
        loss_dict = loss_out[0] if isinstance(loss_out, tuple) else loss_out
        total_nll = sum(loss_dict.values())

        # recover valid positions from the NaN pattern set inside loss()
        valid_mask = ~torch.isnan(total_nll)
        while valid_mask.dim() > x1.dim():
            valid_mask = valid_mask.any(dim=-1)

        # scope masks
        no_event_mask = x1 == NO_EVENT_TOKEN
        real_event_mask = valid_mask & ~no_event_mask
        scope_masks = [real_event_mask]
        if has_no_event:
            scope_masks.append(valid_mask & no_event_mask)
            scope_masks.append(valid_mask)

        # collator steps
        for collator, mask in zip(nll_collators, scope_masks):
            collator.step(loss_dict, mask, batch_idx)
        token_collator.step(total_nll, x1, real_event_mask)


# +
# aggregate metrics
metrics = {}
for collator in nll_collators:
    metrics.update(collator.finalize())
metrics["n_participants"] = len(nll_collator.per_participant)
metrics.update(token_collator.finalize())

# round scalar metrics for readability
for k, v in metrics.items():
    if isinstance(v, float):
        metrics[k] = round(v, 6)

pprint.pp(metrics)
# -

if args.fname is None:
    args.fname = "eval_nll"
out_path = ckpt_path.parent / f"{args.fname}.json"
with open(out_path, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"wrote {out_path}")
