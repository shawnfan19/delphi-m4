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

from delphi.data.ukb import NO_EVENT_TOKEN, UKBDataset
from delphi.env import DELPHI_CKPT_DIR
from delphi.experiment import eval_iter, load_ckpt, move_batch_to_device

# -

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

data_args = ckpt_dict["data_args"].copy()
data_args["subject_list"] = "participants/val_fold.bin"
data_args["perturb"] = False
data_args["deterministic"] = True
pprint.pp(data_args)

ds = UKBDataset(**data_args)

has_no_event = data_args.get("no_event_interval") is not None
print(f"has_no_event: {has_no_event}")

# build reverse tokenizer
idx_to_event = {v: k for k, v in ckpt_dict["tokenizer"].items()}

# determine which tokens are ignored (matching training logic in Delphi2M.forward)
ignored_tokens = {0}
if model.config.ignore_tokens is not None:
    ignored_tokens.update(model.config.ignore_tokens)


# +
total_size = len(ds)
batch_size = args.batch_size
it = tqdm(
    eval_iter(total_size=total_size, batch_size=batch_size),
    total=math.ceil(total_size / batch_size),
    leave=False,
)

# accumulators: per-participant NLL tracking
# keys: participant dataset index -> {scope -> {component -> (sum, count)}}
participant_nll = defaultdict(
    lambda: defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
)

# per-token-type NLL tracking (real events only)
token_nll_sum = defaultdict(float)
token_nll_count = defaultdict(int)

# global accumulators per scope per component
global_nll = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))

with torch.no_grad():
    for batch_idx in it:
        batch_input = ds.get_batch(batch_idx)
        batch_input = move_batch_to_device(batch_input, device=device)
        x0, t0, x1, t1 = batch_input

        # forward pass without targets to get outputs
        outputs, _, _ = model(x0, t0)

        # compute per-position loss
        loss_dict = model.loss(outputs=outputs, targets=x1, age=t0, targets_age=t1)

        # extract mask from loss if present (e.g. homo_cluster_poisson)
        loss_mask = loss_dict.pop("mask", None)

        # compute total NLL per position
        loss_components = list(loss_dict.keys())
        total_nll = sum(loss_dict[k] for k in loss_components)

        # build valid mask (same logic as Delphi2M.forward lines 438-443)
        valid_mask = torch.ones_like(x1, dtype=torch.bool)
        for k in ignored_tokens:
            valid_mask &= x1 != k
        if loss_mask is not None:
            valid_mask &= loss_mask

        # sub-masks
        no_event_mask = x1 == NO_EVENT_TOKEN
        real_event_mask = valid_mask & ~no_event_mask
        all_mask = valid_mask

        # define scopes to accumulate
        scopes = [("", real_event_mask)]
        if has_no_event:
            scopes.append(("_no_event", valid_mask & no_event_mask))
            scopes.append(("_all", all_mask))

        # move tensors to cpu for accumulation
        total_nll_cpu = total_nll.detach().cpu()
        component_nlls_cpu = {k: v.detach().cpu() for k, v in loss_dict.items()}

        for scope_suffix, scope_mask in scopes:
            scope_mask_cpu = scope_mask.detach().cpu()

            for b_local, b_global in enumerate(batch_idx):
                pos_mask = scope_mask_cpu[b_local]
                if pos_mask.sum() == 0:
                    continue

                nll_vals = total_nll_cpu[b_local][pos_mask]
                nll_sum = nll_vals.sum().item()
                nll_count = nll_vals.numel()

                participant_nll[int(b_global)][scope_suffix]["total"][0] += nll_sum
                participant_nll[int(b_global)][scope_suffix]["total"][1] += nll_count

                global_nll[scope_suffix]["total"][0] += nll_sum
                global_nll[scope_suffix]["total"][1] += nll_count

                # per-component
                if len(loss_components) > 1:
                    for comp_key in loss_components:
                        comp_vals = component_nlls_cpu[comp_key][b_local][pos_mask]
                        comp_sum = comp_vals.sum().item()
                        comp_count = comp_vals.numel()

                        participant_nll[int(b_global)][scope_suffix][comp_key][
                            0
                        ] += comp_sum
                        participant_nll[int(b_global)][scope_suffix][comp_key][
                            1
                        ] += comp_count

                        global_nll[scope_suffix][comp_key][0] += comp_sum
                        global_nll[scope_suffix][comp_key][1] += comp_count

            # per-token-type NLL (real events only, vectorized)
            if scope_suffix == "":
                targets_cpu = x1.detach().cpu()
                flat_mask = scope_mask_cpu.reshape(-1)
                flat_targets = targets_cpu.reshape(-1)[flat_mask]
                flat_nll = total_nll_cpu.reshape(-1)[flat_mask]
                for tok_id in flat_targets.unique().tolist():
                    tok_sel = flat_targets == tok_id
                    token_nll_sum[tok_id] += flat_nll[tok_sel].sum().item()
                    token_nll_count[tok_id] += tok_sel.sum().item()


# +
# aggregate metrics
metrics = {}

for scope_suffix, scope_data in global_nll.items():
    suffix = scope_suffix  # "", "_no_event", or "_all"

    # total NLL
    total_sum, total_count = scope_data["total"]
    metrics[f"mean_nll{suffix}"] = total_sum / total_count if total_count > 0 else None
    metrics[f"n_valid_tokens{suffix}"] = total_count

    # per-participant stats
    per_participant_means = []
    for pid, pid_scopes in participant_nll.items():
        if suffix in pid_scopes and "total" in pid_scopes[suffix]:
            s, c = pid_scopes[suffix]["total"]
            if c > 0:
                per_participant_means.append(s / c)

    per_participant_means = np.array(per_participant_means)
    if len(per_participant_means) > 0:
        metrics[f"mean_nll_per_participant{suffix}"] = float(
            np.mean(per_participant_means)
        )
        metrics[f"std_nll_per_participant{suffix}"] = float(
            np.std(per_participant_means)
        )
        metrics[f"median_nll_per_participant{suffix}"] = float(
            np.median(per_participant_means)
        )

    # per-component means
    if len(loss_components) > 1:
        for comp_key in loss_components:
            # strip "loss_" prefix for cleaner key names
            comp_name = comp_key.removeprefix("loss_")
            comp_sum, comp_count = scope_data.get(comp_key, (0.0, 0))
            if comp_count > 0:
                metrics[f"mean_nll_{comp_name}{suffix}"] = comp_sum / comp_count

            # per-participant component means
            comp_per_participant = []
            for pid, pid_scopes in participant_nll.items():
                if suffix in pid_scopes and comp_key in pid_scopes[suffix]:
                    s, c = pid_scopes[suffix][comp_key]
                    if c > 0:
                        comp_per_participant.append(s / c)
            if len(comp_per_participant) > 0:
                metrics[f"mean_nll_{comp_name}_per_participant{suffix}"] = float(
                    np.mean(comp_per_participant)
                )

metrics["n_participants"] = len(participant_nll)

# per-token-type NLL breakdown (real events only)
per_token_nll = {}
for tok_id, nll_sum in token_nll_sum.items():
    count = token_nll_count[tok_id]
    name = idx_to_event.get(tok_id, str(tok_id))
    per_token_nll[name] = round(nll_sum / count, 4) if count > 0 else None

metrics["per_token_nll"] = dict(sorted(per_token_nll.items()))

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
