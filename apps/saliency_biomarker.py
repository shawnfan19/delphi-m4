# +
import pprint
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from delphi.data import MultimodalDataset
from delphi.data.auto import multimodal_reader_cls
from delphi.data.transform import BiomarkerTransform, MultimodalPrompt, TokenTransform
from delphi.env import DELPHI_CKPT_DIR
from delphi.eval import BiomarkerCollator, EventTimeCollator
from delphi.experiment import CliConfig, load_ckpt, move_batch_to_device


# +
@dataclass(kw_only=True)
class TaskConfig(CliConfig):
    ckpt: str = "delphi-m4/delphi-m4/ckpt.pt"
    biomarker: str = "lipid"  # e.g. "lipid", "renal"
    subsample: None | int = None
    fname: None | str = None
    num_workers: int = 0


args = TaskConfig.from_cli()
args.print()

# +
ckpt = Path(DELPHI_CKPT_DIR) / args.ckpt
model, ckpt_dict = load_ckpt(ckpt)
model.eval()
device = next(model.parameters()).device

mod_name = args.biomarker.lower()
biomarker2idx = model.config.biomarker2idx
assert (
    mod_name in biomarker2idx
), f"biomarker {mod_name!r} not in ckpt biomarker2idx: {sorted(biomarker2idx)}"

reader_args = ckpt_dict["reader_args"]
pprint.pp(
    {
        "reader_args": reader_args,
        "token_transform_args": ckpt_dict["token_transform_args"],
        "biomarker_transform_args": ckpt_dict.get("biomarker_transform_args"),
    }
)

# dataset-aware: UKB on the cluster, AoU on the workbench. Honors DELPHI_DATASET
# (set in the dsub env), else auto-detects from the data dir.
ReaderCls = multimodal_reader_cls()
Biomarker = ReaderCls.biomarker_cls

# pass dict (not list) so reader uses the ckpt's index assignments
reader = ReaderCls(
    biomarkers=biomarker2idx or None,
    expansion_packs=reader_args["expansion_packs"],
)
token_transform = TokenTransform.from_ckpt(ckpt_dict)
biomarker_transform = BiomarkerTransform.from_ckpt(ckpt_dict)
if biomarker_transform is not None:
    biomarker_transform = biomarker_transform.replace(
        first_time_only=True, dropout=None, deterministic=True
    )

val_pids = ReaderCls.participants("val")
total_val = val_pids.size
val_pids = ReaderCls.filter_participants_with_biomarkers(
    val_pids, biomarkers=[mod_name], any=True
)
print(f"{val_pids.size} / {total_val} val pids (has {mod_name})")

# cutoff at the (first/only) measurement time of the target modality
cutoff = Biomarker.first_occurrence_times(mod_name, val_pids)
prompt_transform = MultimodalPrompt(
    prompt_age={int(pid): float(age) for pid, age in zip(val_pids, cutoff)},
    biomarker2idx=reader.biomarker2idx,
    append_no_event=False,
)

ds = MultimodalDataset(
    reader=reader,
    pids=val_pids,
    token_transform=token_transform,
    biomarker_transform=biomarker_transform,
    prompt_transform=prompt_transform,
)

model_targets = model.targets.to(device)
# exclude augmentation tokens (no_event + the dx anchor on tiebreak checkpoints) so
# they aren't attributed as diseases.
model_targets = model_targets[
    ~torch.isin(model_targets, model.augmentation_tokens.to(device))
]
target_idx = model_targets.cpu().tolist()
print(
    f"biomarker: {mod_name}, "
    f"n_features: {len(reader.biomarkers[mod_name].features)}, "
    f"n_targets: {len(model_targets)}"
)


# -
def saliency_forward(
    bio_x, *, model, x0, t0, bio_x_rest, bio_T, bio_M, mod_name, target_idx
):
    bio_x_dict = {**bio_x_rest, mod_name: bio_x}
    out, _, _ = model(x0, t0, biomarker=bio_x_dict, mod_age=bio_T, mod_idx=bio_M)
    return out["logits"][:, -1, target_idx]  # (B, n_targets)


# +
pids_list = []
jacobians_list = []
logits_list = []
event_collator = EventTimeCollator(model.config.vocab_size)
bio_collator = BiomarkerCollator()
cutoff_list = []
n = len(ds) if args.subsample is None else args.subsample

loader = DataLoader(
    Subset(ds, range(n)),  # type: ignore[arg-type]  # map-style, not a torch Dataset
    batch_size=1,
    num_workers=args.num_workers,
    collate_fn=ds.collate,
    prefetch_factor=2 if args.num_workers else None,
    persistent_workers=args.num_workers > 0,
)

for i, batch in enumerate(tqdm(loader, total=n)):
    batch = move_batch_to_device(batch, device)
    x0, t0, bio_X_dict, bio_t, bio_m, x1, t1 = batch

    out, _, _ = model(x0, t0, biomarker=bio_X_dict, mod_age=bio_t, mod_idx=bio_m)
    logits = out["logits"][:, -1, target_idx].detach().cpu().numpy()
    pid = ds.participants[i]

    # covariates from this trajectory; .cpu() since EventTimeCollator accumulates on CPU
    event_collator.step(x0.cpu(), t0.cpu())
    bio_collator.step(pid, bio_X_dict)
    cutoff_list.append(float(bio_t[bio_m == biomarker2idx[mod_name]].item()))

    bio_x = bio_X_dict[mod_name].float().detach()  # (n_meas, n_features)

    forward_func = partial(
        saliency_forward,
        model=model,
        x0=x0,
        t0=t0,
        bio_x_rest={m: v.detach() for m, v in bio_X_dict.items() if m != mod_name},
        bio_T=bio_t,
        bio_M=bio_m,
        mod_name=mod_name,
        target_idx=target_idx,
    )

    with torch.enable_grad():
        jac = torch.func.jacfwd(forward_func)(
            bio_x
        )  # (1, n_targets, n_meas, n_features)
    jac = jac[0]  # (n_targets, n_meas, n_features)
    jac = jac.reshape(jac.shape[0], -1)  # (n_targets, n_meas * n_features)
    jac = jac.T.detach().cpu().numpy()  # (n_meas * n_features, n_targets)

    # extinguish: NaN out targets already occurred in this person's history
    occurred = set(x0.ravel().cpu().tolist())
    extinguished = np.array([t in occurred for t in target_idx])
    jac[:, extinguished] = np.nan
    logits[:, extinguished] = -np.inf

    pids_list.append(int(pid))
    jacobians_list.append(jac.astype(np.float32))
    logits_list.append(logits.ravel().astype(np.float32))
# -


# +
pids = np.array(pids_list, dtype=np.int64)
jacobians = np.stack(jacobians_list)  # (N, n_features, n_targets)
logits = np.stack(logits_list)  # (N, n_targets)

# covariates gathered during the loop by the collators (row-aligned with `pids`).
age = (np.array(cutoff_list, dtype=np.float32) / 365.25).astype(np.float32)

# token presence = occurred in the prompt (non-NaN occurrence time); assemble the
# full base vocab, then drop padding/no_event and "male" (male+female == intercept).
occur_time, _ = event_collator.finalize()  # (N, vocab_size), NaN where absent
base_tok = reader.base_tokenizer
base_pairs = sorted(base_tok.items(), key=lambda x: x[1])
token_names = np.array([name for name, _ in base_pairs])
base_ids = np.array([tok for _, tok in base_pairs], dtype=np.int64)
token_matrix = (~np.isnan(occur_time[:, base_ids])).astype(np.uint8)
# drop redundant columns: padding/no_event, "male" (male+female == intercept), and
# the *_mid reference level of each ordinal lifestyle group.
drop_ids = {
    base_tok.get("padding", 0),
    base_tok.get("no_event", 1),
    base_tok.get("bmi_mid"),
    base_tok.get("alcohol_mid"),
    base_tok.get("smoking_mid"),
}
if "male" in base_tok:
    drop_ids.add(base_tok["male"])
keep = np.array([tok not in drop_ids for tok in base_ids])
token_matrix, token_names = token_matrix[:, keep], token_names[keep]

# biomarker context values: z-scored first-occurrence values back to raw units, in
# reader order; absent -> 0.
raw_values = biomarker_transform.untransform(bio_collator.finalize(pids))
bio_names = np.array(
    [f"{name}:{f}" for name, bm in reader.biomarkers.items() for f in bm.features]
)
bio_values = np.zeros((len(pids), bio_names.size), dtype=np.float32)
col = 0
for name, bm in reader.biomarkers.items():
    vals = raw_values.get(name)
    if vals is not None:
        bio_values[:, col : col + bm.n_features] = np.nan_to_num(vals, nan=0.0)
    col += bm.n_features

# jacobian axis labels, so the consumer resolves feature/target by name. feature
# names are qualified "modality:feature" so they match bio_names exactly (lets the
# consumer locate the value column from the feature alone, no modality arg).
feature_names = np.array(
    [f"{mod_name}:{f}" for f in reader.biomarkers[mod_name].features]
)
assert (
    len(feature_names) == jacobians.shape[1]
), "jacobian feature axis != n_features; the first_time_only=True assumption broke"
detokenizer = {v: k for k, v in ckpt_dict["tokenizer"].items()}
target_names = np.array([detokenizer[t] for t in target_idx])  # jac axis 2

fname = args.fname or f"saliency-{mod_name}"
out_path = ckpt.parent / f"{fname}.npz"
np.savez_compressed(
    out_path,
    pids=pids,
    jacobians=jacobians,
    logits=logits,
    age=age,
    token_matrix=token_matrix,
    token_names=token_names,
    bio_values=bio_values,
    bio_names=bio_names,
    feature_names=feature_names,
    target_names=target_names,
)

print(f"saved to {out_path}  (N={len(pids)}, jacobians={jacobians.shape})")
