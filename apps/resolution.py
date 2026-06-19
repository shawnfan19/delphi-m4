"""Generation-vs-ground-truth "resolution" metrics for memorization analysis.

For each prompt (a participant's history up to ``prompt_age``) we sample K
trajectories and, for both the K generations and the real ground-truth
continuation, compute the mark/time-decomposed *conditional* log-likelihood
(joint = marks + times). We then compare the ground truth to each generation
with two trajectory metrics — EventFlow sequence distance (the "when") and
mark-overlap recall (the "what") — keeping the full (N, K) arrays so best-of-K
(min distance / max overlap) and the gen-vs-GT likelihood contrast can be
computed downstream.

Reference driver: apps/forecast-m4.py. Requires a homo_poisson checkpoint.
"""

import pprint
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from delphi.data import MultimodalDataset
from delphi.data.transform import BiomarkerTransform, MultimodalPrompt, TokenTransform
from delphi.data.ukb import MultimodalUKBReader
from delphi.env import DELPHI_CKPT_READ, DELPHI_CKPT_WRITE
from delphi.eval.trajectory import mark_overlap, pack_non_prompt, sequence_distance
from delphi.experiment import GenerateConfig, eval_iter, load_ckpt, move_batch_to_device
from delphi.model.tpp import conditional_log_likelihood, tpp_dispatch
from delphi.model.transformer import generate

DEATH_TOKEN = 1269


@dataclass(kw_only=True)
class TaskConfig(GenerateConfig):
    fold: str = "train"  # which split to draw prompts from (memorization: train)
    n_repeats: int = 16  # K trajectories per prompt (divides default batch_size 512)
    subsample: None | int = (
        None  # keep the first N participants (post-filter); None = all
    )
    fname: None | str = None

    def __post_init__(self):
        if self.prompt_age is None:
            self.prompt_age = "recruitment"
        if not self.fname:
            self.fname = f"resolution_{self.fold}_K{self.n_repeats}"


args = TaskConfig.from_cli()
args.print()

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = Path(DELPHI_CKPT_READ) / args.ckpt
model, ckpt_dict = load_ckpt(ckpt)
assert model.config.block_size is None
assert model.config.loss == "homo_poisson", (
    f"resolution needs a homo_poisson checkpoint (log_p_marks/log_p_times); "
    f"got loss={model.config.loss!r}"
)
vocab_size = model.config.vocab_size
# augmentation tokens (no_event + the dx anchor on tiebreak ckpts): real model
# outputs kept in the LL but NOT diseases, so excluded from the comparison metrics.
augmentation_tokens = model.augmentation_tokens.to(device)

# ---- reader / transforms (mirror forecast-m4) ----
reader_args = ckpt_dict["reader_args"]
reader = MultimodalUKBReader(
    biomarkers=model.config.biomarker2idx or None,
    expansion_packs=reader_args["expansion_packs"],
)
token_transform = TokenTransform.from_ckpt(ckpt_dict)
biomarker_transform = BiomarkerTransform.from_ckpt(ckpt_dict)
if biomarker_transform is not None:
    biomarker_transform = biomarker_transform.replace(dropout=None)

# ---- cohort + prompt cutoff ----
pids = MultimodalUKBReader.participants(args.fold)
prompt_age_arg = args.prompt_age
if prompt_age_arg == "recruitment":
    cutoff_age = reader.recruitment_times(pids)
    valid = ~np.isnan(cutoff_age)
    pids, cutoff_age = pids[valid], cutoff_age[valid]
    print(f"{pids.size} participants with recruitment times")
else:
    cutoff_age = np.full(pids.size, float(prompt_age_arg) * 365.25, dtype=np.float32)

# keep only participants with a ground-truth continuation: exit_times is the last
# token's age, so exit > cutoff <=> at least one token after the prompt cutoff.
followup = reader.exit_times(pids) > cutoff_age
pids, cutoff_age = pids[followup], cutoff_age[followup]
print(f"{pids.size} participants with ground truth after the prompt cutoff")

if args.subsample:
    pids, cutoff_age = pids[: args.subsample], cutoff_age[: args.subsample]

prompt_age_map: Any = (
    {int(p): float(a) for p, a in zip(pids, cutoff_age.tolist())}
    if prompt_age_arg == "recruitment"
    else float(prompt_age_arg) * 365.25
)

prompt_transform = MultimodalPrompt(
    prompt_age=prompt_age_map,
    biomarker2idx=reader.biomarker2idx,
    append_no_event=True,  # anchor the prompt with a no-event at the cutoff
)
ds = MultimodalDataset(
    reader=reader,
    pids=pids,
    token_transform=token_transform,
    biomarker_transform=biomarker_transform,
    prompt_transform=prompt_transform,
)

# ---- main loop: K generations per prompt ----
K = args.n_repeats
assert args.batch_size % K == 0, "batch_size must be a multiple of n_repeats"
eff = args.batch_size // K

out = defaultdict(list)

it = eval_iter(total_size=len(ds), batch_size=eff)
pbar = tqdm(it, total=int(np.ceil(len(ds) / eff)))
for batch_idx in pbar:
    b = batch_idx.shape[0]  # prompts this batch (< eff on the last batch)
    # generation prompts: K identical copies of each prompt (its X1/T1 unused).
    X0, T0, bioX, bioT, bioM, _, _ = move_batch_to_device(
        ds.get_batch(np.repeat(batch_idx, K)), device=device
    )
    # ground truth: a fresh un-repeated batch. get_batch is deterministic, and the
    # per-row biomarker structure (bio_X_dict is flattened across rows) only stays
    # correct when fetched at the right batch size — never recover it by slicing the
    # K-repeated batch.
    _, _, gtbioX, gtbioT, gtbioM, X1, T1 = move_batch_to_device(
        ds.get_batch(batch_idx), device=device
    )
    cut = torch.as_tensor(cutoff_age[batch_idx], dtype=T0.dtype, device=device)  # (b,)
    cut_gen = cut.repeat_interleave(K)  # (b*K,) aligned to the generation rows
    T_gt = T1.max(dim=1).values  # (b,) GT's last timestamp: gen cap + distance bound
    T_gen = T_gt.repeat_interleave(K)  # (b*K,) aligned to the generation rows

    # ---- sample K trajectories per prompt ----
    gen_idx, gen_age, stats = generate(
        model=model,
        idx=X0,
        age=T0,
        max_age=T_gen,
        termination_tokens=[DEATH_TOKEN],
        stop_at_block_size=False,
        cached=True,
        censor=False,  # keep the overflow event raw (no fabricated no-event marker)
        biomarker=bioX,
        mod_age=bioT,
        mod_idx=bioM,
    )
    pbar.set_postfix(
        {"n_gen": stats["n_gen"].mean(), "n_pmt": stats["n_prompt"].mean()}
    )

    # LL keep = continuation (age > cutoff); includes no-event tokens (real model
    # output). Padding (age = -1e4) and prompt (age <= cutoff) fall out naturally.
    keep_gen = gen_age > cut_gen[:, None]
    keep_gt = T1 > cut[:, None]

    # ---- conditional LL of the generations (continuation only) ----
    with torch.no_grad():
        gout, _, _ = model(gen_idx, gen_age, biomarker=bioX, mod_age=bioT, mod_idx=bioM)
    gen_tpp = tpp_dispatch(model, gout)
    gll = conditional_log_likelihood(
        gen_tpp, gen_idx, gen_age, keep=keep_gen, reduce="sum"
    )

    # ---- conditional LL of the ground truth (full un-repeated trajectory) ----
    with torch.no_grad():
        tout, _, _ = model(X1, T1, biomarker=gtbioX, mod_age=gtbioT, mod_idx=gtbioM)
    gt_tpp = tpp_dispatch(model, tout)
    tll = conditional_log_likelihood(gt_tpp, X1, T1, keep=keep_gt, reduce="sum")

    # ---- comparison metrics: GT vs each of its K generations ----
    # metrics compare real diseases only — drop augmentation tokens (no_event + dx;
    # both kept in the LL above) so the dx anchor can't inflate overlap/seq-distance
    # or the gt_n_real length stratifier.
    mgen = keep_gen & ~torch.isin(gen_idx, augmentation_tokens)
    mgt = keep_gt & ~torch.isin(X1, augmentation_tokens)
    gm, ga, gv = pack_non_prompt(gen_idx, gen_age, mgen)  # (b*K, .)
    tm, ta, tv = pack_non_prompt(X1, T1, mgt)  # (b, .)
    # per-prompt GT continuation event count (before the K repeat), kept so plots
    # can stratify the metrics by trajectory length. With first-occurrence data
    # (marks never repeat; no-event already dropped by mgt) this equals the number
    # of distinct marks — i.e. both the seq-distance event count and the
    # mark-overlap recall denominator.
    gt_n_real = tv.sum(1)  # (b,)
    tm, ta, tv = (z.repeat_interleave(K, 0) for z in (tm, ta, tv))  # align to gens
    overlap = mark_overlap(gm, gv, tm, tv, vocab_size)  # (b*K,)
    seq_dist = sequence_distance(ga, gv, ta, tv)  # (b*K,); self-anchored, no horizon

    # ---- collect (reshape per-gen quantities to (b, K)) ----
    r = lambda x: x.detach().reshape(b, K).cpu().numpy()
    out["pids"].append(pids[batch_idx])
    out["gen_ll_m"].append(r(gll["marks"]))
    out["gen_ll_t"].append(r(gll["times"]))
    out["gen_ll"].append(r(gll["joint"]))
    out["gen_n_events"].append(r(gll["n_events"]))
    out["seq_dist"].append(r(seq_dist))
    out["overlap"].append(r(overlap))
    out["ll_m"].append(tll["marks"].detach().cpu().numpy())
    out["ll_t"].append(tll["times"].detach().cpu().numpy())
    out["ll"].append(tll["joint"].detach().cpu().numpy())
    out["n_events"].append(tll["n_events"].detach().cpu().numpy())
    out["gt_n_real"].append(gt_n_real.detach().cpu().numpy())
    # horizon = length of the comparison window (prompt cutoff -> GT's last event,
    # in days); short windows mechanically compress the sequence distance.
    out["gt_horizon"].append((T_gt - cut).detach().cpu().numpy())
    out["pmt_n_events"].append(stats["n_prompt"].reshape(b, K)[:, 0])

out = {k: np.concatenate(v, axis=0) for k, v in out.items()}

path = Path(DELPHI_CKPT_WRITE) / Path(args.ckpt).parent / f"{args.fname}.npz"
path.parent.mkdir(parents=True, exist_ok=True)
np.savez(path, **out)
print(f"wrote {path}  ({out['pids'].size} prompts x {K} generations)")
pprint.pp({k: v.shape for k, v in out.items()})
