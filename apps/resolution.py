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
from delphi.eval.trajectory import mark_overlap, nonprompt, sequence_distance
from delphi.experiment import GenerateConfig, eval_iter, load_ckpt, move_batch_to_device
from delphi.model.tpp import conditional_log_likelihood, tpp_dispatch
from delphi.model.transformer import generate

WHITELIST = [0, 1]  # pad, no-event — excluded from both metrics and the LL
DEATH_TOKEN = 1269


@dataclass(kw_only=True)
class TaskConfig(GenerateConfig):
    fold: str = "train"  # which split to draw prompts from (memorization: train)
    # horizon in years (number), or "gt" to match each participant's
    # ground-truth last continuation-event age.
    horizon: Any = "gt"
    n_repeats: int = 16  # K trajectories per prompt (divides default batch_size 512)
    fname: None | str = None

    def __post_init__(self):
        if self.prompt_age is None:
            self.prompt_age = "recruitment"
        if not self.fname:
            h = self.horizon if isinstance(self.horizon, str) else f"{self.horizon}y"
            self.fname = f"resolution_{self.fold}_{h}_K{self.n_repeats}"


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
whitelist = torch.tensor(WHITELIST, device=device)

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
    rec = reader.recruitment_times(pids)
    valid = ~np.isnan(rec)
    pids, cutoff_age = pids[valid], rec[valid]
    prompt_age_map: Any = {int(p): float(a) for p, a in zip(pids, cutoff_age.tolist())}
    print(f"{pids.size} participants with recruitment times")
else:
    cutoff_age = np.full(pids.size, float(prompt_age_arg) * 365.25, dtype=np.float32)
    prompt_age_map = float(prompt_age_arg) * 365.25

if args.subsample:
    pids, cutoff_age = pids[: args.subsample], cutoff_age[: args.subsample]

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
is_gt_horizon = isinstance(args.horizon, str)

out = {
    k: []
    for k in (
        "pids",
        "gen_ll_m",
        "gen_ll_t",
        "gen_ll",
        "gt_ll_m",
        "gt_ll_t",
        "gt_ll",
        "pmt_n_events",
        "gt_n_events",
        "gen_n_events",
        "seq_dist",
        "overlap",
    )
}


def horizon_age(X1, T1, cut):
    """Per-row support bound T (days), shape (B,)."""
    if not is_gt_horizon:
        return cut + float(args.horizon) * 365.25
    cont = (T1 > cut[:, None]) & ~torch.isin(X1, whitelist)
    last = torch.where(cont, T1, torch.full_like(T1, -1e4)).max(dim=1).values
    return torch.where(last > cut, last, cut)  # fallback: no continuation -> cut


it = eval_iter(total_size=len(ds), batch_size=eff)
pbar = tqdm(it, total=int(np.ceil(len(ds) / eff)))
for batch_idx in pbar:
    b = batch_idx.shape[0]  # prompts this batch (< eff on the last batch)
    rep_idx = np.repeat(batch_idx, K)
    X0, T0, bioX, bioT, bioM, X1, T1 = move_batch_to_device(
        ds.get_batch(rep_idx), device=device
    )
    cut = torch.as_tensor(
        cutoff_age[rep_idx], dtype=T0.dtype, device=device
    )  # (eff*K,)
    T_age = horizon_age(X1, T1, cut)  # (eff*K,) support bound per row

    # ---- sample K trajectories per prompt ----
    gen_idx, gen_age, stats = generate(
        model=model,
        idx=X0,
        age=T0,
        max_age=T_age,
        termination_tokens=[DEATH_TOKEN],
        stop_at_block_size=False,
        cached=True,
        biomarker=bioX,
        mod_age=bioT,
        mod_idx=bioM,
    )
    mask = stats["mask"]
    pbar.set_postfix(
        {"n_gen": stats["n_gen"].mean(), "n_pmt": stats["n_prompt"].mean()}
    )

    # ---- conditional LL of the generations (continuation only) ----
    with torch.no_grad():
        gout, _, _ = model(gen_idx, gen_age, biomarker=bioX, mod_age=bioT, mod_idx=bioM)
    gen_tpp = tpp_dispatch(model, gout)
    keep_gen = (mask == 2) & ~torch.isin(gen_idx, whitelist)
    gll = conditional_log_likelihood(
        gen_tpp, gen_idx, gen_age, keep=keep_gen, reduce="sum"
    )

    # ---- conditional LL of the ground truth (full trajectory, unique prompts) ----
    u = slice(None, None, K)
    X1u, T1u, cutu, Tu = X1[u], T1[u], cut[u], T_age[u]
    bioXu = {k: v[u] for k, v in bioX.items()}
    with torch.no_grad():
        tout, _, _ = model(X1u, T1u, biomarker=bioXu, mod_age=bioT[u], mod_idx=bioM[u])
    gt_tpp = tpp_dispatch(model, tout)
    # GT continuation = events in (cutoff, T], non-whitelist (window matches generation)
    keep_gt = (T1u > cutu[:, None]) & (T1u <= Tu[:, None]) & ~torch.isin(X1u, whitelist)
    tll = conditional_log_likelihood(gt_tpp, X1u, T1u, keep=keep_gt, reduce="sum")

    # ---- comparison metrics: GT vs each of its K generations ----
    gm, ga, gv = nonprompt(gen_idx, gen_age, keep_gen)  # (eff*K, .)
    tm, ta, tv = nonprompt(X1u, T1u, keep_gt)  # (eff, .)
    tm, ta, tv = (z.repeat_interleave(K, 0) for z in (tm, ta, tv))  # align to gens
    overlap = mark_overlap(gm, gv, tm, tv, vocab_size)  # (eff*K,)
    seq_dist = sequence_distance(ga, gv, ta, tv, T_age[:, None])  # (eff*K,)

    # ---- collect (reshape per-gen quantities to (eff, K)) ----
    r = lambda x: x.detach().reshape(b, K).cpu().numpy()
    out["pids"].append(pids[batch_idx])
    out["gen_ll_m"].append(r(gll["marks"]))
    out["gen_ll_t"].append(r(gll["times"]))
    out["gen_ll"].append(r(gll["joint"]))
    out["gen_n_events"].append(r(gll["n_events"]))
    out["seq_dist"].append(r(seq_dist))
    out["overlap"].append(r(overlap))
    out["gt_ll_m"].append(tll["marks"].detach().cpu().numpy())
    out["gt_ll_t"].append(tll["times"].detach().cpu().numpy())
    out["gt_ll"].append(tll["joint"].detach().cpu().numpy())
    out["gt_n_events"].append(tll["n_events"].detach().cpu().numpy())
    out["pmt_n_events"].append(stats["n_prompt"][u])

out = {k: np.concatenate(v, axis=0) for k, v in out.items()}

path = Path(DELPHI_CKPT_WRITE) / Path(args.ckpt).parent / f"{args.fname}.npz"
path.parent.mkdir(parents=True, exist_ok=True)
np.savez(path, **out)
print(f"wrote {path}  ({out['pids'].size} prompts x {K} generations)")
pprint.pp({k: v.shape for k, v in out.items()})
