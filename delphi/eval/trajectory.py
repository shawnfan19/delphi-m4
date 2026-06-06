"""Trajectory-comparison metrics: a generated continuation vs the ground truth.

All functions are pure (no model dependency) and operate on *continuation*
events only: the caller passes a ``keep`` mask = continuation AND non-whitelist
(e.g. ``(mask == 2) & ~isin(marks, [0, 1])``). They are designed to compare one
ground-truth trajectory against K generated ones; pass the ground truth either
as a single row (broadcast, with a scalar horizon) or row-aligned to the
generations (``repeat_interleave``, with a per-row horizon).

Metrics:
- ``mark_overlap``  — recall of the ground-truth continuation marks (the "what").
- ``sequence_distance`` — EventFlow unmarked sequence distance (the "when"),
  arXiv:2410.07430.
"""

import torch
import torch.nn.functional as F


def nonprompt(marks, age, keep):
    """Compact ``(B, L)`` trajectories to ``(B, L_np)`` keeping only ``keep``.

    Kept events are moved to the front with a *stable* sort, so their original
    (age-sorted) order is preserved; ``L_np`` is the max number of kept events
    over the batch. Returns ``marks_np`` (padded with token 0), ``age_np`` (pad
    positions left as-is — overwritten via ``valid_np`` downstream), and
    ``valid_np`` ``(B, L_np)`` marking the real entries. Feed all three to the
    metric functions.
    """
    order = torch.argsort((~keep).to(torch.int8), dim=1, stable=True)
    L_np = int(keep.sum(1).max().clamp(min=1))
    sel = order[:, :L_np]
    valid_np = torch.take_along_dim(keep, sel, 1)
    marks_np = torch.take_along_dim(marks, sel, 1).masked_fill(~valid_np, 0)
    age_np = torch.take_along_dim(age, sel, 1)
    return marks_np, age_np, valid_np


def _present(marks_np, valid_np, vocab_size):
    """``(B, V)`` boolean set-membership of the valid marks (slot 0 stays False)."""
    present = torch.zeros(marks_np.shape[0], vocab_size, device=marks_np.device)
    present.scatter_(1, marks_np.clamp(min=0), valid_np.to(present.dtype))
    return present > 0


def mark_overlap(
    gen_marks_np, gen_valid_np, truth_marks_np, truth_valid_np, vocab_size
):
    """Recall of the ground-truth continuation marks: ``|gen ∩ truth| / |truth|``.

    Set membership over the vocabulary; the truth broadcasts over generations
    (truth rows = 1, or row-aligned to N). Returns ``(N,)`` in ``[0, 1]``, NaN
    where the ground truth has no valid continuation marks.
    """
    gen = _present(gen_marks_np, gen_valid_np, vocab_size)  # (N, V)
    truth = _present(truth_marks_np, truth_valid_np, vocab_size)  # (M, V)
    inter = (gen & truth).sum(-1)
    denom = truth.sum(-1)
    return torch.where(denom > 0, inter / denom.clamp(min=1), torch.nan)


def _as_col(horizon, ref):
    T = torch.as_tensor(horizon, dtype=ref.dtype, device=ref.device)
    if T.ndim == 0:
        return T.reshape(1, 1)
    if T.ndim == 1:
        return T.reshape(-1, 1)
    return T


def sequence_distance(gen_age_np, gen_valid_np, truth_age_np, truth_valid_np, horizon):
    """EventFlow unmarked sequence distance (arXiv:2410.07430).

    ``d(γ, η) = Σ_{k≤n} |t_k^γ − t_k^η| + Σ_{k>n} (T − t_k)`` over the unmatched
    tail. Implemented by replacing invalid slots with the support bound ``T``
    (= ``horizon``) and right-padding both sequences to a common length with
    ``T``, then taking the element-wise ``|·|``: matched pairs give
    ``|t_k^γ − t_k^η|``, an unmatched event gives ``|T − t_k| = T − t_k``, and
    padding beyond both gives ``|T − T| = 0`` — exactly Eq. 12.

    ``horizon`` is a scalar (truth may be a single broadcast row) or a per-row
    tensor broadcastable to ``(N, 1)`` (truth row-aligned to the generations).
    Returns ``(N,)`` distances ≥ 0.
    """
    T = _as_col(horizon, gen_age_np)
    g = torch.where(gen_valid_np, gen_age_np, T).sort(-1).values  # valid asc, T tail
    h = torch.where(truth_valid_np, truth_age_np, T).sort(-1).values
    L = max(g.shape[-1], h.shape[-1])
    g = _pad_T(g, L, T)
    h = _pad_T(h, L, T)
    return (g - h).abs().sum(-1)


def _pad_T(x, L, T):
    width = L - x.shape[-1]
    if width <= 0:
        return x
    return torch.cat([x, T.expand(x.shape[0], width)], dim=-1)
