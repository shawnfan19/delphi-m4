"""Trajectory-comparison metrics: a generated continuation vs the ground truth.

All functions are pure (no model dependency) and operate on *continuation*
events only: the caller passes a ``keep`` mask = continuation AND non-whitelist
(e.g. ``(mask == 2) & ~isin(marks, [0, 1])``). They are designed to compare one
ground-truth trajectory against K generated ones; pass the ground truth either
as a single row (broadcast, with a scalar horizon) or row-aligned to the
generations (``repeat_interleave``, with a per-row horizon).

Metrics:
- ``mark_overlap``  — recall of the ground-truth continuation marks (the "what").
- ``sequence_distance`` — unmarked sequence distance (the "when"); a
  self-anchored variant of EventFlow (arXiv:2410.07430).
"""

import torch
import torch.nn.functional as F


def pack_non_prompt(marks, age, keep):
    """Compact ``(B, L)`` trajectories to ``(B, L_np)`` keeping only ``keep``.

    Kept events are right-aligned with a *stable* sort (left-padding, matching
    the codebase convention), so their original (age-sorted) order is preserved;
    ``L_np`` is the max number of kept events over the batch. Returns ``marks_np``
    (padded with token 0), ``age_np`` (pad positions left as-is — overwritten via
    ``valid_np`` downstream), and ``valid_np`` ``(B, L_np)`` marking the real
    entries. Feed all three to the metric functions.
    """
    # ascending stable sort -> dropped positions first, kept last (in age order);
    # take the last L_np so kept events are right-aligned with padding on the left.
    order = torch.argsort(keep.to(torch.int8), dim=1, stable=True)
    L_np = int(keep.sum(1).max().clamp(min=1))
    sel = order[:, -L_np:]
    valid_np = torch.take_along_dim(keep, sel, 1)
    marks_np = torch.take_along_dim(marks, sel, 1).masked_fill(~valid_np, 0)
    age_np = torch.take_along_dim(age, sel, 1)
    return marks_np, age_np, valid_np


def multi_hot(marks_np, valid_np, vocab_size):
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
    gen = multi_hot(gen_marks_np, gen_valid_np, vocab_size)  # (N, V)
    truth = multi_hot(truth_marks_np, truth_valid_np, vocab_size)  # (M, V)
    inter = (gen & truth).sum(-1)
    denom = truth.sum(-1)
    return torch.where(denom > 0, inter / denom.clamp(min=1), torch.nan)


def _row_max(age, valid):
    """Per-row max of ``age`` over ``valid`` slots, as a ``(B, 1)`` column.

    Rows with no valid events get 0 — a placeholder; the caller NaN's those rows
    (it divides by their zero event count).
    """
    m = age.masked_fill(~valid, float("-inf")).max(-1, keepdim=True).values
    return torch.where(valid.any(-1, keepdim=True), m, torch.zeros_like(m))


def sequence_distance(gen_age_np, gen_valid_np, truth_age_np, truth_valid_np):
    """Unmarked sequence distance between a generation and the ground truth.

    A self-anchored variant of the EventFlow distance (arXiv:2410.07430): after an
    ascending sort, matched positions contribute ``Σ_{k≤n} |t_k^γ − t_k^η|`` and
    the longer sequence's unmatched tail is scored against the *shorter sequence's
    own last event* — each side is padded to the common length with its own
    maximum timestamp. ``n = max(n_gen, n_truth)``; positions beyond ``n`` are
    padding on both sides and contribute nothing.

    Unlike the published EventFlow distance — which scores the tail against a
    single fixed observation horizon ``T`` — anchoring each side to its own
    endpoint makes the distance self-contained (no external horizon) and
    symmetric: a generation running *past* the truth's last event is charged for
    overshooting (its tail vs the truth's endpoint), and one ending *before* it
    (e.g. an early predicted death) is charged for undershooting (the truth's tail
    vs the generation's endpoint). Events at a sequence's own endpoint are no
    longer free, so terminal-day-clustered truths are scored on how close the
    generation actually lands.

    Returns ``(N,)`` mean per-event distances ≥ 0 (comparable across trajectory
    lengths), or ``NaN`` where the *generation* has no valid events — an empty
    generation has no trajectory to score (cf. ``mark_overlap``).
    """
    g_max = _row_max(gen_age_np, gen_valid_np)
    h_max = _row_max(truth_age_np, truth_valid_np)
    # invalid slots, and the length padding below, take the sequence's own
    # endpoint, so after the sort each side "freezes" at its last event.
    g = torch.where(gen_valid_np, gen_age_np, g_max)
    h = torch.where(truth_valid_np, truth_age_np, h_max)
    L = max(g.shape[-1], h.shape[-1])
    g = _pad_T(g, L, g_max).sort(-1).values
    h = _pad_T(h, L, h_max).sort(-1).values
    n_gen, n_truth = gen_valid_np.sum(-1), truth_valid_np.sum(-1)
    n = torch.maximum(n_gen, n_truth)
    # only positions < n are meaningful; beyond that both sides are endpoint
    # padding (|g_max − h_max|), which must not count toward the distance.
    keep = torch.arange(L, device=g.device)[None, :] < n[:, None]
    dist = ((g - h).abs() * keep).sum(-1) / n.clamp(min=1)
    return dist.masked_fill(n_gen == 0, torch.nan)


def _pad_T(x, L, T):
    width = L - x.shape[-1]
    if width <= 0:
        return x
    return torch.cat([x, T.expand(x.shape[0], width)], dim=-1)
