import torch

from .homo_poisson import HomoPoissonTPP
from .neural import NeuralODETPP, NeuralTPP
from .sets import DynamicDPPTPP


def tpp_dispatch(model, outputs):
    loss = model.config.loss
    device = outputs["h"].device
    if loss == "homo_poisson":
        return HomoPoissonTPP(
            hidden_states=outputs["h"],
            logits=outputs["logits"],
            tokens=outputs["idx"],
            timesteps=outputs["age"],
            terminate_except=torch.tensor(
                model.config.self_terminate_except, device=device
            ),
            time_unit=model.config.time_unit,
        )
    if loss == "neural_tpp":
        return NeuralTPP(
            hidden_states=outputs["h"],
            intensity_func=model.neural_tpp_head,
            timesteps=outputs["age"],
            tokens=outputs["idx"],
            n_grid=model.config.n_integrate_grid,
            integrate_method=model.config.integrate_method,
            time_unit=model.config.time_unit,
        )
    if loss == "neural_ode":
        return NeuralODETPP(
            ode=model.neural_head,
            hidden_states=outputs["h"],
            timesteps=outputs["age"],
            tokens=outputs["idx"],
            time_unit=model.config.time_unit,
            method=model.config.ode_method,
            step_size=model.config.ode_step_size,
        )
    if loss == "dynamic_dpp":
        cfg = model.config
        # set is over real disease tokens: drop padding (0), augmentation
        # (no-event / dx anchor) and ignore tokens (sex / lifestyle).
        exclude = sorted(
            {0, *(cfg.augmentation_tokens or []), *(cfg.ignore_tokens or [])}
        )
        return DynamicDPPTPP(
            hidden_states=outputs["h"],
            head=model.dpp_head,
            embedding=model.transformer.wte.weight,
            timesteps=outputs["age"],
            tokens=outputs["idx"],
            exclude=torch.tensor(exclude, device=device),
            terminate_except=torch.tensor(cfg.self_terminate_except, device=device),
            time_unit=cfg.time_unit,
        )
    raise ValueError(f"tpp_dispatch: unsupported model.config.loss={loss!r}")


def conditional_log_likelihood(
    tpp,
    x1: torch.Tensor,
    t1: torch.Tensor,
    keep: None | torch.Tensor = None,
    reduce: None | str = "sum",
):
    """Mark/time-decomposed conditional log-likelihood of events ``(x1, t1)``.

    Scores each event ``x1`` occurring at time ``t1`` against the history
    encoded in ``tpp`` (built via :func:`tpp_dispatch`), splitting the per-event
    log-likelihood into its mark ("what") and time ("when") terms. Operates on
    an already-built TPP only: it does not run the model, and it is agnostic to
    how ``keep`` was derived — the caller owns that policy (e.g. scoring the
    continuation of a prompted generation, dropping ignore-tokens).

    Args:
        tpp: a TPP exposing ``log_p_marks(x1, t1)`` and ``log_p_times(t1)``
            (currently :class:`HomoPoissonTPP`).
        x1: (B, L) event marks (tokens) — the shape the TPP log-likelihood
            methods accept (2-D query; fold any sample axis into the batch).
        t1: (B, L) event times (ages).
        keep: optional boolean mask (B, L) selecting which events to score.
            Excluded positions — and any the TPP marks invalid (NaN: no
            strict-before history) — do not enter the reduction.
        reduce: ``"sum"`` or ``"mean"`` over the event axis (``dim=-1``), giving
            one value per trajectory ``(B,)``; or ``None`` to return the
            per-event terms. Trajectories with zero scored events reduce to NaN.

    Returns:
        dict with ``"marks"``, ``"times"`` and ``"joint"`` (== marks + times).
        For ``"sum"``/``"mean"`` these are reduced over the last axis and an
        integer ``"n_events"`` is included; for ``reduce=None`` they are the
        per-event tensors with excluded positions set to NaN (plus the boolean
        ``"keep"`` actually used).
    """
    lp_marks = tpp.log_p_marks(x1, t1)
    lp_times = tpp.log_p_times(t1)

    if keep is None:
        keep = torch.ones_like(t1, dtype=torch.bool)
    valid = keep.bool() & ~torch.isnan(lp_marks) & ~torch.isnan(lp_times)

    if reduce is None:
        drop = ~valid
        return {
            "marks": lp_marks.masked_fill(drop, torch.nan),
            "times": lp_times.masked_fill(drop, torch.nan),
            "joint": (lp_marks + lp_times).masked_fill(drop, torch.nan),
            "keep": valid,
        }
    if reduce not in ("sum", "mean"):
        raise ValueError(f"reduce must be 'sum', 'mean' or None; got {reduce!r}")

    zeros = torch.zeros((), dtype=lp_marks.dtype, device=lp_marks.device)
    marks = torch.where(valid, lp_marks, zeros).sum(dim=-1)
    times = torch.where(valid, lp_times, zeros).sum(dim=-1)
    counts = valid.sum(dim=-1)
    if reduce == "mean":
        denom = counts.clamp(min=1)
        marks, times = marks / denom, times / denom
    empty = counts == 0
    marks = marks.masked_fill(empty, torch.nan)
    times = times.masked_fill(empty, torch.nan)
    return {"marks": marks, "times": times, "joint": marks + times, "n_events": counts}
