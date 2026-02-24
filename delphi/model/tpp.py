import torch
import torch.nn as nn
from torch.nn import functional as F

from delphi.model.utils import self_terminate


class HawkesHead(nn.Module):

    def __init__(self, n_embd: int, vocab_size: int, n_bins: int = 20):
        super().__init__()
        self.proj_alpha = nn.Linear(n_embd, vocab_size)
        self.proj_beta = nn.Linear(n_embd, vocab_size)
        self.log_mu = nn.Parameter(torch.full((n_bins, vocab_size), -5.0))

    def forward(self, x):

        param_alpha = F.softplus(self.proj_alpha(x))
        param_beta = F.softplus(self.proj_beta(x))

        return {
            "alpha": param_alpha,
            "beta": param_beta,
            "mu": F.softplus(self.log_mu),  # (n_bins, V) — age baseline
        }


def bin_mu_compensator(
    mu: torch.Tensor, bin_size: float, age: torch.Tensor, targets_age: torch.Tensor
):

    n_bins = mu.shape[0]
    bin_starts = (
        torch.arange(n_bins, device=age.device, dtype=age.dtype) * bin_size
    )  # (n_bins,)
    bin_ends = bin_starts + bin_size

    overlap = torch.clamp(
        torch.minimum(targets_age.unsqueeze(-1), bin_ends)
        - torch.maximum(age.unsqueeze(-1), bin_starts),
        min=0.0,
    )

    mu_integral = overlap @ mu  # (B, L, V) — ∫ μ_v(bin(τ)) dτ per event type

    return mu_integral


def nll_hawkes(
    mu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    age: torch.Tensor,
    idx: torch.Tensor,
    targets_age: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    NLL for Hawkes intensity: λ_v(τ) = α_v · exp(-β_v · (τ - t_i))

    Intensity spikes at t_i (previous event) and decays toward next event.

    Args:
        alpha: Excitation amplitude, shape (B, L, V), positive
        beta: Decay rate, shape (B, L, V), positive
        age: Interval start t_i in days, shape (B, L)
        targets_age: Interval end t_{i+1} in days, shape (B, L)
        targets: Event type indices, shape (B, L)

    Returns:
        NLL tensor of shape (B, L)
    """
    n_bins = mu.shape[0]
    bin_size = 5.0
    delta_t = targets_age - age

    # ================================================================
    # Part 1: log λ_k(t_{i+1})
    # ================================================================

    alpha_k = torch.gather(alpha, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    beta_k = torch.gather(beta, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    decay_k = alpha_k * torch.exp(-beta_k * delta_t)

    bin_idx = torch.clamp(
        (targets_age / bin_size).long(), min=0, max=n_bins - 1
    )  # (B, L)
    mu_k = mu[bin_idx, targets]  # (B, L)

    # log(α · exp(-β · Δt)) = log(α) - β · Δt
    part1 = torch.log(decay_k + mu_k)

    # ================================================================
    # Part 2: -∫_{t_i}^{t_{i+1}} Σ_v λ_v(τ) dτ
    # ================================================================

    delta_t_exp = delta_t.unsqueeze(-1)

    mu_integral = bin_mu_compensator(
        mu=mu, bin_size=bin_size, age=age, targets_age=targets_age
    )
    # ∫ α·exp(-β·(τ-t_i)) dτ = (α/β)·[1 - exp(-β·Δt)]
    # integral = (alpha / (beta + eps)) * (1.0 - torch.exp(-beta * delta_t_exp))
    integral = (alpha / beta) * (-torch.expm1(-beta * delta_t_exp))

    compensator = integral + mu_integral
    compensator = self_terminate(
        idx=idx,
        estimator=compensator,
        terminate_except=torch.tensor([1], device=idx.device),
        fill_val=0,
    )

    part2 = -compensator.sum(dim=-1)

    return -(part1 + part2)
