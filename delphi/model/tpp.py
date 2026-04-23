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
    compensator, _ = self_terminate(
        idx=idx,
        estimator=compensator,
        terminate_except=torch.tensor([1], device=idx.device),
        fill_val=0,
    )

    part2 = -compensator.sum(dim=-1)

    return -(part1 + part2)


class NeuralTPPHead(nn.Module):

    def __init__(
        self,
        n_embd: int,
        vocab_size: int,
        time_encoder: nn.Module,
        n_integrate_grid: int = 20,
        self_terminate: bool = True,
        self_terminate_except: list[int] | None = None,
        time_unit: float = 1.0,
    ):
        super().__init__()
        self.time_encoder = time_encoder
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.GELU(),
            nn.Linear(n_embd, vocab_size),
        )
        self.n_integrate_grid = n_integrate_grid
        self.self_terminate = self_terminate
        self.time_unit = time_unit
        self.register_buffer(
            "terminate_except",
            torch.tensor(
                self_terminate_except if self_terminate_except is not None else [],
                dtype=torch.long,
            ),
        )

    def forward(self, h: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, L, n_embd) transformer hidden states
            delta_t: (B, L) or (B, L, G) time since preceding event (days)

        Returns:
            Raw pre-softplus values, shape (B, L, V) or (B, L, G, V)
        """
        has_grid = delta_t.dim() > h.dim() - 1
        if has_grid:
            # delta_t is (B, L, G) — flatten to (B, L*G) for AgeEncoding (expects 3D)
            B, L, G = delta_t.shape
            dt_flat = delta_t.reshape(B, L * G)
            time_emb = self.time_encoder(dt_flat.unsqueeze(-1))  # (B, L*G, n_embd)
            time_emb = time_emb.reshape(B, L, G, -1)  # (B, L, G, n_embd)
            h = h.unsqueeze(-2)  # (B, L, 1, n_embd) for broadcast
        else:
            time_emb = self.time_encoder(delta_t.unsqueeze(-1))  # (B, L, n_embd)
        return self.net(h + time_emb)

    def nll(
        self,
        h: torch.Tensor,
        targets: torch.Tensor,
        age: torch.Tensor,
        targets_age: torch.Tensor,
        idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        NLL for neural TPP: λ_v(τ) = softplus(net(h + time_encode(τ - t_i)))_v

        Compensator computed via trapezoidal numerical integration.

        Args:
            h: (B, L, n_embd) transformer hidden states
            targets: (B, L) next event type indices
            age: (B, L) event timestamps in days
            targets_age: (B, L) next event timestamps in days
            idx: (B, L) input token indices (for self-termination)

        Returns:
            NLL tensor of shape (B, L)
        """
        eps = 1e-8
        delta_t = targets_age - age  # (B, L), in days

        # ================================================================
        # Part 1: log λ_k(t_{i+1})
        # ================================================================
        raw = self(h, delta_t)  # (B, L, V)
        intensity = F.softplus(raw)
        lam_k = torch.gather(intensity, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        log_lam_k = torch.log(lam_k + eps)

        # ================================================================
        # Part 2: compensator via numerical integration
        # ================================================================
        t_unit = torch.linspace(
            0.0, 1.0, self.n_integrate_grid, device=h.device, dtype=h.dtype
        )  # (G,)
        grid_times = t_unit * delta_t.unsqueeze(-1)  # (B, L, G)

        raw_grid = self(h, grid_times)  # (B, L, G, V)
        intensity_grid = F.softplus(raw_grid)  # (B, L, G, V)

        if self.self_terminate:
            # Expand self_terminate mask to grid: reshape intensity_grid to (B, L*G, V),
            # apply, then reshape back. We tile idx across grid points.
            B, L, G, V = intensity_grid.shape
            # self_terminate expects (B, L, V) with idx (B, L)
            # We replicate idx for each grid point and flatten
            intensity_flat = intensity_grid.reshape(B, L * G, V)
            idx_expanded = idx.unsqueeze(-1).expand(B, L, G).reshape(B, L * G)
            intensity_flat, _ = self_terminate(
                idx=idx_expanded,
                estimator=intensity_flat,
                terminate_except=self.terminate_except,
                fill_val=0,
            )
            intensity_grid = intensity_flat.reshape(B, L, G, V)

        total_intensity = intensity_grid.sum(dim=-1)  # (B, L, G)
        compensator = (
            torch.trapezoid(total_intensity, grid_times, dim=-1) / self.time_unit
        )  # (B, L)

        # ================================================================
        # NLL = -log λ_k + compensator
        # ================================================================
        return -log_lam_k + compensator
