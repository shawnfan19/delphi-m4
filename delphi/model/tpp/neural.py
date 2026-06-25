import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

from delphi.model.utils import have_occurred, nearest_input_pos


class NeuralIntensity(nn.Module):
    """
    Parameterized intensity λ_v(τ) = softplus(net(h + time_encode(τ - t_i)))_v.

    Returns non-negative intensities, clamped at 1e-8 from below so that the
    caller can safely take log() for the likelihood term. Wrap with a NeuralTPP
    for likelihood / compensator computations.
    """

    def __init__(
        self, n_embd: int, vocab_size: int, time_encoder: nn.Module, spectral_norm: bool
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.time_encoder = time_encoder
        if spectral_norm:
            self.h = nn.Sequential(
                torch.nn.utils.spectral_norm(nn.Linear(n_embd, n_embd)),
                nn.GELU(),
                torch.nn.utils.spectral_norm(nn.Linear(n_embd, n_embd)),
            )
        else:
            self.h = nn.Sequential(
                nn.Linear(n_embd, n_embd),
                nn.GELU(),
                nn.Linear(n_embd, n_embd),
            )
        self.net = nn.Sequential(
            self.h,
            nn.Linear(n_embd, vocab_size),
        )

    def latent_states(self, h: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_encoder(delta_t.unsqueeze(-1))  # (B, L, n_embd)
        return self.h(h + time_emb)

    def forward(self, h: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, L, n_embd) transformer hidden states
            delta_t: (B, L) or (B, L, G) time since preceding event (days)

        Returns:
            Log-intensities, shape (B, L, V) or (B, L, G, V)
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

        _lambda = torch.nn.functional.softplus(self.net(h + time_emb))
        _lambda = torch.clamp(_lambda, min=1e-8)
        return _lambda


class NeuralTPP:

    def __init__(
        self,
        hidden_states: torch.Tensor,
        intensity_func: NeuralIntensity,
        timesteps: torch.Tensor,
        tokens: torch.Tensor,
        n_grid: int,
        integrate_method: str = "trapezoid",
        time_unit: float = 365.25,
    ):
        self.timesteps = timesteps
        self.first_timesteps = (
            torch.where(self.timesteps == -1e4, float("inf"), self.timesteps)
            .min(dim=1)
            .values
        )
        self.h = hidden_states
        self.f = intensity_func
        self.terminate_except = torch.tensor(
            [1], device=self.h.device, dtype=torch.long
        )
        # cache the history-dependent cumulative-seen mask; queries just gather from it
        self.occurred_mask = have_occurred(
            history_x=tokens,
            terminate_except=self.terminate_except,
            vocab_size=self.f.vocab_size,
        )

        self.n_grid = n_grid
        self.integrate_method = integrate_method
        self.time_unit = time_unit

    @property
    def shape(self):
        return self.timesteps.shape

    def latent_states(self, t: torch.Tensor):
        assert t.shape[0] == self.timesteps.shape[0]

        idx = nearest_input_pos(age=self.timesteps, targets_age=t)
        invalid = idx == -1
        idx = idx.clamp(min=0)
        nearest_t = torch.take_along_dim(self.timesteps, indices=idx, dim=1)
        invalid = invalid | (nearest_t == -1e4)
        nearest_t = nearest_t.masked_fill(invalid, -1e4)

        h = torch.take_along_dim(self.h, indices=idx.unsqueeze(-1), dim=1)
        delta_t = (t - nearest_t).clamp(
            min=0
        )  # invalid rows have nonsense delta_t; NaN-fill below

        return self.f.latent_states(h, delta_t)

    def intensity(self, t: torch.Tensor):
        assert t.shape[0] == self.timesteps.shape[0]

        idx = nearest_input_pos(age=self.timesteps, targets_age=t)
        invalid = idx == -1
        idx = idx.clamp(min=0)
        nearest_t = torch.take_along_dim(self.timesteps, indices=idx, dim=1)
        invalid = invalid | (nearest_t == -1e4)
        nearest_t = nearest_t.masked_fill(invalid, -1e4)

        h = torch.take_along_dim(self.h, indices=idx.unsqueeze(-1), dim=1)
        delta_t = (t - nearest_t).clamp(
            min=0
        )  # invalid rows have nonsense delta_t; NaN-fill below
        intensity = self.f(h, delta_t)

        mask = torch.take_along_dim(self.occurred_mask, idx.unsqueeze(-1), dim=1)
        intensity = intensity.masked_fill(mask, 0).masked_fill(
            invalid.unsqueeze(-1), torch.nan
        )
        return intensity, nearest_t

    def integral(
        self, t0: torch.Tensor, t1: torch.Tensor, n_grid: int, method: str = "trapezoid"
    ):
        t = torch.stack((t0, t1), dim=1)
        t_grid = F.interpolate(
            t.unsqueeze(0), size=n_grid, mode="linear", align_corners=True
        )
        t_grid = t_grid.squeeze(0)
        intensity, _ = self.intensity(t_grid)
        t_grid = t_grid[..., None].expand_as(intensity).clone()
        t_grid /= self.time_unit

        if method == "trapezoid":
            return torch.trapezoid(intensity, t_grid, dim=1), None
        elif method == "monte-carlo":
            raise NotImplementedError
        else:
            raise ValueError

    def log_likelihood(
        self,
        x1: torch.Tensor,
        t1: torch.Tensor,
        n_grid: None | int = None,
    ):

        if n_grid is None:
            n_grid = self.n_grid

        # mark queries with no valid strict-before non-padding history; these
        # positions are nan-filled at the end so they don't leak into the loss.
        # this catches both "no history" (nearest_input_pos returns -1) and the
        # leaky case where strict-before lands on a -1e4 padding entry.
        first = self.first_timesteps.view(-1, *([1] * (t1.dim() - 1)))
        invalid = t1 <= first

        idx = nearest_input_pos(age=self.timesteps, targets_age=t1).clamp(min=0)
        h = torch.take_along_dim(self.h, indices=idx.unsqueeze(-1), dim=1)
        t0 = torch.take_along_dim(self.timesteps, indices=idx, dim=1)
        delta_t = t1 - t0
        intensity = self.f(h, delta_t)
        log_intensity = torch.log(intensity)
        mask = torch.take_along_dim(self.occurred_mask, idx.unsqueeze(-1), dim=1)
        log_intensity = log_intensity.masked_fill(mask, -torch.inf)
        log_intensity_k = torch.gather(
            log_intensity, dim=-1, index=x1.unsqueeze(-1)
        ).squeeze(-1)

        # stochastic grid: sample (n_grid - 2) interior points uniformly in (0, 1)
        # and pin the endpoints at 0 and δ. randomizing the interior breaks the
        # model's ability to reliably hide a sharp peak between fixed grid points,
        # which would otherwise let it underestimate the compensator and lower
        # NLL without genuinely improving fit.
        interior = torch.rand(n_grid - 2, device=h.device, dtype=h.dtype)
        t_unit = (
            torch.cat(
                [
                    torch.zeros(1, device=h.device, dtype=h.dtype),
                    interior,
                    torch.ones(1, device=h.device, dtype=h.dtype),
                ]
            )
            .sort()
            .values
        )  # (G,)
        grid_delta_t = t_unit * delta_t.unsqueeze(-1)  # (B, L, G)

        intensity_grid = self.f(h, grid_delta_t)  # (B, L, G, V) — λ values
        intensity_grid = intensity_grid.masked_fill(mask.unsqueeze(-2), 0)

        total_intensity = intensity_grid.sum(dim=-1)  # (B, L, G) = Σ_v λ_v
        if self.integrate_method == "trapezoid":
            compensator = torch.trapezoid(
                total_intensity, grid_delta_t / self.time_unit, dim=-1
            )  # (B, L)
        elif self.integrate_method == "monte-carlo":
            # average over random interior points only; endpoints (0 and δ)
            # are fixed by the shared grid construction and would bias the MC mean.
            compensator = (
                torch.mean(total_intensity[..., 1:-1], dim=-1)
                * delta_t
                / self.time_unit
            )
        else:
            raise ValueError

        ll = log_intensity_k - compensator
        return ll.masked_fill(invalid, torch.nan)


class NeuralODEIntensity(nn.Module):

    def __init__(self, n_embd: int, vocab_size: int):
        super().__init__()
        self.dh_dt = nn.Sequential(
            nn.Linear(n_embd, n_embd), nn.GELU(), nn.Linear(n_embd, n_embd), nn.Tanh()
        )
        self.vocab_size = vocab_size
        self.intensity_projector = nn.Sequential(
            nn.Linear(n_embd, vocab_size),
        )
        self.delta_t = None
        self.nfe = 0

    def forward_intensity(self, h):
        _lambda = F.softplus(self.intensity_projector(h))
        return _lambda.clamp(min=1e-8)

    def forward_latents(self, tau, h):
        self.nfe += 1
        dh_dt = self.dh_dt(h)
        return dh_dt * self.delta_t.unsqueeze(-1)

    def forward(self, tau, state):
        self.nfe += 1
        h, _ = state
        dh_dt = self.dh_dt(h)
        intensity = self.forward_intensity(h)
        return dh_dt * self.delta_t.unsqueeze(-1), intensity * self.delta_t.unsqueeze(
            -1
        )


class NeuralODETPP:

    def __init__(
        self,
        ode: NeuralODEIntensity,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        tokens: torch.Tensor,
        time_unit: float = 365.25,
        method: str = "rk4",
        step_size: float = 0.25,
    ):

        self.ode = ode
        self.timesteps = timesteps
        self.first_timesteps = (
            torch.where(self.timesteps == -1e4, float("inf"), self.timesteps)
            .min(dim=1)
            .values
        )
        self.h = hidden_states
        self.terminate_except = torch.tensor(
            [1], device=self.h.device, dtype=torch.long
        )
        # cache the history-dependent cumulative-seen mask; queries just gather from it
        self.occurred_mask = have_occurred(
            history_x=tokens,
            terminate_except=self.terminate_except,
            vocab_size=self.ode.vocab_size,
        )

        self.time_unit = time_unit
        self.method = method
        self.step_size = step_size

    @property
    def shape(self):
        return self.timesteps.shape

    def intensity(self, t: torch.Tensor):
        assert t.shape[0] == self.timesteps.shape[0]

        idx = nearest_input_pos(age=self.timesteps, targets_age=t)
        invalid = idx == -1
        idx = idx.clamp(min=0)
        nearest_t = torch.take_along_dim(self.timesteps, indices=idx, dim=1)
        invalid = invalid | (nearest_t == -1e4)
        nearest_t = nearest_t.masked_fill(invalid, -1e4)

        h = torch.take_along_dim(self.h, indices=idx.unsqueeze(-1), dim=1)
        delta_t = (t - nearest_t).clamp(min=0)
        # zero delta_t at invalid positions so the ODE does not integrate
        # over the huge t - (-1e4) gap and leave the training distribution.
        delta_t = delta_t.masked_fill(invalid, 0.0)
        t_normalized = torch.tensor([0.0, 1.0], dtype=torch.float32, device=t.device)
        self.ode.delta_t = delta_t / self.time_unit

        self.ode.nfe = 0
        if self.method == "rk4":
            options = {"step_size": self.step_size}
        else:
            options = None
        latents = odeint(
            func=self.ode.forward_latents,
            y0=h,
            t=t_normalized,
            method=self.method,
            options=options,
            rtol=1e-3,
            atol=1e-4,
        )

        h1 = latents[-1]
        intensity = self.ode.forward_intensity(h1)

        mask = torch.take_along_dim(self.occurred_mask, idx.unsqueeze(-1), dim=1)
        intensity = intensity.masked_fill(mask, 0).masked_fill(
            invalid.unsqueeze(-1), torch.nan
        )
        return intensity, nearest_t

    def log_likelihood(
        self,
        x1: torch.Tensor,
        t1: torch.Tensor,
    ):
        first = self.first_timesteps.view(-1, *([1] * (t1.dim() - 1)))
        invalid = t1 <= first

        idx = nearest_input_pos(age=self.timesteps, targets_age=t1).clamp(min=0)
        h0 = torch.take_along_dim(self.h, indices=idx.unsqueeze(-1), dim=1)
        t0 = torch.take_along_dim(self.timesteps, indices=idx, dim=1)
        mask = torch.take_along_dim(self.occurred_mask, idx.unsqueeze(-1), dim=1)

        delta_t = t1 - t0
        t_normalized = torch.tensor([0.0, 1.0], dtype=torch.float32, device=t1.device)
        self.ode.delta_t = delta_t / self.time_unit
        cumul_lambda_0 = torch.zeros((*t1.shape, self.ode.vocab_size), device=x1.device)

        self.ode.nfe = 0
        if self.method == "rk4":
            options = {"step_size": self.step_size}
        else:
            options = None
        latents, cumulative_intensity = odeint(
            func=self.ode,
            y0=(h0, cumul_lambda_0),
            t=t_normalized,
            method=self.method,
            options=options,
            rtol=1e-3,
            atol=1e-4,
        )

        h1 = latents[-1]
        cumulative_intensity = cumulative_intensity[-1]
        intensity = self.ode.forward_intensity(h1)
        log_intensity = torch.log(intensity)
        log_intensity_k = torch.gather(
            log_intensity, dim=-1, index=x1.unsqueeze(-1)
        ).squeeze(-1)

        cumulative_intensity = cumulative_intensity.masked_fill(mask, 0)
        compensator = cumulative_intensity.sum(dim=-1)

        ll = log_intensity_k - compensator
        return ll.masked_fill(invalid, torch.nan)
