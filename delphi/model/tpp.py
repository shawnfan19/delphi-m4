import torch
import torch.nn as nn

from delphi.model.utils import (
    have_occurred,
    nearest_input_pos,
)


class HomoPoissonTPP:

    def __init__(
        self,
        logits: torch.Tensor,
        timesteps: torch.Tensor,
        tokens: torch.Tensor,
        terminate_except: torch.Tensor,
    ):
        self.timesteps = timesteps
        self.first_timesteps = (
            torch.where(self.timesteps == -1e4, float("inf"), self.timesteps)
            .min(dim=1)
            .values
        )
        self.logits = logits  # raw log-intensities; extinguishment applied per-query
        self.occurred_mask = have_occurred(
            history_x=tokens,
            terminate_except=terminate_except,
            vocab_size=logits.shape[-1],
        )

    @property
    def shape(self):
        return self.timesteps.shape

    def intensity(self, t: torch.Tensor):
        assert t.shape[0] == self.timesteps.shape[0]

        idx = nearest_input_pos(age=self.timesteps, targets_age=t)
        invalid = idx == -1
        idx = idx.clamp(min=0)
        nearest_t = torch.take_along_dim(self.timesteps, indices=idx, dim=1)
        # also invalid when strict-before lands on a padding row (nearest_t == -1e4)
        invalid = invalid | (nearest_t == -1e4)
        nearest_t = nearest_t.masked_fill(invalid, -1e4)

        log_intensity = torch.take_along_dim(
            self.logits, indices=idx.unsqueeze(-1), dim=1
        )
        mask = torch.take_along_dim(self.occurred_mask, idx.unsqueeze(-1), dim=1)
        log_intensity = log_intensity.masked_fill(mask, float("-inf"))
        intensity = log_intensity.exp().masked_fill(invalid.unsqueeze(-1), torch.nan)
        return intensity, nearest_t

    def intensity_at(self, t: torch.Tensor, tokens: torch.Tensor):
        """Per-pair scalar intensity λ_v(t) at the queried token v.

        t: (B, *Q) query timesteps; tokens: broadcastable to (B, *Q).
        Returns intensity (B, *Q) and nearest_t (B, *Q); invalid positions
        (no input strictly before t, or strict-before lands on a padding row)
        get NaN intensity and -1e4 nearest_t. Avoids the (B, *Q, V) intermediate.
        """
        assert t.shape[0] == self.timesteps.shape[0]

        idx = nearest_input_pos(age=self.timesteps, targets_age=t)
        invalid = idx == -1
        idx = idx.clamp(min=0)
        nearest_t = torch.take_along_dim(self.timesteps, indices=idx, dim=1)
        invalid = invalid | (nearest_t == -1e4)
        nearest_t = nearest_t.masked_fill(invalid, -1e4)

        B = self.logits.shape[0]
        tokens_b = torch.broadcast_to(tokens, idx.shape)
        flat_idx = idx.reshape(B, -1)
        flat_tok = tokens_b.reshape(B, -1)
        flat_b = (
            torch.arange(B, device=self.logits.device).unsqueeze(1).expand_as(flat_idx)
        )

        log_intensity = self.logits[flat_b, flat_idx, flat_tok].reshape(idx.shape)
        mask = self.occurred_mask[flat_b, flat_idx, flat_tok].reshape(idx.shape)
        log_intensity = log_intensity.masked_fill(mask, float("-inf"))
        intensity = log_intensity.exp().masked_fill(invalid, torch.nan)
        return intensity, nearest_t

    def integral(self, t0: torch.Tensor, t1: torch.Tensor):
        # require t0 to lie within the timeline (>= first non-padding event per sequence)
        assert t0.shape[0] == t1.shape[0] == self.timesteps.shape[0]
        assert (
            t0 >= self.first_timesteps
        ).all(), "t0 must be >= first non-padding timestep"

        # intensity on the interval starting at timesteps[j] must mask tokens seen
        # in history[0..j] (inclusive) — that's occurred_mask[:, j, :] directly
        _lambda = self.logits.exp().masked_fill(self.occurred_mask, 0)
        # extend the timeline on the right: use t1 when it's past the last event,
        # otherwise use max age (neutralized by the t1 clamp → zero final delta)
        max_age = self.timesteps.max(dim=1).values
        right = torch.maximum(t1, max_age).unsqueeze(-1)
        _timesteps = torch.cat([self.timesteps, right], dim=1)
        _timesteps = torch.clamp(_timesteps, min=t0, max=t1)
        delta_t = torch.diff(_timesteps, dim=1)  # (B, L) — one gap per λ
        _cap_lambda = torch.sum(_lambda * delta_t.unsqueeze(-1), dim=1)
        return _cap_lambda, delta_t.sum(dim=1)

    def log_likelihood(self, x1: torch.Tensor, t1: torch.Tensor):
        # mark queries with no valid strict-before non-padding history; these
        # positions are nan-filled at the end so they don't leak into the loss.
        first = self.first_timesteps.view(-1, *([1] * (t1.dim() - 1)))
        invalid = t1 <= first

        idx = nearest_input_pos(age=self.timesteps, targets_age=t1).clamp(min=0)
        log_intensity = torch.take_along_dim(
            self.logits, indices=idx.unsqueeze(-1), dim=1
        )
        mask = torch.take_along_dim(self.occurred_mask, idx.unsqueeze(-1), dim=1)
        log_intensity = log_intensity.masked_fill(mask, float("-inf"))
        log_intensity_k = torch.gather(
            input=log_intensity, dim=-1, index=x1.unsqueeze(-1)
        )

        t0 = torch.take_along_dim(self.timesteps, indices=idx, dim=1)
        delta_t = t1 - t0
        log_sum_intensity = torch.logsumexp(log_intensity, dim=-1, keepdim=True)
        compensator = -torch.exp(log_sum_intensity) * delta_t.unsqueeze(-1)

        ll = log_intensity_k + compensator
        return ll.masked_fill(invalid.unsqueeze(-1), torch.nan).squeeze(-1)


class NeuralIntensity(nn.Module):
    """
    Parameterized intensity λ_v(τ) = softplus(net(h + time_encode(τ - t_i)))_v.

    Returns non-negative intensities, clamped at 1e-8 from below so that the
    caller can safely take log() for the likelihood term. Wrap with a NeuralTPP
    for likelihood / compensator computations.
    """

    def __init__(
        self,
        n_embd: int,
        vocab_size: int,
        time_encoder: nn.Module,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.time_encoder = time_encoder
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.GELU(),
            nn.Linear(n_embd, vocab_size),
        )

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

    def intensity(self, t: torch.Tensor):
        assert t.shape[0] == self.timesteps.shape[0]
        # broadcast (B,) first_timesteps up to match t's trailing query dims
        first = self.first_timesteps.view(-1, *([1] * (t.dim() - 1)))
        assert (t > first).all(), "t must be strictly > first non-padding timestep"

        # the t > first assertion guarantees nearest_input_pos returns a valid
        # (non -1) index, so the clamp is purely defensive
        idx = nearest_input_pos(age=self.timesteps, targets_age=t).clamp(min=0)
        h = torch.take_along_dim(self.h, indices=idx.unsqueeze(-1), dim=1)
        t0 = torch.take_along_dim(self.timesteps, indices=idx, dim=1)
        delta_t = t - t0
        intensity = self.f(h, delta_t)

        mask = torch.take_along_dim(self.occurred_mask, idx.unsqueeze(-1), dim=1)
        return intensity.masked_fill(mask, 0)

    def integral(self, t0: torch.Tensor, t1: torch.Tensor, n_grid: int):
        assert (
            t0 >= self.first_timesteps
        ).all(), "t0 must be >= first non-padding timestep"
        raise NotImplementedError

    def log_likelihood(self, x1: torch.Tensor, t1: torch.Tensor, n_grid: int):
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
        compensator = torch.trapezoid(
            total_intensity, grid_delta_t / 365.25, dim=-1
        )  # (B, L)

        ll = log_intensity_k - compensator
        return ll.masked_fill(invalid, torch.nan)


class NeuralDecayEmbedding(nn.Module):
    """
    Continuous-time intensity head with exponentially-decaying hidden state.

        destination = destination_projector(h)
        β           = exp(log_beta_projector(h))                 (per-dim decay rates)
        s(τ)        = destination + (h - destination) · exp(-β τ)
        log λ_v(τ)  = intensity_projector(s(τ))_v

    The state s(τ) is structurally smooth in τ (a per-dim sum of exponentials),
    so log λ(τ) and λ(τ) are smooth too. There's no way to express a sharp,
    sub-grid-spacing-scale peak — which is what makes this parameterization
    play well with numerical integration of the compensator.
    """

    def __init__(
        self,
        n_embd: int,
        vocab_size: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.intensity_projector = nn.Linear(n_embd, vocab_size)
        # self.log_beta_projector = nn.Sequential(
        #     nn.Linear(n_embd, n_embd),
        #     nn.GELU(),
        #     nn.Linear(n_embd, n_embd),
        # )
        # per-dim decay rate (1/year units after the /365.25 in forward)
        # init at log(0.05) ≈ -3 → decay timescale ~20 years per dim,
        # giving a slow, well-conditioned default state evolution
        self.log_beta = nn.Parameter(torch.full((n_embd,), -3.0))
        self.destination_projector = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.GELU(),
            nn.Linear(n_embd, n_embd),
        )

    def forward(self, h: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:       (B, L, n_embd) post-event hidden states.
            delta_t: (B, L) or (B, L, G) time elapsed since the corresponding event.

        Returns:
            Log-intensity log λ_v(τ), shape (B, L, V) or (B, L, G, V).
        """
        destination = self.destination_projector(h)  # (B, L, n_embd)
        # beta = self.log_beta_projector(h).exp()              # (B, L, n_embd)
        beta = self.log_beta.exp()

        # broadcast h / destination / beta across the G grid axis when delta_t is 3D
        has_grid = delta_t.dim() > h.dim() - 1
        if has_grid:
            beta = beta.unsqueeze(-2)  # (B, L, 1, n_embd)
            destination = destination.unsqueeze(-2)
            h = h.unsqueeze(-2)
        decay = (-beta * delta_t.unsqueeze(-1) / 365.25).exp()  # (B, L, [G,] n_embd)
        s = destination + (h - destination) * decay
        return self.intensity_projector(s)  # (B, L, [G,] V)


class NeuralDecayTPP:

    def __init__(
        self,
        hidden_states: torch.Tensor,
        intensity_func: NeuralDecayEmbedding,
        timesteps: torch.Tensor,
        tokens: torch.Tensor,
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

    def intensity(self, t: torch.Tensor):
        assert t.shape[0] == self.timesteps.shape[0]
        # broadcast (B,) first_timesteps up to match t's trailing query dims
        first = self.first_timesteps.view(-1, *([1] * (t.dim() - 1)))
        assert (t > first).all(), "t must be strictly > first non-padding timestep"

        # the t > first assertion guarantees nearest_input_pos returns a valid
        # (non -1) index, so the clamp is purely defensive
        idx = nearest_input_pos(age=self.timesteps, targets_age=t).clamp(min=0)
        h = torch.take_along_dim(self.h, indices=idx.unsqueeze(-1), dim=1)
        t0 = torch.take_along_dim(self.timesteps, indices=idx, dim=1)
        delta_t = t - t0
        log_intensity = self.f(h, delta_t)
        intensity = log_intensity.exp()

        mask = torch.take_along_dim(self.occurred_mask, idx.unsqueeze(-1), dim=1)
        return intensity.masked_fill(mask, 0)

    def integral(self, t0: torch.Tensor, t1: torch.Tensor, n_grid: int):
        assert (
            t0 >= self.first_timesteps
        ).all(), "t0 must be >= first non-padding timestep"
        raise NotImplementedError

    def log_likelihood(self, x1: torch.Tensor, t1: torch.Tensor, n_grid: int):
        first = self.first_timesteps.view(-1, *([1] * (t1.dim() - 1)))
        invalid = t1 <= first

        idx = nearest_input_pos(age=self.timesteps, targets_age=t1).clamp(min=0)
        h = torch.take_along_dim(self.h, indices=idx.unsqueeze(-1), dim=1)
        t0 = torch.take_along_dim(self.timesteps, indices=idx, dim=1)
        delta_t = t1 - t0  # (B, L)

        log_intensity = self.f(h, delta_t)  # (B, L, V)
        mask = torch.take_along_dim(self.occurred_mask, idx.unsqueeze(-1), dim=1)
        log_intensity = log_intensity.masked_fill(mask, -torch.inf).clamp(max=10)
        log_intensity_k = torch.gather(
            log_intensity, dim=-1, index=x1.unsqueeze(-1)
        ).squeeze(-1)

        # deterministic uniform grid: the decay parameterization keeps λ(τ)
        # structurally smooth (smallest expressible feature scales as 1/max(β),
        # i.e. years for the default init), so the model can't place a peak
        # between grid points — no need for stochastic sampling.
        t_unit = torch.linspace(0.0, 1.0, n_grid, device=h.device, dtype=h.dtype)
        grid_delta_t = t_unit * delta_t.unsqueeze(-1)  # (B, L, G)

        log_intensity_grid = self.f(h, grid_delta_t).clamp(max=10)  # (B, L, G, V)
        log_intensity_grid = log_intensity_grid.masked_fill(
            mask.unsqueeze(-2), -torch.inf
        )

        total_intensity = torch.logsumexp(log_intensity_grid, dim=-1).exp()  # (B, L, G)
        compensator = torch.trapezoid(
            total_intensity, grid_delta_t / 365.25, dim=-1
        )  # (B, L)

        ll = log_intensity_k - compensator
        return ll.masked_fill(invalid, torch.nan)
