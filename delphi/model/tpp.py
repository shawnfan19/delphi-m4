import torch
import torch.nn as nn
import torch.nn.functional as F

# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint

from delphi.model.utils import (
    have_occurred,
    nearest_input_pos,
)


class HomoPoissonTPP:

    def __init__(
        self,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        timesteps: torch.Tensor,
        tokens: torch.Tensor,
        terminate_except: torch.Tensor,
        time_unit: float = 1.0,
    ):
        self.h = hidden_states
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
        self.time_unit = time_unit

    @property
    def shape(self):
        return self.timesteps.shape

    def __getitem__(self, index):
        """Index the batch (dim-0), returning a TPP over the selected sequences.

        Slices every batch-aligned tensor attribute (leading dim == batch size)
        and shares the rest (e.g. ``time_unit``); does not re-run ``__init__``,
        so ``occurred_mask`` is not recomputed. Use a slice or 1-D index — a
        bare int would collapse the batch dim.
        """
        batch = self.timesteps.shape[0]
        obj = self.__class__.__new__(self.__class__)
        for name, value in self.__dict__.items():
            batched = (
                torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == batch
            )
            setattr(obj, name, value[index] if batched else value)
        return obj

    def latent_states(self, t: torch.Tensor):
        assert t.shape[0] == self.timesteps.shape[0]

        idx = nearest_input_pos(age=self.timesteps, targets_age=t)
        invalid = idx == -1
        idx = idx.clamp(min=0)
        nearest_t = torch.take_along_dim(self.timesteps, indices=idx, dim=1)
        invalid = invalid | (nearest_t == -1e4)
        nearest_t = nearest_t.masked_fill(invalid, -1e4)

        h = torch.take_along_dim(self.h, indices=idx.unsqueeze(-1), dim=1)

        return h

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

    def intensity_at(self, t: torch.Tensor, tokens: torch.Tensor, **kwargs):
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

    def integral(self, t0: torch.Tensor, t1: torch.Tensor, n_grid: None | int = None):
        # n_grid is accepted for API parity with NeuralTPP but unused here: the
        # homogeneous-between-events integral is exact piecewise-constant.
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
        _timesteps = torch.clamp(_timesteps, min=t0.unsqueeze(-1), max=t1.unsqueeze(-1))
        delta_t = torch.diff(_timesteps, dim=1)  # (B, L) — one gap per λ
        delta_t /= self.time_unit
        _cap_lambda = torch.sum(_lambda * delta_t.unsqueeze(-1), dim=1)
        return _cap_lambda, delta_t.sum(dim=1)

    def _intermediates(self, t1: torch.Tensor):
        """Shared marked-TPP building blocks evaluated at query times ``t1``.

        For the piecewise-constant interval each query falls in, returns:

        - ``log_intensity`` (B, *Q, V): per-token log-intensities ``log λ_v``,
          with already-occurred (self-terminated) tokens set to ``-inf``;
        - ``log_sum_intensity`` (B, *Q, 1): log total ground intensity
          ``log Σ_v λ_v``;
        - ``compensator`` (B, *Q, 1): ``-Σ_v λ_v · Δt / time_unit``, the
          integrated intensity over the gap since the preceding event;
        - ``invalid`` (B, *Q): queries with no strict-before non-padding
          history, which callers NaN-fill so they don't leak into the loss.

        ``log_likelihood``, ``log_p_marks`` and ``log_p_times`` are all derived
        from these, so the mark/time split sums back to the joint exactly.
        """
        # mark queries with no valid strict-before non-padding history; these
        # positions are nan-filled by callers so they don't leak into the loss.
        first = self.first_timesteps.view(-1, *([1] * (t1.dim() - 1)))
        invalid = t1 <= first

        idx = nearest_input_pos(age=self.timesteps, targets_age=t1).clamp(min=0)
        log_intensity = torch.take_along_dim(
            self.logits, indices=idx.unsqueeze(-1), dim=1
        )
        mask = torch.take_along_dim(self.occurred_mask, idx.unsqueeze(-1), dim=1)
        log_intensity = log_intensity.masked_fill(mask, float("-inf"))
        log_sum_intensity = torch.logsumexp(log_intensity, dim=-1, keepdim=True)

        t0 = torch.take_along_dim(self.timesteps, indices=idx, dim=1)
        delta_t = t1 - t0
        compensator = (
            -torch.exp(log_sum_intensity) * delta_t.unsqueeze(-1) / self.time_unit
        )
        return log_intensity, log_sum_intensity, compensator, invalid

    def log_likelihood(self, x1: torch.Tensor, t1: torch.Tensor):
        # joint marked-TPP log-likelihood  log λ_{x1} − Σ_v λ_v · Δt;
        # exactly log_p_marks(x1, t1) + log_p_times(t1).
        log_intensity, _, compensator, invalid = self._intermediates(t1)
        log_intensity_k = torch.gather(
            input=log_intensity, dim=-1, index=x1.unsqueeze(-1)
        )
        ll = log_intensity_k + compensator
        return ll.masked_fill(invalid.unsqueeze(-1), torch.nan).squeeze(-1)

    def log_p_marks(self, x1: torch.Tensor, t1: torch.Tensor):
        """Log-probability of the mark ``x1`` *given* an event occurs at ``t1``.

        The categorical "what" term of the marked-TPP factorisation,

            log p(m | t) = log λ_{x1}(t) − log Σ_v λ_v(t),

        i.e. the (history-masked) log-softmax of the per-token log-intensities
        evaluated at the observed token. A proper distribution over the
        still-available marks; with ``log_p_times`` it sums to ``log_likelihood``.
        """
        log_intensity, log_sum_intensity, _, invalid = self._intermediates(t1)
        log_intensity_k = torch.gather(
            input=log_intensity, dim=-1, index=x1.unsqueeze(-1)
        )
        log_p = log_intensity_k - log_sum_intensity
        return log_p.masked_fill(invalid.unsqueeze(-1), torch.nan).squeeze(-1)

    def log_p_times(self, t1: torch.Tensor):
        """Log-density of the event *time* ``t1`` (the "when" term).

        The conditional inter-event-time density of the homogeneous-between-
        events ground process,

            log p(t) = log Σ_v λ_v − Σ_v λ_v · Δt / time_unit,

        i.e. the Exponential(λ*) log-density with total rate λ* = Σ_v λ_v and
        Δt the gap since the preceding event. Independent of which mark occurs;
        with ``log_p_marks`` it sums to ``log_likelihood``.
        """
        _, log_sum_intensity, compensator, invalid = self._intermediates(t1)
        log_p = log_sum_intensity + compensator
        return log_p.masked_fill(invalid.unsqueeze(-1), torch.nan).squeeze(-1)


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
        print(f"\rBatch NFE: {self.ode.nfe}", end="", flush=True)

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
