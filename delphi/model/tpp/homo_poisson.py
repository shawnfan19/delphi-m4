import torch

from delphi.model.utils import have_occurred, nearest_input_pos


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
