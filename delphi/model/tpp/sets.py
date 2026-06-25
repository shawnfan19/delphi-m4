import torch
import torch.nn as nn
import torch.nn.functional as F

from delphi.model.utils import have_occurred, multi_hot, nearest_input_pos


class DPPSetHead(nn.Module):
    """Read-out heads for the dynamic-DPP set-valued TPP.

    Off the transformer hidden state ``h(t)``:

    - ``quality`` — per-item log-quality logits; ``q_v = exp(logit_v)`` is the
      DPP quality term (with unit-norm item vectors ``L_vv = q_v^2``).
    - ``total_intensity`` — scalar log-rate; ``softplus`` of it is the ground
      intensity ``lambda*(t)`` for the timing component.

    The diversity term (cosine similarity of item embeddings) is supplied by the
    model's token-embedding table at call time, so this module owns no item
    vectors of its own.
    """

    def __init__(self, n_embd: int, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.quality = nn.Linear(n_embd, vocab_size, bias=True)
        self.total_intensity = nn.Linear(n_embd, 1, bias=True)


class DynamicDPPTPP:
    """Set-valued marked TPP with a history-dependent DPP over co-occurring marks.

    Implements the Dynamic-DPP model of Chang, Boyd & Smyth, "Probabilistic
    Modeling for Sequences of Sets in Continuous-Time" (AISTATS 2024). Each
    distinct event time carries a *set* of marks (here: tokens sharing an age),
    and the per-event log-likelihood factorises (paper Eqs. 2/6/10/13) into

        log p(set, t) = [log det L_S - log det(L + I)]   # mark  ("what")
                        + [log lambda* - lambda* . dt]    # time  ("when")

    with the quality-diversity kernel

        L_ij = q_i(h) * <e_i, e_j> / (||e_i|| ||e_j||) * q_j(h),

    ``q = exp(quality_head(h))`` and ``e`` the shared token embeddings. The
    K x K normaliser is computed via the d x d identity
    ``det(L + I_K) = det(I_d + Phi^T Phi)`` with ``Phi_v = q_v * normalize(e_v)``
    (cheap: rank(L) <= d = n_embd); ``det L_S`` is the small |S| x |S| principal
    minor. Tokens in ``exclude`` (padding / no-event / ignore) and
    already-occurred tokens are removed from the set (their ``q -> 0``), so a
    no-event position is the empty set, scored by ``-log det(L + I)``.

    Mirrors :class:`HomoPoissonTPP`'s interface: ``log_likelihood`` /
    ``log_p_marks`` / ``log_p_times`` return ``(B, L)`` and are NaN where there
    is no strict-before history, and at cluster-continuation positions so each
    set is scored exactly once. The mark/time split sums back to the joint.
    """

    # ponytail: the normaliser det(L+I) is computed densely at every position —
    # O(B*L*V*d^2) compute, O(B*L*d^2) memory. Fine at V=1270 / d=120; if it
    # dominates, gather the scored (non-dropped) cluster-rep positions first and
    # run slogdet only there.
    _LOGIT_CLAMP = 10.0  # cap quality logits so exp() can't overflow
    # diagonal jitter -> finite slogdet on (near-)singular set Grams. A DPP
    # cannot place more than rank(L) <= n_embd items in one set; a same-age
    # cluster larger than n_embd would be floored near log(jitter) rather than
    # its true ~0 probability — impossible in practice (n_embd=120 >> any
    # same-day disease cluster), so the jitter is just a NaN guard.
    _JITTER = 1e-6

    def __init__(
        self,
        hidden_states: torch.Tensor,
        head: DPPSetHead,
        embedding: torch.Tensor,
        timesteps: torch.Tensor,
        tokens: torch.Tensor,
        exclude: torch.Tensor,
        terminate_except: torch.Tensor,
        time_unit: float = 365.25,
    ):
        self.h = hidden_states
        self.head = head
        self.timesteps = timesteps
        self.first_timesteps = (
            torch.where(self.timesteps == -1e4, float("inf"), self.timesteps)
            .min(dim=1)
            .values
        )
        V = head.vocab_size
        self.vocab_size = V
        self.time_unit = time_unit
        # unit-norm item vectors (diversity); grad flows back to the shared wte
        self.E = F.normalize(embedding, dim=-1)
        # tokens kept in the DPP set (everything but padding / no-event / ignore)
        keep = torch.ones(V, dtype=torch.bool, device=embedding.device)
        keep[exclude.to(embedding.device)] = False
        self.keep_token = keep  # (V,)
        self.occurred_mask = have_occurred(
            history_x=tokens, terminate_except=terminate_except, vocab_size=V
        )

    @property
    def shape(self):
        return self.timesteps.shape

    def _predict(self, t1: torch.Tensor):
        """Prediction point + dropped positions shared by mark and time terms.

        Returns the strict-before input index, the hidden state and time-gap
        there, and a ``drop`` mask (no strict-before history, or a same-age
        cluster-continuation position — ``cooccur`` from :func:`multi_hot`).
        """
        assert t1.dim() == 2, "DynamicDPPTPP scores (B, L) set targets only"
        idx = nearest_input_pos(age=self.timesteps, targets_age=t1).clamp(min=0)
        h_p = torch.take_along_dim(self.h, idx.unsqueeze(-1), dim=1)
        t0 = torch.take_along_dim(self.timesteps, idx, dim=1)
        delta_t = t1 - t0

        # cooccur (same construction as multi_hot): a position continues the
        # previous cluster iff dt == 0 and age > 0; first column padded non-tie.
        dt = torch.diff(t1, dim=1)
        dt = torch.cat((torch.ones_like(t1[:, :1]), dt), dim=1)
        cooccur = (dt == 0) & (t1 > 0)
        drop = (t1 <= self.first_timesteps.unsqueeze(-1)) | cooccur
        return idx, h_p, delta_t, drop

    def _quality(self, idx: torch.Tensor, h_p: torch.Tensor):
        """Per-item quality ``q_v(h_p)`` with excluded / occurred tokens zeroed."""
        ql = self.head.quality(h_p)  # (B, L, V)
        occ = torch.take_along_dim(self.occurred_mask, idx.unsqueeze(-1), dim=1)
        masked = occ | (~self.keep_token).view(1, 1, -1)
        ql = ql.clamp(max=self._LOGIT_CLAMP).masked_fill(masked, float("-inf"))
        return ql.exp()  # (B, L, V); 0 at masked tokens

    def _mark(self, x1: torch.Tensor, t1: torch.Tensor, idx, h_p):
        """DPP set log-prob  log det L_S - log det(L + I)  at each position."""
        B, L = t1.shape
        V, d = self.E.shape
        q = self._quality(idx, h_p)

        # normaliser: log det(L + I_V) = log det(I_d + Phi^T Phi), Phi_v = q_v e_v
        qsq = q * q
        outer = (self.E.unsqueeze(2) * self.E.unsqueeze(1)).reshape(V, d * d)
        M = (qsq.reshape(B * L, V) @ outer).reshape(B, L, d, d)
        eye_d = torch.eye(d, device=M.device, dtype=M.dtype)
        logdet_norm = torch.linalg.slogdet(eye_d + M)[1]  # (B, L)

        # set minor: log det L_S over the disease-only set, batched via padded Gram
        hot, _ = multi_hot(targets=x1, targets_age=t1, vocab_size=V)
        set_hot = hot * self.keep_token.view(1, 1, V)
        m_max = int(set_hot.sum(dim=-1).max().clamp(min=1).item())
        vals, members = set_hot.topk(m_max, dim=-1)  # (B, L, m)
        valid = vals > 0
        q_S = torch.take_along_dim(q, members, dim=-1)  # (B, L, m)
        E_S = self.E[members]  # (B, L, m, d)
        phi_S = (q_S * valid).unsqueeze(-1) * E_S  # zero rows for padding members
        gram = phi_S @ phi_S.transpose(-1, -2)  # (B, L, m, m)
        eye_m = torch.eye(m_max, device=gram.device, dtype=gram.dtype)
        # identity block on padding members (empty set -> det 1) + jitter
        gram = gram + torch.diag_embed((~valid).to(gram.dtype)) + self._JITTER * eye_m
        logdet_set = torch.linalg.slogdet(gram)[1]  # (B, L)

        return logdet_set - logdet_norm

    def _time(self, delta_t: torch.Tensor, h_p: torch.Tensor):
        """Homogeneous-between-events timing log-density  log lambda* - lambda*.dt."""
        lam = F.softplus(self.head.total_intensity(h_p)).squeeze(-1).clamp(min=1e-8)
        return torch.log(lam) - lam * delta_t / self.time_unit

    def log_likelihood(self, x1: torch.Tensor, t1: torch.Tensor):
        idx, h_p, delta_t, drop = self._predict(t1)
        ll = self._mark(x1, t1, idx, h_p) + self._time(delta_t, h_p)
        return ll.masked_fill(drop, torch.nan)

    def log_p_marks(self, x1: torch.Tensor, t1: torch.Tensor):
        idx, h_p, _, drop = self._predict(t1)
        return self._mark(x1, t1, idx, h_p).masked_fill(drop, torch.nan)

    def log_p_times(self, t1: torch.Tensor):
        _, h_p, delta_t, drop = self._predict(t1)
        return self._time(delta_t, h_p).masked_fill(drop, torch.nan)
