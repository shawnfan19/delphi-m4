import torch
from torch.nn import functional as F


def causal_attention_mask(
    pad: torch.Tensor, timestep: None | torch.Tensor = None
) -> torch.Tensor:

    b, l = pad.shape
    device = pad.device

    if timestep is not None:
        assert timestep.shape == (b, l)
        q_time = timestep.view(b, l, 1)
        k_time = timestep.view(b, 1, l)
        attn_mask = (k_time <= q_time).int()
    else:
        attn_mask = torch.tril(torch.ones((l, l), device=device))
        attn_mask = attn_mask.view(1, l, l)
    pad_mask = pad.int().view(b, 1, l).to(torch.int)
    attn_mask = attn_mask * pad_mask
    diag_indices = torch.arange(attn_mask.shape[-1])
    attn_mask[..., diag_indices, diag_indices] = 1

    return attn_mask.unsqueeze(1)


def incremental_attention_mask(
    new_pad: torch.Tensor,
    past_pad: torch.Tensor,
) -> torch.Tensor:
    """
    Build attention mask for incremental (KV-cached) forward pass.

    Args:
        new_pad:  (B, T_new) bool — non-padding positions among new tokens
        past_pad: (B, T_cached) bool — non-padding positions in the KV cache

    Returns:
        (B, 1, T_new, T_cached + T_new) float mask
    """
    B, T_new = new_pad.shape
    T_cached = past_pad.shape[1]
    device = new_pad.device

    # Cached columns [0..T_cached): each new query attends to all non-padding
    # cached positions (age ordering is guaranteed by construction).
    # Shape: (B, T_new, T_cached)
    # Note: unlike causal_attention_mask (time-based), padding new queries
    # (age = -1e4) are not blocked from attending to valid cached positions —
    # the time gate (k_time <= q_time) would normally prevent this since real
    # ages exceed -1e4. The difference is harmless: outputs at padding positions
    # are never consumed during generation.
    cached_block = past_pad.float().view(B, 1, T_cached).expand(B, T_new, T_cached)

    new_block = (
        torch.ones(T_new, T_new, device=device)
        .view(1, T_new, T_new)
        .expand(B, T_new, T_new)
    )

    # Apply key padding for new tokens, then force diagonal to 1
    new_block = new_block * new_pad.float().view(B, 1, T_new)
    diag_indices = torch.arange(T_new, device=device)
    new_block[:, diag_indices, diag_indices] = 1.0

    # Concatenate along key dimension: (B, T_new, T_cached + T_new)
    attn_mask = torch.cat([cached_block.contiguous(), new_block], dim=2)
    return attn_mask.unsqueeze(1)  # (B, 1, T_new, T_cached + T_new)


def nearest_input_pos(age, targets_age, include_ties: bool = False):
    """
    For each target, find the position of the nearest input earlier than it;
    on ties (multiple inputs at the same time), pick the latest tied position
    (last in sequence order).

    Args:
        age: (B, L0) input timestamps, sorted along dim=-1 (the codebase's
            convention: left-padded with -1e4, then monotone real ages).
        targets_age: (B, *Q) target timestamps; arbitrary trailing query dims.
        include_ties: if False (default), "strictly before" — only inputs with
            age < target count; returns -1 if no such input exists.
            If True, "at or before" — inputs with age <= target count;
            returns -1 only if no input satisfies age <= target.

    Callers that cannot tolerate -1 should clamp after calling (e.g.,
    ``.clamp(min=0)``).
    """
    B = age.shape[0]
    q_shape = targets_age.shape[1:]
    age = age.contiguous()
    t_flat = targets_age.reshape(B, -1).contiguous()
    # right=False → first i with age[i] >= t  → idx-1 = last age[i] < t   (strict)
    # right=True  → first i with age[i] > t   → idx-1 = last age[i] <= t  (with ties)
    pos = torch.searchsorted(age, t_flat, right=include_ties) - 1
    return pos.reshape(B, *q_shape)


def multi_hot(
    targets: torch.Tensor, targets_age: torch.Tensor, vocab_size: int
) -> tuple[torch.Tensor, torch.Tensor]:

    device = targets.device
    batch_size, seq_len = targets.shape[0], targets.shape[1]

    dt = torch.diff(targets_age, dim=1)
    dt = torch.cat((torch.ones(batch_size, 1).to(device), dt), dim=1)
    # pad with ones to ensuring first position will not be cooccur
    cooccur = torch.logical_and(dt == 0, targets_age > 0)
    cum_cooccur = torch.cumsum(cooccur, dim=1)

    cluster_idx = torch.arange(seq_len).to(device)
    cluster_idx = cluster_idx.unsqueeze(0) - cum_cooccur
    cluster_seq_len = int(torch.max(cluster_idx).item()) + 1

    hot_targets = torch.zeros(batch_size, cluster_seq_len, vocab_size).to(device)
    batch_idx = torch.arange(batch_size).unsqueeze(1).to(device).long()
    hot_targets[batch_idx, cluster_idx, targets] = 1

    hot_targets = torch.take_along_dim(
        indices=cluster_idx.unsqueeze(-1), input=hot_targets, dim=1
    )

    return hot_targets, cooccur


def self_terminate_single(
    idx: torch.Tensor, logits: torch.Tensor, terminate_except: torch.Tensor
):
    fill = idx.clone()
    fill[torch.isin(fill, terminate_except.to(fill.device))] = 0

    batch_size = fill.shape[0]
    vocab_size = logits.shape[-1]

    # Scatter directly into [batch, vocab]
    mask = torch.zeros(batch_size, vocab_size, dtype=torch.bool, device=fill.device)
    mask.scatter_(1, fill, True)

    return logits.masked_fill(mask, float("-inf"))


def have_occurred(
    history_x: torch.Tensor,
    terminate_except: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """
    Per-history cumulative-seen mask. cum_mask[b, j, v] is True iff token v
    has appeared in history_x[b, 0..j] (ignoring tokens in terminate_except).

    Depends only on history, so can be cached across queries.
    """
    fill = history_x.clone()
    fill[torch.isin(fill, terminate_except.to(fill.device))] = 0
    B, L_hist = fill.shape
    one_hot = torch.zeros(B, L_hist, vocab_size, device=fill.device)
    one_hot.scatter_(2, fill.unsqueeze(-1), 1.0)
    return one_hot.cumsum(dim=1) > 0


def self_terminate(
    idx: torch.Tensor,
    estimator: torch.Tensor,
    terminate_except: torch.Tensor,
    fill_val: float = float("-inf"),
):
    fill = idx.clone()
    fill[torch.isin(fill, terminate_except.to(fill.device))] = 0

    batch_size, seq_len = fill.shape
    vocab_size = estimator.shape[-1]

    # One-hot encode: [batch, seq, vocab]
    one_hot = torch.zeros(batch_size, seq_len, vocab_size, device=fill.device)
    one_hot.scatter_(2, fill.unsqueeze(-1), 1.0)

    # Cumsum: mask[b, j, v] = True if token v appeared in positions 0..j
    mask = one_hot.cumsum(dim=1) > 0

    return estimator.masked_fill(mask, fill_val), mask

def _as_multihot(idx: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Convert either token ids (B, L) or already-multihot set tensors (B, L, V)
    into a boolean multihot tensor (B, L, V).
    """
    if idx.dim() == 3:
        if idx.shape[-1] != vocab_size:
            raise ValueError(
                f"Expected set input last dim={vocab_size}, got {idx.shape[-1]}"
            )
        return idx > 0

    if idx.dim() == 2:
        idx_l = idx.to(torch.long)
        out = torch.zeros(
            (*idx_l.shape, vocab_size),
            dtype=torch.bool,
            device=idx_l.device,
        )
        valid = (idx_l >= 0) & (idx_l < vocab_size)
        safe_idx = idx_l.clamp(min=0, max=vocab_size - 1)
        out.scatter_(2, safe_idx.unsqueeze(-1), True)
        out = out & valid.unsqueeze(-1)
        return out

    raise ValueError(f"Unsupported idx shape {tuple(idx.shape)}")


def set_history_availability(
    idx: torch.Tensor,
    vocab_size: int,
    terminate_except: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Availability mask for set-valued losses.

    At source position i we predict the next set after observing idx[:, i].
    Therefore, with self-termination enabled, items seen up to and including
    source position i are unavailable.

    terminate_except tokens remain always available, e.g. no_event.
    """
    seen_now = _as_multihot(idx, vocab_size=vocab_size)
    seen_cum = seen_now.cumsum(dim=1) > 0

    if terminate_except is not None and terminate_except.numel() > 0:
        ex = terminate_except.to(device=idx.device, dtype=torch.long)
        ex = ex[(ex >= 0) & (ex < vocab_size)]
        if ex.numel() > 0:
            seen_cum.index_fill_(-1, ex, False)

    return ~seen_cum


def _set_active_mask(
    *,
    idx: torch.Tensor,
    vocab_size: int,
    candidate_mask: torch.Tensor | None,
    terminate: bool,
    terminate_except: torch.Tensor | None,
) -> torch.Tensor:
    """
    Returns boolean active mask of shape (B, L, V).

    candidate_mask excludes PAD and ignored tokens globally.
    terminate optionally excludes previously seen tokens per sequence.
    """
    device = idx.device

    if candidate_mask is None:
        candidate_mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
        candidate_mask[0] = False
    else:
        candidate_mask = candidate_mask.to(device=device, dtype=torch.bool)

    active = candidate_mask.view(1, 1, vocab_size)

    if terminate:
        available = set_history_availability(
            idx=idx,
            vocab_size=vocab_size,
            terminate_except=terminate_except,
        )
        active = active & available

    return active


def nll_dynamic_bernoulli_set(
    *,
    log_ground_intensity: torch.Tensor,  # (B, L), log Lambda_i
    set_logits: torch.Tensor,            # (B, L, V), Bernoulli logits for rho_k
    targets: torch.Tensor,               # (B, L, V), multihot next set
    idx: torch.Tensor,                   # (B, L, V), multihot current set
    targets_age: torch.Tensor,           # (B, L)
    age: torch.Tensor,                   # (B, L)
    time_unit: float = 1.0,
    candidate_mask: torch.Tensor | None = None,
    terminate: bool = False,
    terminate_except: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pure Dynamic Bernoulli set likelihood from Chang/Boyd/Smyth.

    Time NLL:
        -log Lambda_i + Lambda_i * Delta t_i

    Set NLL:
        sum_k BCEWithLogits(a_ik, y_ik),
        where rho_ik = sigmoid(a_ik).

    Important:
        set_logits are Bernoulli logits, NOT log intensities.
    """
    if set_logits.shape != targets.shape:
        raise ValueError(
            f"set_logits and targets must have same shape; "
            f"got {tuple(set_logits.shape)} and {tuple(targets.shape)}"
        )

    _, _, V = set_logits.shape

    delta_t = (targets_age - age).clamp_min(0.0)
    delta_t_unit = delta_t / float(time_unit)

    time_nll = (
        -log_ground_intensity
        + torch.exp(log_ground_intensity) * delta_t_unit
    )

    active = _set_active_mask(
        idx=idx,
        vocab_size=V,
        candidate_mask=candidate_mask,
        terminate=terminate,
        terminate_except=terminate_except,
    )

    target_f = targets.to(dtype=set_logits.dtype).clamp(0.0, 1.0)

    bce = F.binary_cross_entropy_with_logits(
        set_logits,
        target_f,
        reduction="none",
    )

    set_nll = (bce * active.to(dtype=bce.dtype)).sum(dim=-1)

    return time_nll, set_nll


def nll_interval_dynamic_bernoulli_set(
    *,
    log_ground_intensity: torch.Tensor | None,  # (B, L), optional log Lambda_i
    set_log_intensity: torch.Tensor,            # (B, L, V), log lambda_ik
    targets: torch.Tensor,                      # (B, L, V), multihot next observed set
    idx: torch.Tensor,                          # (B, L, V), multihot current set
    targets_age: torch.Tensor,                  # (B, L)
    age: torch.Tensor,                          # (B, L)
    time_unit: float = 1.0,
    candidate_mask: torch.Tensor | None = None,
    terminate: bool = False,
    terminate_except: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Interval-censored Dynamic Bernoulli likelihood.

    Latent model over interval length Delta t:
        N_k(Delta t) ~ Poisson(lambda_k * Delta t)
        y_k = 1[N_k(Delta t) >= 1]

    Therefore:
        rho_k(Delta t) = 1 - exp(-lambda_k * Delta t)

    Set NLL:
        - sum_{k in S} log(1 - exp(-lambda_k Delta t))
        + sum_{k not in S} lambda_k Delta t

    Important:
        set_log_intensity are log latent item rates.
    """
    if set_log_intensity.shape != targets.shape:
        raise ValueError(
            f"set_log_intensity and targets must have same shape; "
            f"got {tuple(set_log_intensity.shape)} and {tuple(targets.shape)}"
        )

    _, _, V = set_log_intensity.shape

    delta_t = (targets_age - age).clamp_min(0.0)
    delta_t_unit = delta_t / float(time_unit)

    if log_ground_intensity is None:
        time_nll = torch.zeros_like(delta_t_unit)
    else:
        time_nll = (
            -log_ground_intensity
            + torch.exp(log_ground_intensity) * delta_t_unit
        )

    active = _set_active_mask(
        idx=idx,
        vocab_size=V,
        candidate_mask=candidate_mask,
        terminate=terminate,
        terminate_except=terminate_except,
    )

    target_f = targets.to(dtype=set_log_intensity.dtype).clamp(0.0, 1.0)

    rate_dt = torch.exp(set_log_intensity) * delta_t_unit.unsqueeze(-1)

    eps = (
        1e-6
        if set_log_intensity.dtype in (torch.float16, torch.bfloat16)
        else 1e-8
    )

    # Stable log(1 - exp(-rate_dt)).
    log_occ_prob = torch.log((-torch.expm1(-rate_dt)).clamp_min(eps))

    pos_nll = -target_f * log_occ_prob
    neg_nll = (1.0 - target_f) * rate_dt

    set_nll = ((pos_nll + neg_nll) * active.to(dtype=rate_dt.dtype)).sum(dim=-1)

    return time_nll, set_nll


def nll_homogeneous_cluster_poisson(
    log_intensity: torch.Tensor,
    log_aux_intensity: torch.Tensor,
    targets: torch.Tensor,
    targets_age: torch.Tensor,
    age: torch.Tensor,
):
    hot_targets, cooccur = multi_hot(
        targets=targets, targets_age=targets_age, vocab_size=log_intensity.shape[-1]
    )
    delta_t = targets_age - age
    EPS = 1e-8
    delta_t = torch.clamp(delta_t, min=EPS)
    part1 = log_aux_intensity
    part2 = -torch.exp(log_aux_intensity) * delta_t

    rate_times_dt = torch.exp(log_intensity) * delta_t.unsqueeze(-1)
    log_cdf = torch.log(-torch.expm1(-rate_times_dt))
    ll_have_occur = (hot_targets * log_cdf).sum(dim=-1)
    ll_have_not_occur = (
        -((1 - hot_targets) * torch.exp(log_intensity)).sum(dim=-1) * delta_t
    )
    ll_cluster = ll_have_occur + ll_have_not_occur

    return -(part1 + part2), -ll_cluster, cooccur


def sample_competing_exponentials(
    logits: torch.Tensor,
    time_unit: float,
    clamp_min: float = 0.0,
    clamp_max: float = 365.25 * 80.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (next token, time-to-next in days) by competing exponentials.

    ``exp(logits)`` are per-``time_unit`` intensities λ_v; each token's waiting
    time (1/λ_v)·Exp(1) is drawn in ``time_unit`` units, scaled to days by
    ``time_unit``, clamped to ``[clamp_min, clamp_max]`` (days), and the
    earliest-firing token wins. ``time_unit`` is the single source of truth for
    the intensity time scale and must match what the model was trained with
    (the same value the TPP uses in its compensator).
    """
    dt = -torch.exp(-logits) * torch.rand(logits.shape, device=logits.device).log()
    dt = (dt * time_unit).clamp(min=clamp_min, max=clamp_max)  # time_unit units -> days
    time_til_next, next_token = dt.min(dim=1)
    return next_token[:, None], time_til_next[:, None]


def sample_homo_cluster_poisson(
    logits: torch.Tensor,
    thresh_logits: torch.Tensor,
    time_unit: float,
    clamp_min: float = 0.0,
    clamp_max: float = 365.25 * 80.0,
):
    batch_size = logits.shape[0]
    assert thresh_logits.shape == (batch_size,)
    thresh_logits = thresh_logits.unsqueeze(-1)
    device = logits.device

    # waiting times are drawn in time_unit units, then scaled to days (see
    # sample_competing_exponentials); time_unit is the single source of truth.
    t_next = torch.clamp(
        -torch.exp(-logits) * torch.rand(logits.shape, device=device).log() * time_unit,
        min=clamp_min,
        max=clamp_max,
    )
    t_nod_next = torch.clamp(
        -torch.exp(-thresh_logits)
        * torch.rand(thresh_logits.shape, device=device).log()
        * time_unit,
        min=clamp_min,
        max=clamp_max,
    )
    sample_mask = t_next <= t_nod_next
    max_n = sample_mask.sum(dim=1).max().item()
    if max_n > 0:
        subject_idx, token_idx = torch.nonzero(sample_mask, as_tuple=True)
        pseudo_idx = sample_mask.cumsum(1) - 1
        pseudo_idx = pseudo_idx[sample_mask]

        next_token = torch.zeros((batch_size, int(max_n)), device=device).long()
        next_token[subject_idx, pseudo_idx] = token_idx

        no_event = (next_token == 0).all(dim=1)
        next_token[no_event, 0] = 1

        time_til_next = t_nod_next.expand(-1, int(max_n)).clone()
        time_til_next[next_token == 0] = -1e4

        sort_by_age = torch.argsort(time_til_next, dim=1)
        next_token = torch.take_along_dim(next_token, sort_by_age, dim=1)
        time_til_next = torch.take_along_dim(time_til_next, sort_by_age, dim=1)
    else:
        next_token = torch.ones(batch_size, 1, device=device).long()
        time_til_next = t_nod_next.expand(-1, 1).clone()

    return next_token, time_til_next
