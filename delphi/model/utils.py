import torch


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


def untie_idx(age: torch.Tensor, targets_age: torch.Tensor):
    dt = targets_age - age
    is_tie = dt == 0
    is_tie[age == -1e4] = False
    corr_idx = torch.where(is_tie, 0, torch.arange(age.shape[1], device=age.device))
    corr_idx = torch.cummax(corr_idx, dim=1)[0]
    return corr_idx


def untie(
    outputs: dict[str, torch.Tensor], age: torch.Tensor, targets_age: torch.Tensor
):
    corr_idx = untie_idx(age, targets_age)
    age = torch.take_along_dim(input=age, indices=corr_idx, dim=1)
    batch_size = corr_idx.shape[0]
    for key, tensor in outputs.items():
        if tensor.dim() <= 1 or tensor.shape[0] != batch_size:
            continue

        if tensor.dim() > 2:
            indices = corr_idx.unsqueeze(-1)
        else:
            indices = corr_idx
        outputs[key] = torch.take_along_dim(input=outputs[key], indices=indices, dim=1)
    return outputs, age


def nearest_input_pos(age, targets_age):
    """
    For each target, find the position of the nearest input strictly before it;
    on ties, pick the latest tied position (last in sequence order).

    Returns -1 for targets with no strictly-earlier input position. Callers that
    cannot tolerate -1 should clamp after calling (e.g., ``.clamp(min=0)``).
    """

    L = age.shape[-1]
    targets_age = targets_age.view(*targets_age.shape, 1)
    age = age.view(age.shape[0], 1, age.shape[1])
    age_diff = targets_age - age  # B, L1, L0
    age_diff = age_diff.masked_fill(age_diff <= 0, float("inf"))
    pos = L - 1 - torch.argmin(age_diff.flip(-1), dim=-1)
    # argmin on an all-inf row returns 0 (→ pos = L-1), which is indistinguishable
    # from a valid match at the last position. Emit -1 for those rows instead.
    no_valid = torch.isinf(age_diff).all(dim=-1)
    pos = pos.masked_fill(no_valid, -1)

    return pos


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


def exponential_nll(
    delta_t: torch.Tensor,
    log_lambda: torch.Tensor,
    t_min: float,
    n: None | torch.Tensor = None,
):
    """
    when n > 1, return nll according to the erlang distribution
    """
    ldt = -torch.log(delta_t + t_min)
    lse = -torch.log(torch.exp(-log_lambda) + t_min)
    # when n == 1: nll = -(lse - torch.exp(lse - ldt))
    if n is None:
        n = torch.ones_like(delta_t)
    nll = -(n * lse + (n - 1) * (-ldt) - torch.exp(lse - ldt) - torch.lgamma(n))
    return nll


def nll_homogeneous_poisson(
    log_intensity: torch.Tensor,
    targets: torch.Tensor,
    idx: torch.Tensor,
    targets_age: torch.Tensor,
    age: torch.Tensor,
    terminate: bool,
    terminate_except: torch.Tensor,
):

    delta_t = targets_age - age
    assert delta_t.min() >= 0

    part1 = torch.gather(input=log_intensity, dim=-1, index=targets.unsqueeze(-1))

    if terminate:
        log_intensity = self_terminate(
            idx=idx,
            estimator=log_intensity,
            terminate_except=terminate_except,
            fill_val=float("-inf"),
        )

    log_sum_intensity = torch.logsumexp(log_intensity, dim=-1, keepdim=True)
    part2 = -torch.exp(log_sum_intensity) * delta_t.unsqueeze(-1)

    return -(part1 + part2)


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
    logits: torch.Tensor, clamp_min: float = 0.0, clamp_max: float = 365.25 * 80.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    inverse CDF method
    """

    t_next = torch.clamp(
        -torch.exp(-logits) * torch.rand(logits.shape, device=logits.device).log(),
        min=clamp_min,
        max=clamp_max,
    ).min(1)
    next_token = t_next[1][:, None]
    time_til_next = t_next[0][:, None]

    return next_token, time_til_next


def sample_homo_cluster_poisson(
    logits: torch.Tensor,
    thresh_logits: torch.Tensor,
    clamp_min: float = 0.0,
    clamp_max: float = 365.25 * 80.0,
):
    batch_size = logits.shape[0]
    assert thresh_logits.shape == (batch_size,)
    thresh_logits = thresh_logits.unsqueeze(-1)
    device = logits.device

    t_next = torch.clamp(
        -torch.exp(-logits) * torch.rand(logits.shape, device=device).log(),
        min=clamp_min,
        max=clamp_max,
    )
    t_nod_next = torch.clamp(
        -torch.exp(-thresh_logits)
        * torch.rand(thresh_logits.shape, device=device).log(),
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
