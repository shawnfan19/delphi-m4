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


def untie_idx(age: torch.Tensor, targets_age: torch.Tensor):
    dt = targets_age - age
    is_tie = dt == 0
    is_tie[age == -1e4] = False
    corr_idx = torch.where(is_tie, 0, torch.arange(age.shape[1], device=age.device))
    corr_idx = torch.cummax(corr_idx, dim=1)[0]
    return corr_idx


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
    delta_t: torch.Tensor,
):

    part1 = torch.gather(input=log_intensity, dim=-1, index=targets.unsqueeze(-1))
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
    else:
        next_token = torch.ones(batch_size, 1, device=device).long()
        time_til_next = t_nod_next.expand(-1, 1).clone()

    return next_token, time_til_next


def nll_gompertz_hawkes(
    A: torch.Tensor,
    B: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    age: torch.Tensor,
    targets_age: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the negative log-likelihood for a Gompertz-Hawkes mixture TPP.

    Intensity for event type v at time τ in interval [t_i, t_{i+1}]:
        λ_v(τ) = A_v · exp(B_v · τ) + α_v · exp(-β_v · (τ - t_i))
                 \___ aging ___/     \____ acute trigger ____/

    Log-likelihood:
        LL = log λ_k(t_{i+1}) - ∫_{t_i}^{t_{i+1}} Σ_v λ_v(τ) dτ

    Args:
        A: Gompertz amplitude, shape (B, L, V), positive
        B: Gompertz growth rate, shape (B, L, V), positive
        alpha: Hawkes excitation amplitude, shape (B, L, V), positive
        beta: Hawkes decay rate, shape (B, L, V), positive
        age: Interval start time t_i, shape (B, L)
        targets_age: Interval end time t_{i+1}, shape (B, L)
        targets: Event type indices, shape (B, L), dtype=long

    Returns:
        NLL tensor of shape (B, L)
    """
    delta_t = targets_age - age
    eps = 1e-8

    # ================================================================
    # Part 1: log λ_k(t_{i+1}) — point process term
    # ================================================================

    idx = targets.unsqueeze(-1)  # (B, L, 1)
    A_k = torch.gather(A, dim=-1, index=idx).squeeze(-1)
    B_k = torch.gather(B, dim=-1, index=idx).squeeze(-1)
    alpha_k = torch.gather(alpha, dim=-1, index=idx).squeeze(-1)
    beta_k = torch.gather(beta, dim=-1, index=idx).squeeze(-1)

    # Use logsumexp for numerical stability:
    # log(A·e^x + α·e^y) = logsumexp([log A + x, log α + y])
    log_gompertz = torch.log(A_k + eps) + B_k * targets_age
    log_hawkes = torch.log(alpha_k + eps) - beta_k * delta_t

    part1 = torch.logsumexp(torch.stack([log_gompertz, log_hawkes], dim=-1), dim=-1)

    # ================================================================
    # Part 2: -∫_{t_i}^{t_{i+1}} Σ_v λ_v(τ) dτ — survival term
    # ================================================================

    age_exp = age.unsqueeze(-1)
    targets_age_exp = targets_age.unsqueeze(-1)
    delta_t_exp = delta_t.unsqueeze(-1)

    # Gompertz integral: (A/B)·[exp(B·t_{i+1}) - exp(B·t_i)]
    gompertz_integral = (A / (B + eps)) * (
        torch.exp(B * targets_age_exp) - torch.exp(B * age_exp)
    )

    # Hawkes integral: (α/β)·[1 - exp(-β·Δt)]
    hawkes_integral = (alpha / (beta + eps)) * (1.0 - torch.exp(-beta * delta_t_exp))

    # Sum over all event types v
    total_integral = (gompertz_integral + hawkes_integral).sum(dim=-1)

    part2 = -total_integral

    # ================================================================
    # NLL = -(part1 + part2)
    # ================================================================

    return -(part1 + part2)
