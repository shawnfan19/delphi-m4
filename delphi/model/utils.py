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


def nll_hawkes(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    age: torch.Tensor,
    targets_age: torch.Tensor,
    targets: torch.Tensor,
    time_unit: float = 365.25,
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
    eps = 1e-8

    delta_t = targets_age - age
    delta_t /= time_unit

    # ================================================================
    # Part 1: log λ_k(t_{i+1})
    # ================================================================

    idx = targets.unsqueeze(-1)
    alpha_k = torch.gather(alpha, dim=-1, index=idx).squeeze(-1)
    beta_k = torch.gather(beta, dim=-1, index=idx).squeeze(-1)

    # log(α · exp(-β · Δt)) = log(α) - β · Δt
    part1 = torch.log(alpha_k + eps) - beta_k * delta_t

    # ================================================================
    # Part 2: -∫_{t_i}^{t_{i+1}} Σ_v λ_v(τ) dτ
    # ================================================================

    delta_t_exp = delta_t.unsqueeze(-1)

    # ∫ α·exp(-β·(τ-t_i)) dτ = (α/β)·[1 - exp(-β·Δt)]
    integral = (alpha / (beta + eps)) * (1.0 - torch.exp(-beta * delta_t_exp))

    part2 = -integral.sum(dim=-1)

    return -(part1 + part2)


def nll_weibull(
    weibull_A: torch.Tensor,
    weibull_k: torch.Tensor,
    weibull_lam: torch.Tensor,
    age: torch.Tensor,
    targets_age: torch.Tensor,
    targets: torch.Tensor,
    time_unit: float = 365.25,
) -> torch.Tensor:
    """
    NLL for Weibull kernel intensity:
        λ_v(Δt) = A_v · (k_v/λ_v) · (Δt/λ_v)^(k_v-1) · exp(-(Δt/λ_v)^k_v)

    where Δt = t - t_i is the time since the previous event.

    All parameters are context-dependent (output by the transformer).

    Args:
        weibull_A: Amplitude, shape (B, L, V), positive
        weibull_k: Shape parameter, shape (B, L, V), positive
        weibull_lam: Scale parameter, shape (B, L, V), positive
        age: Interval start t_i in days, shape (B, L)
        targets_age: Interval end t_{i+1} in days, shape (B, L)
        targets: Event type indices, shape (B, L)
        time_unit: Time normalization factor (days per unit)

    Returns:
        NLL tensor of shape (B, L)
    """
    eps = 1e-8

    delta_t = (targets_age - age) / time_unit  # (B, L)
    delta_t = torch.clamp(delta_t, min=eps)

    # ================================================================
    # Part 1: log λ_k(Δt) for the observed target event
    # = log(A_k) + log(k_k) - log(λ_k) + (k_k-1)·log(Δt/λ_k) - (Δt/λ_k)^k_k
    # ================================================================
    idx = targets.unsqueeze(-1)  # (B, L, 1)
    A_k = torch.gather(weibull_A, dim=-1, index=idx).squeeze(-1)  # (B, L)
    k_k = torch.gather(weibull_k, dim=-1, index=idx).squeeze(-1)  # (B, L)
    lam_k = torch.gather(weibull_lam, dim=-1, index=idx).squeeze(-1)  # (B, L)

    log_dt_over_lam_k = torch.log(delta_t) - torch.log(lam_k)  # (B, L)
    dt_over_lam_k_pow_k = torch.exp(k_k * log_dt_over_lam_k)  # (Δt/λ_k)^k_k

    part1 = (
        torch.log(A_k + eps)
        + torch.log(k_k + eps)
        - torch.log(lam_k + eps)
        + (k_k - 1) * log_dt_over_lam_k
        - dt_over_lam_k_pow_k
    )  # (B, L)

    # ================================================================
    # Part 2: -∫_0^Δt Σ_v λ_v(τ) dτ = -Σ_v A_v · [1 - exp(-(Δt/λ_v)^k_v)]
    # ================================================================
    delta_t_exp = delta_t.unsqueeze(-1)  # (B, L, 1)
    log_dt_over_lam = torch.log(delta_t_exp) - torch.log(weibull_lam)  # (B, L, V)
    dt_over_lam_pow_k = torch.exp(weibull_k * log_dt_over_lam)  # (Δt/λ_v)^k_v

    compensator = weibull_A * (1.0 - torch.exp(-dt_over_lam_pow_k))  # (B, L, V)
    part2 = -compensator.sum(dim=-1)  # (B, L)

    return -(part1 + part2)


def nll_weibull_lite(
    weibull_k: torch.Tensor,
    weibull_lam: torch.Tensor,
    age: torch.Tensor,
    targets_age: torch.Tensor,
    targets: torch.Tensor,
    time_unit: float = 365.25,
) -> torch.Tensor:
    """
    NLL for pure Weibull PDF kernel (no amplitude parameter):
        λ_v(Δt) = (k_v/λ_v) · (Δt/λ_v)^(k_v-1) · exp(-(Δt/λ_v)^k_v)

    Equivalent to nll_weibull with A=1 for all event types.

    Args:
        weibull_k: Shape parameter, shape (B, L, V), positive
        weibull_lam: Scale parameter, shape (B, L, V), positive
        age: Interval start t_i in days, shape (B, L)
        targets_age: Interval end t_{i+1} in days, shape (B, L)
        targets: Event type indices, shape (B, L)
        time_unit: Time normalization factor (days per unit)

    Returns:
        NLL tensor of shape (B, L)
    """
    eps = 1e-8

    delta_t = (targets_age - age) / time_unit  # (B, L)
    delta_t = torch.clamp(delta_t, min=eps)

    # ================================================================
    # Part 1: log λ_k(Δt) for the observed target event
    # = log(k_k) - log(λ_k) + (k_k-1)·log(Δt/λ_k) - (Δt/λ_k)^k_k
    # ================================================================
    idx = targets.unsqueeze(-1)  # (B, L, 1)
    k_k = torch.gather(weibull_k, dim=-1, index=idx).squeeze(-1)  # (B, L)
    lam_k = torch.gather(weibull_lam, dim=-1, index=idx).squeeze(-1)  # (B, L)

    log_dt_over_lam_k = torch.log(delta_t) - torch.log(lam_k)  # (B, L)
    dt_over_lam_k_pow_k = torch.exp(k_k * log_dt_over_lam_k)  # (Δt/λ_k)^k_k

    part1 = (
        torch.log(k_k + eps)
        - torch.log(lam_k + eps)
        + (k_k - 1) * log_dt_over_lam_k
        - dt_over_lam_k_pow_k
    )  # (B, L)

    # ================================================================
    # Part 2: -∫_0^Δt Σ_v λ_v(τ) dτ = -Σ_v [1 - exp(-(Δt/λ_v)^k_v)]
    # ================================================================
    delta_t_exp = delta_t.unsqueeze(-1)  # (B, L, 1)
    log_dt_over_lam = torch.log(delta_t_exp) - torch.log(weibull_lam)  # (B, L, V)
    dt_over_lam_pow_k = torch.exp(weibull_k * log_dt_over_lam)  # (Δt/λ_v)^k_v

    compensator = 1.0 - torch.exp(-dt_over_lam_pow_k)  # (B, L, V)
    part2 = -compensator.sum(dim=-1)  # (B, L)

    return -(part1 + part2)


def nll_hawkes_weibull(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    weibull_k: torch.Tensor,
    weibull_lam: torch.Tensor,
    weibull_A: torch.Tensor,
    age: torch.Tensor,
    targets_age: torch.Tensor,
    targets: torch.Tensor,
    time_unit: float = 365.25,
) -> torch.Tensor:
    """
    NLL for Hawkes + Weibull baseline intensity:
        λ_v(t) = μ_v(t) + α_v · exp(-β_v · (t - t_i))

    where μ_v(t) is the Weibull density baseline:
        μ_v(t) = A_v · (k_v/λ_v) · (t/λ_v)^(k_v-1) · exp(-(t/λ_v)^k_v)

    Args:
        alpha: Excitation amplitude, shape (B, L, V), positive
        beta: Decay rate, shape (B, L, V), positive
        weibull_k: Weibull shape per event type, shape (V,), positive
        weibull_lam: Weibull scale per event type, shape (V,), positive
        weibull_A: Weibull amplitude per event type, shape (V,), positive
        age: Interval start t_i in days, shape (B, L)
        targets_age: Interval end t_{i+1} in days, shape (B, L)
        targets: Event type indices, shape (B, L)
        time_unit: Time normalization factor (days per unit)

    Returns:
        NLL tensor of shape (B, L)
    """
    eps = 1e-8

    # Normalize times
    t_i = age / time_unit
    t_next = targets_age / time_unit
    delta_t = t_next - t_i

    # ================================================================
    # Weibull baseline at target time: μ_v(t_{i+1})
    # μ_v(t) = A · (k/λ) · (t/λ)^(k-1) · exp(-(t/λ)^k)
    # Computed in log-space for stability
    # ================================================================
    # All Weibull params are (V,), broadcast over (B, L)
    t_next_pos = torch.clamp(t_next, min=eps)  # (B, L)
    log_t_over_lam = torch.log(t_next_pos).unsqueeze(-1) - torch.log(
        weibull_lam
    )  # (B, L, V)
    t_over_lam_k = torch.exp(weibull_k * log_t_over_lam)  # (t/λ)^k in (B, L, V)

    log_baseline = (
        torch.log(weibull_A)
        + torch.log(weibull_k)
        - torch.log(weibull_lam)
        + (weibull_k - 1) * log_t_over_lam
        - t_over_lam_k
    )  # (B, L, V)
    baseline = torch.exp(log_baseline)  # μ_v(t_{i+1})

    # ================================================================
    # Part 1: log λ_k(t_{i+1}) = log(μ_k(t_{i+1}) + α_k · exp(-β_k · Δt))
    # ================================================================
    idx = targets.unsqueeze(-1)
    alpha_k = torch.gather(alpha, dim=-1, index=idx).squeeze(-1)  # (B, L)
    beta_k = torch.gather(beta, dim=-1, index=idx).squeeze(-1)  # (B, L)
    baseline_k = torch.gather(baseline, dim=-1, index=idx).squeeze(-1)  # (B, L)

    excitation_k = alpha_k * torch.exp(-beta_k * delta_t)  # (B, L)
    part1 = torch.log(baseline_k + excitation_k + eps)

    # ================================================================
    # Part 2: -∫_{t_i}^{t_{i+1}} Σ_v μ_v(τ) dτ  (Weibull compensator)
    # = -Σ_v A_v · [exp(-(t_i/λ_v)^k_v) - exp(-(t_{i+1}/λ_v)^k_v)]
    # ================================================================
    t_i_pos = torch.clamp(t_i, min=eps)
    log_ti_over_lam = torch.log(t_i_pos).unsqueeze(-1) - torch.log(
        weibull_lam
    )  # (B, L, V)
    ti_over_lam_k = torch.exp(weibull_k * log_ti_over_lam)  # (t_i/λ)^k

    # t_over_lam_k is already (t_{i+1}/λ)^k from above
    weibull_integral = weibull_A * (
        torch.exp(-ti_over_lam_k) - torch.exp(-t_over_lam_k)
    )
    part2 = -weibull_integral.sum(dim=-1)  # (B, L)

    # ================================================================
    # Part 3: -∫_{t_i}^{t_{i+1}} Σ_v α_v · exp(-β_v · (τ-t_i)) dτ
    # = -Σ_v (α_v/β_v) · [1 - exp(-β_v · Δt)]
    # ================================================================
    delta_t_exp = delta_t.unsqueeze(-1)  # (B, L, 1)
    excitation_integral = (alpha / (beta + eps)) * (
        1.0 - torch.exp(-beta * delta_t_exp)
    )
    part3 = -excitation_integral.sum(dim=-1)  # (B, L)

    return -(part1 + part2 + part3)
