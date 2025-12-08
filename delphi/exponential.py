import torch
import torch.nn.functional as F


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


def sample_zlpr_competing_exponentials(
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
    subject_idx, token_idx = torch.nonzero(sample_mask, as_tuple=True)
    pseudo_idx = sample_mask.cumsum(1) - 1
    pseudo_idx = pseudo_idx[sample_mask]

    next_token = torch.zeros((batch_size, int(max_n)), device=device).long()
    next_token[subject_idx, pseudo_idx] = token_idx

    time_til_next = t_nod_next.expand(-1, int(max_n))
    time_til_next[next_token == 0] = -1e4

    return next_token, time_til_next


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


def sample_zero_inflated_exponentials(
    logits: torch.Tensor, pi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:

    next_token, time_til_next = sample_competing_exponentials(logits)

    pi = torch.sigmoid(pi)
    is_comorbid = torch.bernoulli(pi).to(torch.bool)
    time_til_next[is_comorbid] = 0.0
    next_token[is_comorbid.squeeze(-1)] = torch.multinomial(
        F.softmax(logits[is_comorbid.squeeze(-1), :], dim=-1), num_samples=1
    )

    return next_token, time_til_next
