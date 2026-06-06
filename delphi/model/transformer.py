import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn import functional as F

from delphi.model.tpp import (
    HomoPoissonTPP,
    NeuralIntensity,
    NeuralTPP,
)
from delphi.model.utils import (
    causal_attention_mask,
    exponential_nll,
    incremental_attention_mask,
    nll_homogeneous_cluster_poisson,
    sample_competing_exponentials,
    sample_homo_cluster_poisson,
    self_terminate_single,
    untie,
)


def _mask_invalid(
    loss: dict[str, torch.Tensor],
    is_valid: torch.Tensor,
    reduce: bool,
) -> dict[str, torch.Tensor]:
    """
    Nan-fill invalid positions on each per-position loss tensor, then optionally
    reduce with nanmean. Broadcasts ``is_valid`` up to match each tensor's rank.
    """
    out = {}
    for key, tensor in loss.items():
        mask = is_valid
        while mask.dim() < tensor.dim():
            mask = mask.unsqueeze(-1)
        value = tensor.masked_fill(~mask.expand_as(tensor), torch.nan)
        if reduce:
            value = torch.nanmean(value)
        out[key] = value
    return out


class AgeEncoding(nn.Module):

    def __init__(
        self, n_embd: int, norm_factor: float = 365.25, max_wavelen: float = 10000.0
    ):
        super().__init__()
        div_term = torch.exp(
            torch.arange(0, n_embd, 2) * (-math.log(max_wavelen) / n_embd)
        )
        self.register_buffer("div_term", div_term)
        self.n_embd = n_embd
        self.linear = torch.nn.Linear(n_embd, n_embd, bias=False)

        self.norm_factor = norm_factor

    def forward(self, x: torch.Tensor):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        time_years = x / self.norm_factor
        y = torch.zeros(
            x.shape[0], x.shape[1], self.n_embd, device=x.device, dtype=x.dtype
        )
        y[..., 0::2] = torch.sin(time_years * self.div_term)  # * (1-self.div_term)
        y[..., 1::2] = torch.cos(time_years * self.div_term)  # * (1-self.div_term)
        y = self.linear(y)

        return y


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x, attn_mask, past_kv=None, return_attn=False):
        B, T, C = x.size()
        # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)  # (B, nh, T_total, hs)
            v = torch.cat([past_kv[1], v], dim=2)  # (B, nh, T_total, hs)

        # Self-attend: (B, nh, T, hs) x (B, nh, hs, T_total) -> (B, nh, T, T_total).
        # Fast path (default) uses fused SDPA, which never materializes the
        # (B, nh, T, T_total) score matrix — avoiding the dominant eval-time
        # allocation — but cannot return attention weights. The manual path is kept
        # for callers that need `att` (e.g. attention visualization): return_attn=True.
        if return_attn:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(attn_mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T_total) x (B, nh, T_total, hs) -> (B, nh, T, hs)
        else:
            # attn_mask is a 0/1 keep-mask (nonzero = attend); SDPA boolean masks use
            # True = keep. Default scale is 1/sqrt(head_dim), matching the manual path.
            att = None
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask.to(torch.bool),
                dropout_p=self.dropout if self.training else 0.0,
            )
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, att, (k, v)


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask, past_kv=None, return_attn=False):
        y, att, new_kv = self.attn(
            self.ln_1(x), attn_mask, past_kv=past_kv, return_attn=return_attn
        )
        x = x + y
        x = x + self.mlp(self.ln_2(x))
        return x, att, new_kv


@dataclass
class Delphi2MConfig:
    # defaults to config of the OG delphi-2m ckpt
    block_size: None | int = 48
    vocab_size: int = 1270
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 120
    dropout: float = 0.1
    token_dropout: float = 0.0
    t_min: float = 0.1
    bias: bool = False
    mask_ties: bool = True
    attn_mask: str = "time"
    weight_tying: bool = True
    ignore_tokens: None | list = field(
        default_factory=lambda: [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    )  # 0 always ignored
    no_event_rate: None | float = None
    mask_no_event_attention: bool = False
    loss: str = "default"  # homo_poisson, homo_cluster_poisson
    time_unit: float = 1.0
    aux_head: str = "linear"
    ce_beta: float = 1.0
    dt_beta: float = 1.0
    self_terminate: bool = True
    self_terminate_except: None | list = field(default_factory=lambda: [1])
    n_integrate_grid: int = 20  # number of grid points for neural_tpp compensator

    def __post_init__(self):
        if "cluster" in self.loss:
            assert not self.mask_ties


class Delphi2M(nn.Module):

    model_type = "delphi-2m"

    def __init__(self, config: Delphi2MConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wae=AgeEncoding(n_embd=config.n_embd),
                token_drop=nn.Dropout(config.token_dropout),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        if config.loss in {"default", "homo_poisson", "homo_cluster_poisson"}:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            if self.config.weight_tying:
                self.transformer.wte.weight = self.lm_head.weight

        if config.loss == "neural_tpp":
            self.neural_tpp_head = NeuralIntensity(
                n_embd=config.n_embd,
                vocab_size=config.vocab_size,
                time_encoder=AgeEncoding(n_embd=config.n_embd),
            )

        if "cluster" in config.loss:
            if self.config.aux_head == "linear":
                self.aux_head = nn.Linear(config.n_embd, 1, bias=True)
            elif self.config.aux_head == "mlp":
                self.aux_head = nn.Sequential(
                    nn.Linear(config.n_embd, 32), nn.GELU(), nn.Linear(32, 1)
                )
            else:
                raise ValueError

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def targets(self):
        all = torch.arange(self.config.vocab_size)
        targets = all[~torch.isin(all, torch.tensor(self.config.ignore_tokens))]
        return targets

    def loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
        targets_age: torch.Tensor,
        reduce: bool = True,
    ):
        terminate_except = torch.tensor(
            self.config.self_terminate_except, device=targets.device
        )

        cooccur = None
        aux = None
        if self.config.loss == "default":
            if self.config.mask_ties:
                outputs, age = untie(outputs, outputs["age"], targets_age)
            else:
                age = outputs["age"]

            logits = outputs["logits"]
            loss_ce = F.cross_entropy(
                # (b, l, n_vocab) -> (b, n_vocab, l)
                logits.permute(0, 2, 1),
                targets,
                reduction="none",
            )
            dt = targets_age - age
            dt = torch.clamp(dt, min=self.config.t_min)
            loss_dt = exponential_nll(
                delta_t=dt,
                log_lambda=torch.logsumexp(logits, -1),
                t_min=self.config.t_min,
            )
            loss = {"loss_ce": loss_ce, "loss_dt": loss_dt}
        elif self.config.loss == "homo_poisson":
            tpp = HomoPoissonTPP(
                hidden_states=outputs["h"],
                logits=outputs["logits"],
                timesteps=outputs["age"],
                tokens=outputs["idx"],
                terminate_except=terminate_except,
                time_unit=self.config.time_unit,
            )
            log_p = tpp.log_likelihood(x1=targets, t1=targets_age)
            loss = {"loss_nll": -log_p}
        elif self.config.loss == "homo_cluster_poisson":
            logits, thresh_logits = outputs["logits"], outputs["aux_rates"]
            nll, nll_cluster, cooccur = nll_homogeneous_cluster_poisson(
                log_intensity=logits,
                log_aux_intensity=thresh_logits,
                targets=targets,
                targets_age=targets_age,
                age=outputs["age"],
            )
            loss = {"loss_nll": nll, "loss_cluster": nll_cluster}
        elif self.config.loss == "neural_tpp":
            tpp = NeuralTPP(
                hidden_states=outputs["h"],
                intensity_func=self.neural_tpp_head,
                timesteps=outputs["age"],
                tokens=outputs["idx"],
                n_grid=self.config.n_integrate_grid,
            )
            log_p = tpp.log_likelihood(
                x1=targets,
                t1=targets_age,
            )
            loss = {"loss_nll": -log_p}
        else:
            raise NotImplementedError

        is_valid = targets != 0
        if self.config.ignore_tokens is not None:
            for k in self.config.ignore_tokens:
                is_valid &= targets != k
        if cooccur is not None:
            is_valid &= ~cooccur

        return _mask_invalid(loss, is_valid, reduce=reduce), aux

    def forward(
        self,
        idx,
        age,
        targets=None,
        targets_age=None,
        past_kvs=None,
        past_pad=None,
        return_attn=False,
    ):
        tok_emb = self.transformer.wte(idx)
        age_emb = self.transformer.wae(age.unsqueeze(-1))
        x = self.transformer.token_drop(tok_emb) * (1 - self.config.token_dropout)
        x = x + age_emb
        x = self.transformer.drop(x)

        pad = idx > 0
        if self.config.mask_no_event_attention:
            pad = idx > 1

        if past_kvs is not None:
            attn_mask = incremental_attention_mask(new_pad=pad, past_pad=past_pad)
        elif self.config.attn_mask == "triangular":
            attn_mask = causal_attention_mask(pad=pad)
        else:
            attn_mask = causal_attention_mask(pad=pad, timestep=age)

        att = [] if return_attn else None
        new_kvs = []
        for i, block in enumerate(self.transformer.h):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, a, new_kv = block(x, attn_mask, past_kv=past_kv, return_attn=return_attn)
            if return_attn:
                att.append(a)
            new_kvs.append(new_kv)
        x = self.transformer.ln_f(x)

        misc = dict()
        misc["attn_mask"] = attn_mask
        misc["attn"] = torch.stack(att) if return_attn else None
        misc["past_kvs"] = new_kvs
        misc["cur_pad"] = pad

        outputs = dict()
        outputs["age"] = age
        outputs["idx"] = idx
        outputs["h"] = x

        if hasattr(self, "lm_head"):
            logits = self.lm_head(x)
            if self.config.no_event_rate is not None:
                logits[..., 1] = math.log(self.config.no_event_rate)
            outputs["logits"] = logits

        if hasattr(self, "aux_head"):
            aux_rates = self.aux_head(x)
            outputs["aux_rates"] = aux_rates.squeeze(-1)

        if (targets is not None) and (targets_age is not None):
            loss, aux = self.loss(outputs, targets=targets, targets_age=targets_age)
            if aux:
                outputs = outputs | aux
        else:
            loss = None

        return outputs, loss, misc

    @torch.no_grad()
    def sample_next(self, outputs: dict[str, torch.Tensor], idx: torch.Tensor):
        if self.config.loss in {"default", "homo_poisson"}:
            logits = outputs["logits"][:, -1, :]
            logits = self_terminate_single(
                idx=idx,
                logits=logits,
                terminate_except=torch.tensor(self.config.self_terminate_except).to(
                    idx.device
                ),
            )
            idx_next, time_til_next = sample_competing_exponentials(logits=logits)
            time_til_next *= 365.25
        elif self.config.loss == "homo_cluster_poisson":
            logits = outputs["logits"][:, -1, :]
            logits = self_terminate_single(
                idx=idx,
                logits=logits,
                terminate_except=torch.tensor(self.config.self_terminate_except).to(
                    idx.device
                ),
            )
            idx_next, time_til_next = sample_homo_cluster_poisson(
                logits=logits, thresh_logits=outputs["aux_rates"][:, -1]
            )
        else:
            raise NotImplementedError
        return idx_next, time_til_next


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    age: torch.Tensor,
    termination_tokens: list | torch.Tensor,
    max_new_tokens: None | int | float = None,
    max_age: None | float | torch.Tensor = 85 * 365.25,
    stop_at_block_size: bool = True,
    exclude_pad: bool = True,
    cached: bool = True,
    **kwargs,
):

    termination_tokens = torch.tensor(
        termination_tokens, dtype=torch.int64, device=idx.device
    )

    if max_new_tokens is None:
        max_new_tokens = float("inf")

    if max_age is None:
        pass
    elif isinstance(max_age, torch.Tensor):
        assert len(max_age.shape) == 1
        assert max_age.shape[0] == age.shape[0]
        max_age = max_age.unsqueeze(1)
    else:
        max_age = torch.full((age.shape[0], 1), fill_value=max_age).to(idx.device)

    batch_size = idx.shape[0]
    active_indices = torch.arange(batch_size, device=idx.device)
    completed_idx, completed_age, completed_mask = dict(), dict(), dict()
    cur_idx = idx.clone()
    cur_age = age.clone()
    # mask rides alongside cur_idx through every cat/filter/sort/trim below,
    # so it stays aligned with the returned idx/age. Encoding: 0=pad,
    # 1=prompt (set here), 2=continuation (appended per step), 3=censored
    # (set after the age cap). The pad-is-0 invariant is re-asserted before return.
    cur_mask = (cur_idx > 0).long()

    ignore_tokens = [0]
    if (
        hasattr(model.config, "ignore_tokens")
        and model.config.ignore_tokens is not None
    ):
        ignore_tokens += model.config.ignore_tokens

    pmt_cnt = (idx > 0).sum(dim=1)
    gen_cnt = torch.zeros_like(pmt_cnt)

    cache_kvs = None  # list of (k, v) per layer; None triggers full pass
    cache_pad = None  # (B_active, T_cached) bool

    while len(active_indices) > 0:
        if not cached or cache_kvs is None:
            outputs, _, misc = model(cur_idx, cur_age, **kwargs)
            kwargs = {}  # only pass on first call
        else:
            outputs, _, misc = model(
                idx_next, age_next, past_kvs=cache_kvs, past_pad=cache_pad
            )
            cache_pad = torch.cat([cache_pad, misc["cur_pad"]], dim=1)

        if cached:
            cache_kvs = misc["past_kvs"]
            if cache_pad is None:
                cache_pad = misc["cur_pad"]

        idx_next, time_til_next = model.sample_next(outputs=outputs, idx=cur_idx)
        age_next = cur_age[..., [-1]] + time_til_next
        age_next[time_til_next == -1e4] = -1e4

        gen_cnt[active_indices] += (idx_next > 0).sum(dim=1)
        cur_idx = torch.cat((cur_idx, idx_next), dim=1)
        cur_age = torch.cat((cur_age, age_next), dim=1)
        cur_mask = torch.cat((cur_mask, (idx_next > 0).long() * 2), dim=1)

        terminated = torch.isin(idx_next, termination_tokens).any(-1)
        if max_age is None:
            aged_out = torch.zeros_like(terminated)
        else:
            aged_out = (age_next > max_age[active_indices]).any(-1)
        if stop_at_block_size and (model.config.block_size is not None):
            # cur_idx includes the newly added token
            if exclude_pad:
                block_size = (cur_idx != 0).sum(dim=1)
            else:
                block_size = torch.full_like(
                    active_indices, fill_value=cur_idx.shape[1]
                )
            reached_block = block_size >= model.config.block_size
        else:
            reached_block = torch.zeros_like(terminated)
        maxed_out = gen_cnt[active_indices] >= max_new_tokens
        should_stop = terminated | aged_out | reached_block | maxed_out

        if should_stop.any():
            # identify indices relative to the current active batch
            stop_indices = torch.where(should_stop)[0]
            for local_i in stop_indices:
                global_i = active_indices[local_i].item()
                completed_idx[global_i] = cur_idx[local_i]
                completed_age[global_i] = cur_age[local_i]
                completed_mask[global_i] = cur_mask[local_i]
            # filter the running batch to keep only unfinished sequences
            cur_idx = cur_idx[~should_stop]
            cur_age = cur_age[~should_stop]
            cur_mask = cur_mask[~should_stop]
            active_indices = active_indices[~should_stop]
            if cached and cache_kvs is not None:
                cache_kvs = [(k[~should_stop], v[~should_stop]) for k, v in cache_kvs]
                cache_pad = cache_pad[~should_stop]
                idx_next = idx_next[~should_stop]
                age_next = age_next[~should_stop]

        if len(active_indices) == 0:
            break

    max_len = max(t.numel() for t in completed_idx.values())
    final_idx = torch.full((batch_size, max_len), 0, dtype=idx.dtype, device=idx.device)
    final_age = torch.full(
        (batch_size, max_len), -1e4, dtype=age.dtype, device=age.device
    )
    # left-pad fill stays 0 == pad
    final_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=idx.device)
    for i in range(batch_size):
        idx_i, age_i, mask_i = completed_idx[i], completed_age[i], completed_mask[i]
        final_idx[i, -idx_i.numel() :] = idx_i
        final_age[i, -age_i.numel() :] = age_i
        final_mask[i, -mask_i.numel() :] = mask_i

    if max_age is not None:
        censored = final_age > max_age
        final_idx[censored] = 1
        final_mask[censored] = 3
        final_age = torch.clamp(final_age, max=max_age)

    sort_by_age = torch.argsort(final_age, dim=1)
    age = torch.take_along_dim(input=final_age, indices=sort_by_age, dim=1)
    idx = torch.take_along_dim(input=final_idx, indices=sort_by_age, dim=1)
    mask = torch.take_along_dim(input=final_mask, indices=sort_by_age, dim=1)

    margin = torch.min(torch.sum(idx == 0, dim=1)).item()
    idx, age, mask = idx[:, margin:], age[:, margin:], mask[:, margin:]
    # re-assert the pad-is-0 invariant exactly after all transforms
    mask = mask.masked_fill((idx == 0) | (age == -1e4), 0)

    return (
        idx,
        age,
        {
            "n_prompt": pmt_cnt.detach().cpu().numpy(),
            "n_gen": gen_cnt.detach().cpu().numpy(),
            # (B, L) long, aligned to idx/age: 0=pad 1=prompt 2=continuation 3=censored
            "mask": mask,
        },
    )
