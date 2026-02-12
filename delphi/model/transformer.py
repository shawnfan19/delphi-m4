import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from delphi.data.utils import collate_batch
from delphi.model.utils import (
    causal_attention_mask,
    exponential_nll,
    nll_homogeneous_cluster_poisson,
    nll_homogeneous_poisson,
    sample_competing_exponentials,
    sample_homo_cluster_poisson,
    untie_idx,
)


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
        y = torch.zeros(x.shape[0], x.shape[1], self.n_embd, device=x.device)
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

    def forward(self, x, attn_mask):
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

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = att.masked_fill(attn_mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, att


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

    def forward(self, x, attn_mask):
        y, att = self.attn(self.ln_1(x), attn_mask)
        x = x + y
        x = x + self.mlp(self.ln_2(x))
        return x, att


@dataclass
class Delphi2MConfig:
    # defaults to config of the OG delphi-2m ckpt
    # additional flags:
    # – ce_beta
    # - dt_beta
    # - mask_no_event_attention
    # - no_event_rate
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
        default_factory=lambda: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    )  # 0 always ignored
    no_event_rate: None | float = None
    mask_no_event_attention: bool = False
    loss: str = "default"  # homo_poisson, homo_cluster_poisson
    aux_head: str = "linear"
    ce_beta: float = 1.0
    dt_beta: float = 1.0

    def __post_init__(self):
        if "cluster" in self.loss:
            assert not self.mask_ties


class Delphi2M(nn.Module):
    """
    slightly cleaned up version of delphi-2m with extra features:
        - zero inflation
        - fix no-event rate as a model parameter
        - mask attention to previous no-event tokens
    """

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
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        if self.config.weight_tying:
            self.transformer.wte.weight = (
                self.lm_head.weight
            )  # https://paperswithcode.com/method/weight-tying

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
        logits: torch.Tensor,
        targets: torch.Tensor,
        age: torch.Tensor,
        targets_age: torch.Tensor,
        **kwargs,
    ):

        if self.config.mask_ties:
            corr_idx = untie_idx(age, targets_age)
            age = torch.take_along_dim(input=age, indices=corr_idx, dim=1)
            logits = torch.take_along_dim(
                input=logits, indices=corr_idx.unsqueeze(-1), dim=1
            )

        if self.config.loss == "default":
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

            return {"loss_ce": loss_ce, "loss_dt": loss_dt}
        elif self.config.loss == "homo_poisson":
            dt = targets_age - age
            dt = torch.clamp(dt, min=0)
            nll = nll_homogeneous_poisson(
                log_intensity=logits, targets=targets, delta_t=dt
            )

            return {"loss_nll": nll}
        elif self.config.loss == "homo_cluster_poisson":
            thresh_logits = kwargs["aux_rates"]
            nll, nll_cluster, cooccur = nll_homogeneous_cluster_poisson(
                log_intensity=logits,
                log_aux_intensity=thresh_logits,
                targets=targets,
                targets_age=targets_age,
                age=age,
            )
            return {"loss_nll": nll, "loss_cluster": nll_cluster, "mask": ~cooccur}
        else:
            raise NotImplementedError

    def forward(self, idx, age, targets=None, targets_age=None):
        tok_emb = self.transformer.wte(idx)
        age_emb = self.transformer.wae(age.unsqueeze(-1))
        x = self.transformer.token_drop(tok_emb) * (1 - self.config.token_dropout)
        x = x + age_emb
        x = self.transformer.drop(x)

        pad = idx > 0
        if self.config.mask_no_event_attention:
            pad = idx > 1

        if self.config.attn_mask == "triangular":
            attn_mask = causal_attention_mask(pad=pad)
        else:
            attn_mask = causal_attention_mask(pad=pad, timestep=age)

        att = []
        for block in self.transformer.h:
            x, a = block(x, attn_mask)
            att.append(a)
        x = self.transformer.ln_f(x)
        att = torch.stack(att)

        logits = self.lm_head(x)
        if self.config.no_event_rate is not None:
            logits[..., 1] = math.log(self.config.no_event_rate)

        outputs = dict()
        outputs["logits"] = logits
        outputs["attn_mask"] = attn_mask
        if hasattr(self, "aux_head"):
            aux_rates = self.aux_head(x)
            outputs["aux_rates"] = aux_rates.squeeze(-1)

        if (targets is not None) and (targets_age is not None):

            ignored_tokens = [0]
            if self.config.ignore_tokens is not None:
                ignored_tokens += self.config.ignore_tokens.copy()
            is_valid_target = targets != 0
            for k in ignored_tokens:
                is_valid_target *= targets != k
            loss = self.loss(
                targets=targets, age=age, targets_age=targets_age, **outputs
            )
            loss_mask = is_valid_target
            if "mask" in loss:
                loss_mask *= loss["mask"]
                del loss["mask"]
            for loss_key in loss.keys():
                loss[loss_key] = torch.mean(loss[loss_key][loss_mask])
        else:
            loss = None

        return outputs, loss, att

    @torch.no_grad()
    def sample_next(self, logits: torch.Tensor, outputs: dict[str, torch.Tensor]):
        if self.config.loss in {"default", "homo_poisson"}:
            idx_next, time_til_next = sample_competing_exponentials(logits=logits)
        elif self.config.loss == "homo_cluster_poisson":
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
    max_new_tokens: None | int = 100,
    max_age: float | torch.Tensor = 85 * 365.25,
    no_repeat: bool = True,
    no_repeat_except: None | torch.Tensor = None,
    top_k: None | int = None,
    stop_at_block_size: bool = True,
    exclude_pad: bool = True,
):

    termination_tokens = torch.tensor(
        termination_tokens, dtype=torch.int64, device=idx.device
    )

    if max_new_tokens is None:
        max_new_tokens = 128
    if no_repeat_except is None:
        no_repeat_except = torch.tensor([1])

    if isinstance(max_age, torch.Tensor):
        assert len(max_age.shape) == 1
        assert max_age.shape[0] == age.shape[0]
    else:
        max_age = torch.full((age.shape[0],), fill_value=max_age).to(idx.device)  # type: ignore
    max_age = max_age.unsqueeze(1)  # type: ignore

    batch_size = idx.shape[0]
    active_indices = torch.arange(batch_size, device=idx.device)
    completed_idx, completed_age = dict(), dict()
    cur_idx = idx.clone()
    cur_age = age.clone()

    ignore_tokens = [0]
    if (
        hasattr(model.config, "ignore_tokens")
        and model.config.ignore_tokens is not None
    ):
        ignore_tokens += model.config.ignore_tokens

    pmt_cnt = (idx > 0).sum(dim=1).detach().cpu().numpy()
    for _ in range(max_new_tokens):
        outputs, _, _ = model(cur_idx, cur_age)
        if isinstance(outputs, torch.Tensor):
            # for backwards compatibility with legacy model definition
            logits = outputs
        else:
            assert isinstance(outputs, dict)
            logits = outputs["logits"]
        logits = logits[:, -1, :]
        logits[:, ignore_tokens] = -torch.inf

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -torch.inf

        if no_repeat:
            fill = cur_idx.clone()
            fill[torch.isin(fill, no_repeat_except.to(fill.device))] = 0
            logits = logits.scatter_(1, fill, -torch.inf)

        if hasattr(model, "sample_next"):
            idx_next, time_til_next = model.sample_next(logits=logits, outputs=outputs)
        else:
            # fallback
            idx_next, time_til_next = sample_competing_exponentials(logits=logits)
        age_next = cur_age[..., [-1]] + time_til_next
        age_next[time_til_next == -1e4] = -1e4

        cur_idx = torch.cat((cur_idx, idx_next), dim=1)
        cur_age = torch.cat((cur_age, age_next), dim=1)
        sort_by_age = torch.argsort(cur_age, dim=1)
        cur_age = torch.take_along_dim(cur_age, sort_by_age, dim=1)
        cur_idx = torch.take_along_dim(cur_idx, sort_by_age, dim=1)
        margin = torch.min(torch.sum(cur_idx == 0, dim=1)).item()
        cur_idx, cur_age = cur_idx[:, margin:], cur_age[:, margin:]

        terminated = torch.isin(idx_next, termination_tokens).any(-1)
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
        should_stop = terminated | aged_out | reached_block

        if should_stop.any():
            # identify indices relative to the current active batch
            stop_indices = torch.where(should_stop)[0]
            for local_i in stop_indices:
                global_i = active_indices[local_i].item()
                completed_idx[global_i] = cur_idx[local_i]
                completed_age[global_i] = cur_age[local_i]
            # filter the running batch to keep only unfinished sequences
            cur_idx = cur_idx[~should_stop]
            cur_age = cur_age[~should_stop]
            active_indices = active_indices[~should_stop]

        if len(active_indices) == 0:
            break

    # collect stragglers (reached max_new_tokens without terminating)
    for local_i, global_i in enumerate(active_indices):
        completed_idx[global_i.item()] = cur_idx[local_i]
        completed_age[global_i.item()] = cur_age[local_i]

    max_len = max(t.numel() for t in completed_idx.values())
    final_idx = torch.full((batch_size, max_len), 0, dtype=idx.dtype, device=idx.device)
    final_age = torch.full(
        (batch_size, max_len), -1e4, dtype=age.dtype, device=age.device
    )
    for i in range(batch_size):
        idx_i, age_i = completed_idx[i], completed_age[i]
        final_idx[i, -idx_i.numel() :] = idx_i
        final_age[i, -age_i.numel() :] = age_i

    final_idx[final_age > max_age] = 1
    final_age = torch.clamp(final_age, max=max_age)

    sort_by_age = torch.argsort(final_age, dim=1)
    age = torch.take_along_dim(input=final_age, indices=sort_by_age, dim=1)
    idx = torch.take_along_dim(input=final_idx, indices=sort_by_age, dim=1)

    margin = torch.min(torch.sum(idx == 0, dim=1)).item()
    idx, age = idx[:, margin:], age[:, margin:]

    outputs, _, _ = model(idx, age)
    if isinstance(outputs, torch.Tensor):
        logits = outputs
    else:
        logits = outputs["logits"]

    if no_repeat:
        fill = idx + 0
        fill[torch.isin(fill, no_repeat_except.to(fill.device))] = 0
        logits = torch.stack(
            [
                logits[:, j].scatter_(1, fill[:, : j + 1], -torch.inf)
                for j in range(fill.shape[1])
            ]
        ).transpose(0, 1)

    gen_cnt = (idx > 0).sum(dim=1).detach().cpu().numpy()

    return idx, age, logits, {"n_prompt": pmt_cnt, "n_gen": gen_cnt}


@torch.no_grad
def shap_forward(
    idx: list[np.ndarray],
    age: list[np.ndarray],
    model: torch.nn.Module,
    doi: list[int],
):
    x_lst, t_lst = list(), list()
    for x, t in zip(idx, age):
        x_lst.append(x)
        t_lst.append(t)
    x = collate_batch(x_lst)
    t = collate_batch(t_lst)
    device = next(model.parameters()).device
    x = torch.tensor(x).to(device).long()
    t = torch.tensor(t).to(device)

    outputs, _, _ = model.forward(x, t)
    # for compatibility with legacy model definition
    if isinstance(outputs, torch.Tensor):
        logits = outputs
    else:
        logits = outputs["logits"]
    doi_logits = logits[:, -1, doi].detach().cpu().numpy()

    return doi_logits
