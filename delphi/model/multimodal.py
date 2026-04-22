import math
from dataclasses import dataclass, field
from typing import Required, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from delphi.model.transformer import (
    AgeEncoding,
    Block,
    LayerNorm,
    causal_attention_mask,
    exponential_nll,
)
from delphi.model.utils import (
    incremental_attention_mask,
    nearest_input_pos,
    sample_competing_exponentials,
    self_terminate_single,
)
from delphi.multimodal import Modality, module_name

tensor_dict = dict[str, torch.Tensor]


class BiomarkerEmbedConfig(TypedDict, total=False):
    """
    attributes:
        input_size [required]: dimensionality of the raw biomarker input features.
        projector [required]: type of projection network to use (e.g., "mlp", "linear").
        n_layers: number of layers in the projector network.
        n_hidden: hidden layer dimensionality for multi-layer projectors.
        bias: whether to include bias terms in projector layers.
    """

    input_size: Required[int]
    projector: Required[str]
    n_layers: None | int
    n_hidden: None | int
    bias: bool


class BiomarkerEmbedding(nn.Module):

    def __init__(
        self,
        n_embed: int,
        input_size: int,
        projector: str,
        n_layers: None | int = None,
        n_hidden: None | int = None,
        bias: bool = False,
    ) -> None:

        super().__init__()
        if projector == "linear":
            self.projector = nn.Linear(input_size, n_embed, bias=bias)
        elif projector == "mlp":
            layers = []
            if n_layers is None:
                n_layers = 2
            if n_hidden is None:
                n_hidden = 32
            for i in range(n_layers):
                in_size = input_size if i == 0 else n_hidden
                out_size = n_embed if i == n_layers - 1 else n_hidden
                layers.append(nn.Linear(in_size, out_size, bias=bias))
                if i < n_layers - 1:
                    layers.append(nn.ReLU())
            self.projector = nn.Sequential(*layers)
        else:
            raise ValueError(f"unknown projector type: {projector}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class DelphiEmbedding(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.n_embd, padding_idx=0
        )
        self.age_encoding = AgeEncoding(n_embd=config.n_embd)
        self.token_drop = nn.Dropout(config.token_dropout)

        self.biomarker_embed = nn.ModuleDict()
        biomarker_modalities = []
        for biomarker, biomarker_cfg in config.biomarkers.items():
            bm_key = module_name(Modality[biomarker.upper()])
            self.biomarker_embed[bm_key] = BiomarkerEmbedding(
                n_embed=config.n_embd, **biomarker_cfg
            )
            biomarker_modalities.append(Modality[biomarker.upper()])

        if config.modality_emb:
            max_modality_idx = (
                max([modality.value for modality in biomarker_modalities])
                if len(biomarker_modalities) > 0
                else 1
            )
            self.mod_embedding = nn.Embedding(
                max_modality_idx + 1, config.n_embd, padding_idx=0
            )

    def forward(
        self,
        idx: torch.Tensor,
        age: torch.Tensor,
        mod_idx: None | torch.Tensor = None,
        mod_age: None | torch.Tensor = None,
        biomarker_x: None | dict[Modality, torch.Tensor] = None,
        emb: None | torch.Tensor = None,
    ):

        raw = dict()
        if emb is None:
            idx_emb = self.token_embedding(idx)
            idx_emb = self.token_drop(idx_emb) * (1 - self.config.token_dropout)
            age_emb = self.age_encoding(age.unsqueeze(-1))
            emb = idx_emb + age_emb
            raw["idx"] = idx_emb
            raw["age"] = age_emb

        if biomarker_x is None:
            return emb, None, None

        biomarker_emb = dict()
        mod_age_emb = self.age_encoding(mod_age.unsqueeze(-1))
        for modality in biomarker_x.keys():
            biomarker_emb[modality] = self.biomarker_embed[module_name(modality)](
                biomarker_x[modality]
            )  # N * H
            mod_mask = mod_idx == modality.value
            biomarker_emb[modality] += mod_age_emb[mod_mask]
            if self.config.modality_emb:
                mod_emb = self.mod_embedding(
                    torch.tensor(modality.value).to(idx.device)
                )
                biomarker_emb[modality] += mod_emb.unsqueeze(0)

        raw = {
            "idx": idx_emb,
            "age": age_emb,
            "mod_age": mod_age_emb,
            "biomarker": biomarker_emb,
        }
        return emb, biomarker_emb, raw


def fuse_embed(
    mod_idx: torch.Tensor,
    mod_age: torch.Tensor,
    mod_emb: dict[Modality, torch.Tensor],
    emb: torch.Tensor,
    age: torch.Tensor,
    idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    fuse modality embeddings and base embeddings, sorted by age.

    construct the full unsorted tensors first (concatenating modality and base data),
    then apply the time-sort index to the whole block at once.

    disease token positions get mod_idx=1 only when idx > 0 (non-padding),
    so that fused_mod_idx > 0 can be used as a padding mask. fused_idx carries
    the disease token ids (0 at biomarker and padding positions).
    """
    _, _, n_embd = emb.shape
    device = emb.device
    mod_emb_dense = torch.zeros(
        (*mod_idx.shape, n_embd), dtype=emb.dtype, device=device
    )
    for modality, m_tensor in mod_emb.items():
        mask = mod_idx == modality.value
        if m_tensor.shape[0] != mask.sum():
            raise ValueError(
                f"Shape mismatch for {modality}: mask expects {mask.sum()} tokens, "
                f"got {m_tensor.shape[0]}"
            )
        mod_emb_dense[mask] = m_tensor

    fused_emb_unsorted = torch.cat((mod_emb_dense, emb), dim=1)
    disease_mod_idx = (idx > 0).to(mod_idx.dtype)
    fused_mod_idx_unsorted = torch.cat((mod_idx, disease_mod_idx), dim=1)
    fused_idx_unsorted = torch.cat(
        (torch.zeros_like(mod_idx, dtype=idx.dtype), idx), dim=1
    )
    fused_age_unsorted = torch.cat((mod_age, age), dim=1)

    # stable=True ensures biomarkers (mod_emb) precede disease tokens (emb) when ages are equal
    sort_indices = torch.argsort(fused_age_unsorted, stable=True, dim=1)
    fused_emb = torch.take_along_dim(
        fused_emb_unsorted, sort_indices.unsqueeze(-1), dim=1
    )
    fused_age = torch.take_along_dim(fused_age_unsorted, sort_indices, dim=1)
    fused_mod_idx = torch.take_along_dim(fused_mod_idx_unsorted, sort_indices, dim=1)
    fused_idx = torch.take_along_dim(fused_idx_unsorted, sort_indices, dim=1)

    return fused_emb, fused_age, fused_mod_idx, fused_idx


@dataclass
class DelphiM4Config:
    """
    attributes:
        block_size: maximum sequence length for the model, None for unlimited.
        vocab_size: size of the vocabulary.
        n_layer: # transformer layers.
        n_head: # attention heads.
        n_embd: dimensionality of the embeddings and hidden states.
        dropout: dropout probability for regularization.
        token_dropout: dropout probability specifically for token embeddings.
        t_min: epsilon for time to next event.
        bias: whether to include bias terms in linear layers.
        mask_ties: how to handle targets occurring at the same timestep.
            False: allow next-token predictions with 0 time-to-next-event values.
            True: compute loss separately for each target where multiple targets occur together at the next timestep.
        attn_mask: type of attention masking.
            "time" [default]: causal masking based on timestep.
            "triangular": classic lower triangular, causal masking.
        weight_tying: whether to tie input and output embedding weights.
        ignore_tokens: list of tokens to ignore as targets.
            default [0, 2-12] includes zero padding tokens and gender and lifestyle tokens in the UKB.
        biomarkers: mapping biomarker names to their embedding configs.
        modality_emb: whether to introduce modality-specific embeddings.
        ce_beta: weight coefficient for cross-entropy loss.
        dt_beta: weight coefficient for delta-time loss.
        fuse: strategy for fusing multimodal information (only "early" is currently supported).
    """

    block_size: None | int = 256
    vocab_size: int = 1270
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 120
    dropout: float = 0.1
    token_dropout: float = 0.0
    t_min: float = 0.1
    bias: bool = True
    mask_ties: bool = True
    attn_mask: str = "time"
    weight_tying: bool = False
    ignore_tokens: list = field(
        default_factory=lambda: [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    )
    biomarkers: dict[str, BiomarkerEmbedConfig] = field(default_factory=dict)
    modality_emb: bool = True
    self_terminate_except: list = field(default_factory=lambda: [1])
    ce_beta: float = 1.0
    dt_beta: float = 1.0
    fuse: str = "early"  # early, cross, concat, concat-raw


class DelphiM4(torch.nn.Module):
    model_type = "delphi-m4"

    def __init__(self, config: DelphiM4Config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                embed=DelphiEmbedding(config),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.weight_tying:
            self.transformer.embed.token_embedding.weight = self.lm_head.weight

        self.fuse_early = self.config.fuse == "early"
        if not self.fuse_early:
            raise NotImplementedError

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

        age = outputs["age"]
        logits = outputs["logits"]
        # clamp the -1 sentinel (no earlier input) to 0; training targets are
        # expected to come after at least one input token
        pos = nearest_input_pos(age, targets_age).clamp(min=0)
        logits = torch.take_along_dim(logits, dim=1, indices=pos.unsqueeze(-1))
        age = torch.take_along_dim(age, dim=1, indices=pos)

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

        is_valid_target = targets != 0
        for k in self.config.ignore_tokens:
            is_valid_target *= targets != k
        loss_ce[~is_valid_target] = torch.nan
        loss_dt[~is_valid_target] = torch.nan

        if reduce:
            loss_ce = torch.nanmean(loss_ce)
            loss_dt = torch.nanmean(loss_dt)

        loss = {
            "loss_ce": loss_ce,
            "loss_dt": loss_dt,
        }

        return loss

    @torch.no_grad()
    def sample_next(self, outputs: dict[str, torch.Tensor], idx: torch.Tensor):
        logits = outputs["logits"][:, -1, :]
        logits = self_terminate_single(
            idx=idx,
            logits=logits,
            terminate_except=torch.tensor(self.config.self_terminate_except).to(
                idx.device
            ),
        )
        idx_next, time_til_next = sample_competing_exponentials(logits=logits)
        return idx_next, time_til_next

    def forward(
        self,
        idx: torch.Tensor,
        age: torch.Tensor,
        biomarker: None | dict[Modality, torch.Tensor] = None,
        mod_age: None | torch.Tensor = None,
        mod_idx: None | torch.Tensor = None,
        targets: None | torch.Tensor = None,
        targets_age: None | torch.Tensor = None,
        past_kvs: None | list = None,
        past_pad: None | torch.Tensor = None,
        emb: None | torch.Tensor = None,
    ):

        x, mod_emb, _ = self.transformer.embed(
            idx=idx,
            age=age,
            mod_idx=mod_idx,
            mod_age=mod_age,
            biomarker_x=biomarker,
            emb=emb,
        )

        if mod_emb is not None:
            x, fused_age, fused_mod_idx, fused_idx = fuse_embed(
                emb=x,
                age=age,
                mod_idx=mod_idx,
                mod_age=mod_age,
                mod_emb=mod_emb,
                idx=idx,
            )
            pad = fused_mod_idx > 0
        else:
            fused_age = age
            fused_idx = idx
            pad = (idx > 0).to(idx.dtype)

        if past_kvs is not None:
            assert past_pad is not None
            attn_mask = incremental_attention_mask(new_pad=pad, past_pad=past_pad)
        elif self.config.attn_mask == "triangular":
            attn_mask = causal_attention_mask(pad=pad)
        else:
            attn_mask = causal_attention_mask(pad=pad, timestep=fused_age)

        x = self.transformer.drop(x)
        att = []
        new_kvs = []
        for i, block in enumerate(self.transformer.h):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, a, new_kv = block(x, attn_mask, past_kv=past_kv)
            att.append(a)
            new_kvs.append(new_kv)
        x = self.transformer.ln_f(x)
        att = torch.stack(att)

        misc = dict()
        misc["attn_mask"] = attn_mask
        misc["attn"] = att
        misc["past_kvs"] = new_kvs
        misc["cur_pad"] = pad

        outputs = dict()
        outputs["age"] = fused_age
        outputs["idx"] = fused_idx
        outputs["logits"] = self.lm_head(x)

        if targets is not None and targets_age is not None:
            loss = self.loss(outputs, targets=targets, targets_age=targets_age)
        else:
            loss = None

        return outputs, loss, misc
