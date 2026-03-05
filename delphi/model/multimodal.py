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
from delphi.model.utils import untie_idx
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
        mod_idx: torch.Tensor,
        mod_age: torch.Tensor,
        biomarker_x: dict[Modality, torch.Tensor],
    ):

        idx_emb = self.token_embedding(idx)
        idx_emb = self.token_drop(idx_emb) * (1 - self.config.token_dropout)
        age_emb = self.age_encoding(age.unsqueeze(-1))
        emb = idx_emb + age_emb

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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    fuse modality embeddings and base embeddings, sorted by age.

    construct the full unsorted tensors first (concatenating modality and base data),
    then apply the time-sort index to the whole block at once.
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
    fused_idx_unsorted = torch.cat(
        (mod_idx, torch.ones_like(age, dtype=mod_idx.dtype)), dim=1
    )
    fused_age_unsorted = torch.cat((mod_age, age), dim=1)

    # stable=True ensures biomarkers (mod_emb) precede disease tokens (emb) when ages are equal
    sort_indices = torch.argsort(fused_age_unsorted, stable=True, dim=1)
    fused_emb = torch.take_along_dim(
        fused_emb_unsorted, sort_indices.unsqueeze(-1), dim=1
    )
    fused_age = torch.take_along_dim(fused_age_unsorted, sort_indices, dim=1)
    fused_mod_idx = torch.take_along_dim(fused_idx_unsorted, sort_indices, dim=1)

    return fused_emb, fused_age, fused_mod_idx


def fuse_targets_mask(targets_age: torch.Tensor, mod_age: torch.Tensor):
    fused_age = torch.cat((mod_age, targets_age), dim=1)
    time_sort = torch.argsort(fused_age, stable=True, dim=1)
    is_target = torch.cat(
        (torch.zeros_like(mod_age), torch.ones_like(targets_age)), dim=1
    )
    is_target = torch.take_along_dim(is_target, time_sort, dim=1)
    return is_target


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
        ablate_biomarker: settings for biomarker ablation experiments
            None [default]: no ablation setting applied.
            "biomarker": only attend to biomarker(s).
            "token": only attend to tokens before the first biomarker.
            "both": attend both biomarker(s) and the tokens before the first biomarker.
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
    weight_tying: bool = True
    ignore_tokens: list = field(
        default_factory=lambda: [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    )
    biomarkers: dict[str, BiomarkerEmbedConfig] = field(default_factory=dict)
    modality_emb: bool = True
    ablate_biomarker: None | str = None  # biomarker, token, both
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
        logits: torch.Tensor,
        targets: torch.Tensor,
        age: torch.Tensor,
        targets_age: torch.Tensor,
    ):
        if self.config.mask_ties:
            corr_idx = untie_idx(age, targets_age)
            age = torch.take_along_dim(input=age, indices=corr_idx, dim=1)
            logits = torch.take_along_dim(
                input=logits, indices=corr_idx.unsqueeze(-1), dim=1
            )

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

        return loss_ce, loss_dt

    def forward(
        self,
        idx: torch.Tensor,
        age: torch.Tensor,
        biomarker: dict[Modality, torch.Tensor],
        mod_age: torch.Tensor,
        mod_idx: torch.Tensor,
        targets: None | torch.Tensor = None,
        targets_age: None | torch.Tensor = None,
    ):

        if self.config.ablate_biomarker is not None:
            if mod_age.numel() > 0:
                _mod_age = mod_age.clone()
                _mod_age[_mod_age == -1e4] = torch.inf
                min_mod_age = _mod_age.min(dim=1, keepdim=True)[0]
            else:
                min_mod_age = torch.full(
                    (mod_age.shape[0], 1), fill_value=torch.inf
                ).to(age.device)
            if self.config.ablate_biomarker == "biomarker":
                idx *= 0
            elif self.config.ablate_biomarker in {"token", "both"}:
                idx[age > min_mod_age] = 0
            else:
                raise NotImplementedError

        x, mod_emb, _ = self.transformer.embed(
            idx=idx, age=age, mod_idx=mod_idx, mod_age=mod_age, biomarker_x=biomarker
        )
        x, fused_age, fused_mod_idx = fuse_embed(
            emb=x, age=age, mod_idx=mod_idx, mod_age=mod_age, mod_emb=mod_emb
        )
        pad = fused_age != -1e4
        if self.config.ablate_biomarker is not None:
            if self.config.ablate_biomarker == "biomarker":
                pad *= fused_mod_idx != 1
            elif self.config.ablate_biomarker == "token":
                pad *= torch.logical_and(fused_age <= min_mod_age, fused_mod_idx == 1)  # type: ignore
            elif self.config.ablate_biomarker == "both":
                pad *= fused_age <= min_mod_age  # type: ignore

        if self.config.attn_mask == "triangular":
            attn_mask = causal_attention_mask(pad=pad)
        else:
            attn_mask = causal_attention_mask(pad=pad, timestep=fused_age)

        x = self.transformer.drop(x)
        att = []
        for block in self.transformer.h:
            x, a, _ = block(x, attn_mask)
            att.append(a)
        x = self.transformer.ln_f(x)
        att = torch.stack(att)

        misc = dict()
        misc["attn_mask"] = attn_mask
        misc["attn"] = att

        outputs = dict()
        logits = self.lm_head(x)

        if (targets is not None) and (targets_age is not None):
            is_target = fuse_targets_mask(
                targets_age=targets_age, mod_age=mod_age
            ).bool()
            logits = logits[is_target].view(*idx.shape, -1)
            age = fused_age[is_target].view(*idx.shape)
            outputs["age"] = age

            is_valid_target = targets != 0
            for k in self.config.ignore_tokens:
                is_valid_target *= targets != k
            if self.config.ablate_biomarker is not None:
                is_valid_target *= age > min_mod_age  # type: ignore
            loss_ce, loss_dt = self.loss(
                logits=logits,
                targets=targets,
                age=age,
                targets_age=targets_age,
            )
            loss_ce = torch.mean(loss_ce[is_valid_target])
            loss_dt = torch.mean(loss_dt[is_valid_target])
            loss = {
                "loss_ce": loss_ce * self.config.ce_beta,
                "loss_dt": loss_dt * self.config.dt_beta,
            }
        else:
            loss = None

        outputs["logits"] = logits

        return outputs, loss, misc
