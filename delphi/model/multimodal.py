import copy
import math
from dataclasses import dataclass, field
from typing import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from typing_extensions import Required

from delphi.model.tpp import (
    NeuralIntensity,
    NeuralODEIntensity,
    tpp_dispatch,
)
from delphi.model.transformer import (
    AgeEncoding,
    Block,
    causal_attention_mask,
)
from delphi.model.utils import (
    incremental_attention_mask,
    sample_competing_exponentials,
    self_terminate_single,
)

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


class BiomarkerEmbeddingDict(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.ModuleDict()
        for biomarker, biomarker_cfg in config.biomarkers.items():
            self.embed[biomarker.lower()] = BiomarkerEmbedding(
                n_embed=config.n_embd, **biomarker_cfg
            )

    def forward(
        self,
        biomarker_x: dict[str, torch.Tensor],
    ):
        biomarker_emb = dict()
        for biomarker in biomarker_x.keys():
            biomarker_emb[biomarker] = self.embed[biomarker.lower()](
                biomarker_x[biomarker]
            )  # N * H

        return biomarker_emb


def fuse_embed(
    mod_emb: dict[str, torch.Tensor],
    mod_idx: torch.Tensor,
    mod_age: torch.Tensor,
    mod_idx_emb: None | torch.Tensor,
    mod_age_emb: torch.Tensor,
    emb: torch.Tensor,
    age: torch.Tensor,
    idx: torch.Tensor,
    biomarker2idx: dict[str, int],
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
    for biomarker, m_tensor in mod_emb.items():
        mask = mod_idx == biomarker2idx[biomarker]
        if m_tensor.shape[0] != mask.sum():
            raise ValueError(
                f"Shape mismatch for {biomarker}: mask expects {mask.sum()} tokens, "
                f"got {m_tensor.shape[0]}"
            )
        # out-of-place (vs mod_emb_dense[mask] = m_tensor) so this composes with an
        # outer vmap over a batched m_tensor — e.g. integrated_jacobian batching its
        # interpolation steps. Numerically identical to the in-place scatter.
        mod_emb_dense = torch.index_put(mod_emb_dense, (mask,), m_tensor)
    mod_emb_dense += mod_age_emb
    if mod_idx_emb is not None:
        mod_emb_dense += mod_idx_emb

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


class BiomarkerDecoder(nn.Module):

    def __init__(
        self,
        n_embed: int,
        projector: str,
        n_layers: None | int = None,
        n_hidden: None | int = None,
        bias: bool = False,
    ):

        super().__init__()
        if projector == "linear":
            self.projector = nn.Linear(n_embed, n_embed, bias=bias)
        elif projector == "mlp":
            layers = []
            if n_layers is None:
                n_layers = 2
            if n_hidden is None:
                n_hidden = 32
            for i in range(n_layers):
                in_size = n_embed if i == 0 else n_hidden
                out_size = n_embed if i == n_layers - 1 else n_hidden
                layers.append(nn.Linear(in_size, out_size, bias=bias))
                if i < n_layers - 1:
                    layers.append(nn.ReLU())
            self.projector = nn.Sequential(*layers)
        else:
            raise ValueError(f"unknown projector type: {projector}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class DelphiDecoder(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.biomarker_decoder = nn.ModuleDict()
        for biomarker, _ in config.biomarkers.items():
            self.biomarker_decoder[biomarker.lower()] = BiomarkerDecoder(
                n_embed=config.n_embd, projector="mlp", bias=True
            )
        self.idx2biomarker = {v: k for k, v in config.biomarker2idx.items()}

    def forward(self, embeddings, mod_idx):
        output = dict()
        for idx in torch.unique(mod_idx[mod_idx > 1]):
            idx = idx.item()
            biomarker = self.idx2biomarker[idx]
            output[biomarker] = self.biomarker_decoder[biomarker.lower()](
                embeddings[mod_idx == idx]
            )  # N * H
        return output


def multitask_loss(
    bio_emb: dict[str, torch.Tensor],
    bio_emb_hat: dict[str, torch.Tensor],
    mse_beta: float,
):
    bio_loss = dict()
    for biomarker in bio_emb.keys():
        mse = F.mse_loss(
            input=bio_emb_hat[biomarker], target=bio_emb[biomarker], reduction="none"
        )
        mse = torch.mean(mse, dim=-1)
        cos = 1 - F.cosine_similarity(bio_emb_hat[biomarker], bio_emb[biomarker])
        bio_loss[biomarker] = mse_beta * mse + (1 - mse_beta) * cos

    return bio_loss


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
        biomarker2idx: mapping biomarker names to the integer used in mod_idx
            (i.e. the modality channel). Saved with the checkpoint so the same
            mapping can be reconstructed at inference. 0 and 1 are reserved
            for padding and event tokens, so values start at 2.
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
    biomarker2idx: dict[str, int] = field(default_factory=dict)
    modality_emb: bool = True
    self_terminate_except: list = field(default_factory=lambda: [1])
    # valid model targets that are NOT meaningful diseases (synthetic/inserted
    # tokens), so they are scored in the loss / generatable but excluded from
    # disease evaluation. Default [1]=no_event; tiebreak training appends the dx
    # anchor. Distinct axis from ignore_tokens (loss-excluded) and
    # self_terminate_except (recurrence). See the DelphiM4.augmentation_tokens
    # property for the call-site exclusion idiom.
    augmentation_tokens: list = field(default_factory=lambda: [1])
    loss: str = "homo_poisson"
    time_unit: float = 365.25
    multitask: bool = False
    ema: None | float = 0.999
    n_integrate_grid: int = 20
    integrate_method: str = "trapezoid"
    ode_method: str = "rk4"
    ode_step_size: float = 0.25
    ce_beta: float = 1.0
    dt_beta: float = 1.0
    multitask_beta: float = 0.1
    mse_beta: float = 1.0
    spectral_norm: bool = False
    fuse: str = "early"  # early, cross, concat, concat-raw


class DelphiM4(torch.nn.Module):
    model_type = "delphi-m4"

    def __init__(self, config: DelphiM4Config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wae=AgeEncoding(n_embd=config.n_embd),
                token_drop=nn.Dropout(config.token_dropout),
                # embed=DelphiEmbedding(config),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        if config.loss == "neural_tpp":
            self.neural_tpp_head = NeuralIntensity(
                n_embd=config.n_embd,
                vocab_size=config.vocab_size,
                time_encoder=AgeEncoding(n_embd=config.n_embd),
                spectral_norm=config.spectral_norm,
            )
        elif config.loss == "homo_poisson":
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            if config.weight_tying:
                self.transformer.wte.weight = self.lm_head.weight
        elif config.loss == "neural_ode":
            self.neural_head = NeuralODEIntensity(
                n_embd=config.n_embd, vocab_size=config.vocab_size
            )
        else:
            raise ValueError

        if len(config.biomarkers) > 0:
            self.bio_embed = BiomarkerEmbeddingDict(config)
            if config.modality_emb:
                max_modality_idx = max(config.biomarker2idx.values())
                self.mod_embedding = nn.Embedding(
                    max_modality_idx + 1, config.n_embd, padding_idx=0
                )
            if config.multitask:
                self.decoder = DelphiDecoder(config)
                if config.ema is not None:
                    self.encoder = AveragedModel(
                        self.bio_embed,
                        multi_avg_fn=get_ema_multi_avg_fn(config.ema),
                        use_buffers=True,
                    )
                    for param in self.encoder.parameters():
                        param.requires_grad = False
                else:
                    self.encoder = self.bio_embed

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

    def update_ema(self):
        if isinstance(getattr(self, "encoder", None), AveragedModel):
            self.encoder.update_parameters(self.bio_embed)

    @property
    def targets(self):
        all = torch.arange(self.config.vocab_size)
        targets = all[~torch.isin(all, torch.tensor(self.config.ignore_tokens))]
        return targets

    @property
    def augmentation_tokens(self):
        """Model targets that are NOT meaningful diseases (no_event / dx anchor),
        as a tensor. Eval scripts exclude these from ``targets`` to get the disease
        set, e.g. ``targets[~torch.isin(targets, model.augmentation_tokens)]`` —
        kept explicit at the call site rather than hidden behind a property."""
        return torch.tensor(self.config.augmentation_tokens or [])

    def loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
        targets_age: torch.Tensor,
        reduce: bool = True,
    ):
        tpp = tpp_dispatch(self, outputs)

        log_p = tpp.log_likelihood(x1=targets, t1=targets_age)
        nll = -log_p

        is_valid = targets != 0
        if self.config.ignore_tokens is not None:
            for k in self.config.ignore_tokens:
                is_valid &= targets != k
            nll = nll.masked_fill(~is_valid, torch.nan)
        if reduce:
            nll = torch.nanmean(nll)
        loss = {"loss_nll": nll}

        if self.config.multitask:
            latent_states = tpp.latent_states(outputs["bio_age"])
            bio_emb_hat = self.decoder(latent_states, outputs["bio_idx"])
            with torch.no_grad():
                self.encoder.eval()
                bio_emb = self.encoder(outputs["bio_x"])
            bio_loss = multitask_loss(
                bio_emb_hat=bio_emb_hat,
                bio_emb=bio_emb,
                mse_beta=self.config.mse_beta,
            )
            if reduce:
                for key in bio_loss.keys():
                    loss[f"loss_{key}"] = (
                        torch.mean(bio_loss[key])
                        * self.config.multitask_beta
                        / len(self.config.biomarkers)
                    )

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
        idx_next, time_til_next = sample_competing_exponentials(
            logits=logits, time_unit=self.config.time_unit
        )
        return idx_next, time_til_next

    def forward(
        self,
        idx: torch.Tensor,
        age: torch.Tensor,
        biomarker: None | dict[str, torch.Tensor] = None,
        mod_age: None | torch.Tensor = None,
        mod_idx: None | torch.Tensor = None,
        targets: None | torch.Tensor = None,
        targets_age: None | torch.Tensor = None,
        past_kvs: None | list = None,
        past_pad: None | torch.Tensor = None,
        return_attn: bool = False,
        # emb: None | torch.Tensor = None,
    ):

        tok_emb = self.transformer.wte(idx)
        age_emb = self.transformer.wae(age.unsqueeze(-1))
        x = self.transformer.token_drop(tok_emb) * (1 - self.config.token_dropout)
        x = x + age_emb

        if biomarker:
            mod_emb = self.bio_embed(biomarker)
            mod_age_emb = self.transformer.wae(mod_age.unsqueeze(-1))
            mod_idx_emb = self.mod_embedding(mod_idx)

            x, fused_age, fused_mod_idx, fused_idx = fuse_embed(
                emb=x,
                age=age,
                mod_idx=mod_idx,
                mod_idx_emb=mod_idx_emb,
                mod_age=mod_age,
                mod_age_emb=mod_age_emb,
                mod_emb=mod_emb,
                idx=idx,
                biomarker2idx=self.config.biomarker2idx,
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
        outputs["age"] = fused_age
        outputs["idx"] = fused_idx
        if biomarker:
            outputs["bio_age"] = mod_age
            outputs["bio_idx"] = mod_idx
            outputs["bio_x"] = biomarker
        outputs["h"] = x

        if hasattr(self, "lm_head"):
            outputs["logits"] = self.lm_head(x)

        if targets is not None and targets_age is not None:
            loss = self.loss(outputs, targets=targets, targets_age=targets_age)
        else:
            loss = None

        return outputs, loss, misc
