"""Equivalence of the SDPA fast path and the manual attention path.

`CausalSelfAttention.forward` runs `F.scaled_dot_product_attention` by default
(`return_attn=False`) and only falls back to the explicit
`softmax(QKᵀ·scale)·V` implementation when a caller needs the attention weights
(`return_attn=True`). Both branches must agree up to floating-point reordering.

We drive the real `CausalSelfAttention` module directly (one small nn.Module — no
full model) with random inputs and the real mask builders, comparing its own two
branches. This guards the actual dispatch, scale, and 0/1→bool mask translation,
rather than a reimplementation of the formula that could silently drift from
production.
"""

import torch

from delphi.model.multimodal import DelphiM4Config
from delphi.model.transformer import CausalSelfAttention
from delphi.model.utils import causal_attention_mask, incremental_attention_mask

ATOL, RTOL = 1e-5, 1e-4
N_HEAD, N_EMBD = 4, 32


def _attention():
    torch.manual_seed(0)
    cfg = DelphiM4Config(n_head=N_HEAD, n_embd=N_EMBD, dropout=0.0)
    return CausalSelfAttention(cfg).eval()


def test_sdpa_matches_manual_full_sequence():
    attn = _attention()
    B, T = 3, 6
    # Row 1 is left-padded (idx 0 / age -1e4): its pad-query rows keep only the
    # forced diagonal — the all-but-diagonal-masked case that risks a softmax NaN.
    pad = torch.tensor([[True] * 6, [False, False, True, True, True, True], [True] * 6])
    age = torch.tensor(
        [
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            [-1e4, -1e4, 15.0, 25.0, 35.0, 45.0],
            [5.0, 12.0, 18.0, 22.0, 33.0, 41.0],
        ]
    )
    mask = causal_attention_mask(pad=pad, timestep=age)  # (B,1,T,T) int, diag=1

    torch.manual_seed(1)
    x = torch.randn(B, T, N_EMBD)
    with torch.no_grad():
        y_manual, att, _ = attn(x, mask, return_attn=True)
        y_sdpa, att_none, _ = attn(x, mask, return_attn=False)

    assert att is not None and att.shape == (B, N_HEAD, T, T)
    assert att_none is None
    assert torch.isfinite(y_manual).all() and torch.isfinite(y_sdpa).all()
    torch.testing.assert_close(y_manual, y_sdpa, atol=ATOL, rtol=RTOL)


def test_sdpa_matches_manual_incremental():
    """Rectangular KV-cache mask + past_kv concat (the generation path)."""
    attn = _attention()
    B, T_cached, hs = 3, 4, N_EMBD // N_HEAD

    torch.manual_seed(2)
    x = torch.randn(B, 1, N_EMBD)  # one new query token
    past_kv = (
        torch.randn(B, N_HEAD, T_cached, hs),
        torch.randn(B, N_HEAD, T_cached, hs),
    )
    new_pad = torch.ones(B, 1, dtype=torch.bool)
    past_pad = torch.tensor([[True] * 4, [False, False, True, True], [True] * 4])
    # (B, 1, 1, T_cached + 1)
    mask = incremental_attention_mask(new_pad=new_pad, past_pad=past_pad)

    with torch.no_grad():
        y_manual, att, _ = attn(x, mask, past_kv=past_kv, return_attn=True)
        y_sdpa, att_none, _ = attn(x, mask, past_kv=past_kv, return_attn=False)

    assert att is not None and att_none is None
    assert torch.isfinite(y_manual).all() and torch.isfinite(y_sdpa).all()
    torch.testing.assert_close(y_manual, y_sdpa, atol=ATOL, rtol=RTOL)
