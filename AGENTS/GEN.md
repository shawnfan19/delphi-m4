# GEN.md — `generate()` Design Guide

`generate()` is a standalone function in `delphi/model/transformer.py` that autoregressively simulates patient disease trajectories from a trained Delphi model. This document explains the design decisions and non-obvious invariants an agent needs to understand before modifying it.

---

## What It Does

Given a batch of patients with prompt sequences (observed history), `generate()` extends each patient's trajectory by sampling new events — event type and time-to-event — one step at a time until a stopping condition is met. It returns the completed trajectories and generation statistics.

---

## Signature

```python
@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,              # (B, L) prompt token sequences
    age: torch.Tensor,              # (B, L) prompt timestamps in days
    termination_tokens: list | torch.Tensor,  # tokens that end generation (e.g. death)
    max_new_tokens: None | int = 100,
    max_age: float | torch.Tensor = 85 * 365.25,  # per-patient age ceiling
    stop_at_block_size: bool = True,
    exclude_pad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
```

**Returns**: `(idx, age, metadata)` where `metadata` contains `n_prompt` and `n_gen` per patient.

> **No logits returned**: unlike previous versions, `generate()` no longer runs a final forward pass and does not return logits. If downstream analysis needs logits over the full generated trajectory, call `model.forward(idx, age)` explicitly after `generate()` returns, then apply `self_terminate` from `delphi.model.utils` to mask already-seen tokens.

---

## Stopping Conditions

Generation halts for a sequence when **any** of three conditions is met:

| Condition | Parameter | Notes |
|-----------|-----------|-------|
| Termination token generated | `termination_tokens` | E.g. death token; use `.any(-1)` to handle multi-token samplers |
| Patient age exceeds ceiling | `max_age` | Scalar or per-patient `(B,)` tensor — allows conditioning on individual follow-up windows |
| Sequence fills model's context | `stop_at_block_size=True` | Compares non-padding token count against `config.block_size`; `exclude_pad=False` counts total length instead |

A fourth implicit ceiling: `max_new_tokens` — a hard upper bound on loop iterations to prevent infinite loops when none of the above trigger.

Stragglers (sequences still active when `max_new_tokens` is exhausted) are collected at the end rather than discarded.

---

## Active Batch Management

The generation loop maintains a *shrinking active batch* rather than keeping terminated sequences live. This is the most important structural feature:

```python
active_indices = torch.arange(batch_size)   # global indices of still-running sequences
completed_idx, completed_age = dict(), dict()  # keyed by global index

# When sequences stop:
completed_idx[global_i] = cur_idx[local_i]   # save to dict
cur_idx = cur_idx[~should_stop]               # drop from active batch
active_indices = active_indices[~should_stop] # update index mapping
```

`active_indices` maps local positions in the current batch back to global patient indices. This is critical for `max_age` indexing: `max_age[active_indices]` selects per-patient ceilings for the current active slice.

**Why this matters**: In EHR simulation, sequences terminate at wildly different times (some patients die young, some reach study end late). Without active batch management, the full batch must run for as long as the slowest sequence, wasting compute on already-terminated patients.

---

## Temporal Sorting Invariant

After each generated token, the sequence is **re-sorted by age**:

```python
cur_idx = torch.cat((cur_idx, idx_next), dim=1)
cur_age = torch.cat((cur_age, age_next), dim=1)
sort_by_age = torch.argsort(cur_age, dim=1)
cur_age = torch.take_along_dim(cur_age, sort_by_age, dim=1)
cur_idx = torch.take_along_dim(cur_idx, sort_by_age, dim=1)
```

This preserves the invariant that `cur_idx` and `cur_age` are always in non-decreasing temporal order. This matters because the transformer uses a **time-based causal attention mask** (`attn_mask="time"` mode): position $j$ can attend to position $i$ iff `age[i] <= age[j]`. Inserting tokens out of order without re-sorting would silently break the attention mask semantics.

In practice, newly generated tokens are almost always the most recent (appended to the end of time), so re-sorting is usually a no-op. The sort handles edge cases where `homo_cluster_poisson` generates multiple simultaneous events, or where numerical timing puts events at the same age as the previous one.

---

## Leading Padding Trim

After sorting, leading padding is trimmed to the minimum across the active batch:

```python
margin = torch.min(torch.sum(cur_idx == 0, dim=1)).item()
cur_idx, cur_age = cur_idx[:, margin:], cur_age[:, margin:]
```

The `min` is conservative: it trims only as many leading zeros as every sequence in the current batch has. This prevents memory from growing indefinitely with each generated token and keeps sequence length roughly stable. The invariant is that no real event is ever trimmed — only confirmed padding positions.

---

## Self-Termination (No-Repeat)

Self-termination — preventing the model from re-generating already-seen event types — is now handled **inside `model.sample_next()`**, not in `generate()`. The `generate()` loop calls:

```python
idx_next, time_til_next = model.sample_next(outputs=outputs, idx=cur_idx)
```

`sample_next` applies `self_terminate_single(idx=cur_idx, logits=..., terminate_except=...)` internally, using `model.config.self_terminate_except` (default: `[1]`, exempting the no-event token) to decide which tokens may repeat.

**Consequence**: `generate()` has no `no_repeat`, `no_repeat_except`, or `top_k` parameters. Logit masking behavior is fully determined by the model config and `sample_next` implementation.

**Post-generation logits**: After calling `generate()`, if you need causally-masked logits over the full trajectory (e.g. for risk scoring), run:
```python
output, _, _ = model.forward(idx, age)
logits = output["logits"]
logits = self_terminate(
    idx,
    logits,
    terminate_except=torch.tensor(model.config.self_terminate_except).to(idx.device),
)
```

`self_terminate` (from `delphi.model.utils`) applies the cumulative mask: at position $j$, tokens seen in positions $0..j$ are set to $-\infty$.

---

## Final Assembly (No Forward Pass)

After all sequences complete, they are re-padded into a uniform batch. Events generated past `max_age` are clipped:

```python
# Re-pad to uniform length (left-padded, right-aligned — consistent with codebase-wide convention)
final_idx[i, -idx_i.numel():] = idx_i

# Clip events generated past max_age
final_idx[final_age > max_age] = 1      # replace with no-event token
final_age = torch.clamp(final_age, max=max_age)
```

**No final forward pass is performed.** `generate()` returns `(idx, age, metadata)` only. This means callers who need logits must run a forward pass themselves — see the post-generation logits example above.

---

## Multi-Token Sampling (`homo_cluster_poisson`)

`sample_next` can return `idx_next` with shape `(B, N)` where `N > 1` — the cluster loss generates multiple simultaneous events in one step. `generate()` handles this transparently:

- `torch.cat((cur_idx, idx_next), dim=1)` works for any N
- `terminated = torch.isin(idx_next, termination_tokens).any(-1)` — `.any(-1)` catches a termination token in any of the N slots
- `aged_out = (age_next > max_age[active_indices]).any(-1)` — same

The sentinel value `time_til_next == -1e4` marks "dummy" slots in multi-token outputs (padding within the cluster), and `age_next[time_til_next == -1e4] = -1e4` ensures these sort to the front and get trimmed as padding.

---

## What `generate()` Is Not Responsible For

- **Beam search / best-of-N**: `generate()` is purely stochastic — it samples from the model distribution. Ranking or filtering multiple trajectories must be done at the call site.
- **Input conditioning beyond prompts**: Any patient covariates (sex, genotype, etc.) must be encoded into the prompt sequence before calling `generate()`.
- **Logit masking / no-repeat**: Fully delegated to `model.sample_next()`. `generate()` passes `idx` (the current full sequence) to `sample_next` for this purpose.
- **Parametric losses**: `generate()` calls `model.sample_next()`, which currently raises `NotImplementedError` for `hawkes` and `hawkes_weibull`. `weibull` sampling is now implemented via `sample_tpp` / `thinning_sample` in `delphi/model/utils.py`. Implementing sampling for the remaining losses requires extending `sample_next()`, not `generate()` itself.
- **Returning logits**: Callers that need logits over the generated trajectory must run a separate `model.forward()` call after `generate()` returns.
