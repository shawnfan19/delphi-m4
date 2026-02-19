# DESIGN.md

## Project Overview

This project develops a **Temporal Point Process (TPP) foundation model** for electronic health record (EHR) data. The model learns to predict the timing and type of future medical events from a patient's history.

### Architecture

The model uses a transformer-based architecture (documented separately) that processes sequences of medical events, each represented as a (token, timestamp) pair where timestamps are age in days since birth.

### Current Scope

- **Data**: UK Biobank, with plans to incorporate additional datasets (e.g., All of Us)
- **Event types**: ICD disease codes, with planned expansion to procedures, medications, laboratory values, and other clinical observations

### Applications

The primary focus is **disease risk prediction**: estimating the probability of specific conditions within a given time horizon. However, the framework supports a broader range of applications:

- **Trajectory simulation**: Generating realistic patient disease trajectories
- **Mechanistic understanding**: Identifying temporal patterns and disease progressions
- **Causal inference**: Elucidating potential causal factors and relationships between conditions

### Design Philosophy

The codebase prioritizes:
- **Flexibility**: Modular preprocessing transforms that can be toggled independently
- **Reproducibility**: Deterministic mode for exact replication of experiments
- **Scalability**: Memory-efficient data storage supporting large cohorts
- **Correctness**: Careful handling of edge cases (e.g., same-day clusters) that would otherwise violate model assumptions

## Modeling Same-Day Disease Clusters

### The Problem

Temporal Point Process (TPP) models typically parameterize inter-event times with exponential (or similar) distributions, which have support on (0, ∞). However, EHR data frequently contains **same-day clusters**: multiple diagnoses recorded during a single hospital visit or clinic encounter.

These clusters create inter-event times of exactly zero, which:
- Have zero probability under exponential distributions
- Cause numerical issues (log-likelihood of -∞)
- Are fundamentally incompatible with the model's assumptions

### Alternative Approaches (Not Taken)

| Approach | Drawback |
|----------|----------|
| Discard simultaneous events | Loses critical co-occurrence information |
| Add point mass at zero | Complicates the distribution; mixture models are harder to train |
| Treat as single multi-label event | Combinatorial explosion of "event types"; loses sequential structure |
| Jitter randomly | Destroys the information that events co-occurred |

### Our Solution: Dissolve and Mark

We introduce a **reversible transformation** that makes clusters compatible with exponential inter-event times while preserving all information.

**Training time** (`dissolve_clusters`):
1. For each unique timestamp with disease tokens, insert a `dx_token` marker
2. Scatter each disease token backward: `t → t - ε`, where `ε ∈ (0, Δt_prev)`
3. Re-sort by time

**Inference time** (`pack_clusters`):
1. When a `dx_token` is encountered, collect preceding disease predictions
2. Assign them all the timestamp of the `dx_token`
3. Remove the `dx_token` from output

### Why Scatter Backward?

Scattering forward would place disease tokens *after* their true occurrence time, leaking information about what diagnoses will be "confirmed" at the upcoming `dx_token`. Scattering backward is **causally safe**: the model sees diseases slightly before their true time, then learns that `dx_token` finalizes a cluster.

### Information Preservation

The transformation is **lossless**:
- The `dx_token` marks the true timestamp
- The set of diseases in the cluster is preserved
- The transformation is deterministic given the RNG state (reproducible with `deterministic=True`)

### Conceptual Interpretation

One way to interpret this: the scattered disease tokens represent the **latent clinical process** leading up to a diagnosis event. Symptoms and conditions develop over time, but are only recorded when the patient visits a clinician. The `dx_token` represents that moment of observation.

## No-Event Tokens: Input-Side Effects on NLL

### Context

The `hawkes_weibull` loss (see [MODEL.md](MODEL.md)) was designed to obviate the need for no-event tokens by providing an explicit age-dependent Weibull baseline intensity. The `weibull` loss takes a different approach: it replaces the exponential excitation kernel entirely with a Weibull density kernel that can flexibly capture short, mid, and long-range temporal behavior — potentially eliminating the need for no-event tokens without requiring a separate baseline. To compare these models fairly against the `hawkes` loss (which relies on no-event tokens), we exclude no-event target positions when computing NLL (see [EVAL_NLL.md](EVAL_NLL.md)).

However, empirically the `hawkes` model with no-event tokens still achieves slightly lower NLL on real-event targets than `hawkes_weibull` without no-event tokens. This is not solely a matter of model capacity — there is a structural advantage from no-event tokens on the **input side**. The same input-side considerations apply to `weibull` models trained without no-event tokens.

### The Input-Side Advantage

Excluding no-event tokens from the **target** side removes them from the loss computation, but they still appear as **input** tokens in the hawkes model. This provides three advantages:

1. **More frequent hidden state updates**: The transformer processes no-event input tokens and updates its hidden state at each one. This gives the model more opportunities to re-evaluate its predictions based on updated age/context. The hawkes_weibull model must carry its hidden state across longer gaps between real events with fewer update points.

2. **Shorter effective delta-t for the exponential kernel**: In the hawkes model, `Δt` in `α·exp(-β·Δt)` is the time since the *previous position*, which is often a nearby no-event token — so `Δt` is small and the excitation kernel remains informative. In the hawkes_weibull model, `Δt` is the time since the last *real event*, which can be years. The Weibull baseline compensates, but the excitation kernel is essentially zeroed out over long gaps.

3. **Finer temporal resolution**: No-event tokens act as "temporal checkpoints" that give the transformer attention mechanism more positions to attend to along the time axis. With `block_size=None` (full sequences), both models see all real events, but the hawkes model additionally has these intermediate time points.

### Implications

- The NLL comparison between models with and without no-event tokens has an inherent confound that cannot be eliminated by target-side masking alone
- The hawkes_weibull model's slightly higher NLL does not necessarily mean the Weibull baseline is insufficient — the gap may reflect the reduced temporal resolution on the input side
- Scaling up model capacity or training duration for hawkes_weibull may help close the gap, but the input-side advantage is structural
- No obvious architectural changes have been identified that would address this without reintroducing token-like temporal checkpoints
