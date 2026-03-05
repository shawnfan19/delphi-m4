# AGENTS.md — Delphi2M Loss Function Implementation Guide

This document provides everything a coding agent needs to implement new loss functions for the Delphi2M temporal point process model.

## Model Overview

**Delphi2M** is a transformer-based temporal point process (TPP) model for electronic health records (EHR). It predicts:
1. **What** event happens next (event type)
2. **When** it happens (time-to-event)

The model outputs either:
- **Log-intensities** (logits) for each event type, or
- **Distribution parameters** (e.g., α, β for Hawkes) that define intensity functions

---

## Data Format & Semantics

### Input Tensors

| Tensor | Shape | Description |
|--------|-------|-------------|
| `idx` | `(B, L)` | Token indices (event types) at each position |
| `age` | `(B, L)` | Timestamp (in days) when each event occurred |
| `targets` | `(B, L)` | Next event type (what to predict) |
| `targets_age` | `(B, L)` | Timestamp of next event (when to predict) |

### Special Tokens

| Token ID | Meaning |
|----------|---------|
| `0` | Padding — always excluded from loss |
| `1` | No-event — synthetic token to advance time without an event (enables piecewise-constant intensity) |
| `2-12` | Reserved tokens (configurable via `ignore_tokens`) |

### Time Representation

- All timestamps are in **days**
- `config.time_unit` (default: `1.0`) normalizes time (e.g., `365.25` → years)
- `delta_t = targets_age - age` is the time-to-next-event

---

## Architecture Components

### Prediction Heads

The model uses different heads depending on loss type:

```python
# For intensity-based losses (default, homo_poisson, homo_cluster_poisson)
self.lm_head = nn.Linear(config.n_embd, config.vocab_size)  # outputs log-intensities

# For hawkes loss
self.param_head = HawkesHead(n_embd, vocab_size)  # outputs alpha, beta, mu

# For neural_tpp loss
self.neural_tpp_head = NeuralTPPHead(
    n_embd, vocab_size, time_encoder, n_integrate_grid, self_terminate, self_terminate_except, time_unit
)  # neural intensity

# For cluster losses (homo_cluster_poisson)
self.aux_head = nn.Linear(config.n_embd, 1)  # outputs auxiliary rate
```

### HawkesHead

Outputs excitation parameters (α, β) from the transformer hidden state, plus a binned age-dependent baseline intensity μ:

```python
class HawkesHead(nn.Module):
    def __init__(self, n_embd: int, vocab_size: int, n_bins: int = 20):
        super().__init__()
        self.proj_alpha = nn.Linear(n_embd, vocab_size)
        self.proj_beta = nn.Linear(n_embd, vocab_size)
        self.log_mu = nn.Parameter(torch.full((n_bins, vocab_size), -5.0))

    def forward(self, x):
        param_alpha = F.softplus(self.proj_alpha(x))
        param_beta = F.softplus(self.proj_beta(x))
        return {
            "alpha": param_alpha,
            "beta": param_beta,
            "mu": F.softplus(self.log_mu),  # (n_bins, V) — age baseline
        }
```

The `log_mu` parameter provides a piecewise-constant baseline intensity over age bins, so the model can maintain non-zero intensity during long event-free intervals without relying solely on no-event tokens.

---

## Loss Function Interface

### Input Signature

```python
def loss(
    self,
    outputs: dict[str, torch.Tensor],  # from forward pass (logits, params, etc.)
    targets: torch.Tensor,              # (B, L) next event types
    age: torch.Tensor,                  # (B, L) current timestamps
    targets_age: torch.Tensor,          # (B, L) next event timestamps
) -> dict[str, torch.Tensor]:
```

### Output Format

Return a dictionary of loss components:

```python
# Each component has shape (B, L) — reduction happens outside
return {
    "loss_ce": cross_entropy_loss,      # naming convention: loss_<component>
    "loss_dt": time_likelihood_loss,
}

# Special key: "mask" (optional)
# Boolean tensor (B, L) where True = include in loss, False = exclude
return {
    "loss_nll": nll,
    "mask": valid_positions,  # e.g., ~cooccur for cluster losses
}
```

**Loss aggregation** (in training loop):
```python
loss_agg = sum([loss[key] for key in loss.keys() if key != "mask"])
```

All returned loss components are summed with equal weight. To expose tunable weights, return separate components (e.g., `loss_ce`, `loss_dt`) and let users tune via config.

---

## Temporal Point Process Theory

### Basic TPP Framework

A temporal point process models a sequence of events occurring at times $t_1 < t_2 < \cdots < t_n$. The **conditional intensity function** $\lambda(t | \mathcal{H}_t)$ defines the instantaneous rate of events given history $\mathcal{H}_t$:

$$\lambda(t) \, dt = P(\text{event in } [t, t+dt) \mid \mathcal{H}_t)$$

**Log-likelihood for observing event type $k$ at time $t_{i+1}$:**

$$\ell = \underbrace{\log \lambda_k(t_{i+1})}_{\text{event likelihood}} - \underbrace{\int_{t_i}^{t_{i+1}} \sum_v \lambda_v(\tau) \, d\tau}_{\text{survival (no event before } t_{i+1}\text{)}}$$

The integral term (called the **compensator**) penalizes high intensity over periods with no events.

### Homogeneous Poisson Process

The simplest TPP: constant intensity $\lambda_v$ for each event type.

$$\lambda_v(t) = \lambda_v \quad \text{(constant)}$$

**NLL:**
$$-\ell = -\log \lambda_k + \Delta t \sum_v \lambda_v$$

**Limitation**: Cannot model time-varying risk or event dependencies.

---

### Hawkes Process

The **Hawkes process** is a self-exciting point process where past events increase the intensity of future events. Originally developed for earthquake modeling (aftershocks trigger more aftershocks).

#### Classical Multivariate Hawkes Process

For event types $v \in \{1, \ldots, V\}$, the intensity is:

$$\lambda_v(t) = \mu_v + \sum_{t_i < t} \phi_{v}(t - t_i)$$

Where:
- $\mu_v \geq 0$ is the **baseline intensity** (spontaneous event rate)
- $\phi_v(s)$ is the **triggering kernel** — how past events influence future intensity
- The sum runs over all past events $t_i$ in the history

#### Exponential Kernel (Most Common)

$$\phi_v(s) = \alpha_v \cdot e^{-\beta_v \cdot s}$$

Where:
- $\alpha_v > 0$ is the **excitation amplitude** — how much each past event boosts intensity
- $\beta_v > 0$ is the **decay rate** — how quickly the excitation fades

This gives:

$$\lambda_v(t) = \mu_v + \sum_{t_i < t} \alpha_v \cdot e^{-\beta_v (t - t_i)}$$

#### Key Properties

1. **Self-excitation**: Events increase the probability of future events
2. **Memory decay**: Influence of past events fades exponentially
3. **Clustering**: Events tend to cluster in time (bursts of activity)
4. **Branching structure**: Can be viewed as an immigration-birth process:
   - "Immigrants" arrive at baseline rate $\mu$
   - Each event spawns "children" according to the kernel

#### Stability Condition

For the process to be stationary (not explode), the **branching ratio** must be < 1:

$$\sum_v \frac{\alpha_v}{\beta_v} < 1$$

This ensures each event triggers < 1 child on average.

#### Full History vs. Single-Step: Delphi's Approach

**Classical Hawkes** sums over the entire history:
$$\lambda_v(t) = \mu_v + \sum_{\text{all } t_i < t} \alpha_v e^{-\beta_v(t - t_i)}$$

This requires tracking every past event and computing its decayed contribution — computationally expensive and requires specialized recursive updates.

**Delphi's implementation** simplifies to single-step (only the immediately preceding event):
$$\lambda_v(t) = \alpha_v \cdot e^{-\beta_v(t - t_{\text{prev}})}$$

Where $t_{\text{prev}}$ is the timestamp of the most recent event.

#### Why Single-Step is Reasonable

This simplification works because **the transformer implicitly captures history**:

1. **Context-dependent parameters**: Unlike classical Hawkes where $\alpha_v, \beta_v$ are global constants, Delphi's parameters are **position-dependent outputs** of the transformer:
   ```
   α_v^{(i)}, β_v^{(i)} = f_θ(x_1, x_2, ..., x_i)
   ```
   The transformer has seen the entire history $(x_1, ..., x_i)$ and encodes it into the hidden state that produces these parameters.

2. **Neural baseline absorption**: The classical baseline $\mu_v$ and cumulative historical excitation $\sum_{t_j < t_{\text{prev}}} \alpha_v e^{-\beta_v(t - t_j)}$ are implicitly absorbed into the learned $\alpha_v^{(i)}$. The transformer learns to output higher $\alpha$ when history suggests elevated risk.

3. **Flexible dependency structure**: Classical Hawkes assumes all past events contribute via the same exponential decay. The transformer can learn **arbitrary dependency patterns** — some past events might matter more than others, dependencies might be non-monotonic, etc.

4. **Computational simplicity**: No need for recursive intensity updates or tracking $O(L)$ decaying terms. The forward pass is a standard transformer + simple exponential computation.

#### Trade-offs

| Aspect | Classical Hawkes | Delphi Single-Step |
|--------|------------------|-------------------|
| History dependence | Explicit mathematical sum | Implicit in transformer |
| Parameters | Global $\mu, \alpha, \beta$ | Context-dependent $\alpha^{(i)}, \beta^{(i)}$ |
| Interpretability | High (clear excitation structure) | Lower (neural black box) |
| Flexibility | Limited to exponential decay | Arbitrary learned patterns |
| Computation | $O(L^2)$ or recursive $O(L)$ | $O(1)$ per position (after transformer) |

#### When to Use Full History

If you need **explicit multi-event excitation** (e.g., modeling aftershock cascades where the 3rd-to-last event still matters independently), you would need to implement full-history Hawkes. This requires:

1. Cumulative intensity tracking across positions
2. Recursive update formulas for efficiency:
   $$R_v^{(i)} = e^{-\beta_v \Delta t_{i-1}} \cdot (R_v^{(i-1)} + \alpha_v)$$
   where $R_v^{(i)}$ tracks the cumulative decayed excitation

For most EHR applications, the single-step approximation with transformer context is sufficient — the transformer learns when history matters.

#### Hawkes NLL Derivation

For the single-step exponential kernel, the integral has a closed form:

$$\int_{t_i}^{t_{i+1}} \alpha_v e^{-\beta_v(\tau - t_i)} d\tau = \frac{\alpha_v}{\beta_v}\left(1 - e^{-\beta_v \Delta t}\right)$$

**Full NLL:**
$$-\ell = -\log \alpha_k + \beta_k \Delta t + \sum_v \frac{\alpha_v}{\beta_v}\left(1 - e^{-\beta_v \Delta t}\right)$$

Note: The actual implementation includes a binned baseline $\mu_v(\text{bin}(\tau))$ via `HawkesHead.log_mu` — see the `hawkes` loss section below.


---

## Existing Loss Functions

### `default`

Decomposes TPP likelihood into:
1. **Cross-entropy**: probability that next event is type k (multinomial over intensities)
2. **Exponential NLL**: probability of observed time-to-event (exponential with summed intensity)

```python
# Cross-entropy over event types
loss_ce = F.cross_entropy(logits.permute(0, 2, 1), targets, reduction="none")

# Time likelihood: exponential distribution
loss_dt = exponential_nll(
    delta_t=dt,
    log_lambda=torch.logsumexp(logits, -1),
    t_min=config.t_min,
)
```

### `homo_poisson`

Mathematically equivalent to `default`, but computed as standard Poisson process NLL:

```python
def nll_homogeneous_poisson(
    log_intensity: torch.Tensor,  # (B, L, V)
    targets: torch.Tensor,        # (B, L)
    delta_t: torch.Tensor,        # (B, L)
) -> torch.Tensor:
    # log λ_k for observed event k
    part1 = torch.gather(log_intensity, dim=-1, index=targets.unsqueeze(-1))

    # -∫ Σ_v λ_v dt = -Σ_v λ_v · Δt (homogeneous)
    log_sum_intensity = torch.logsumexp(log_intensity, dim=-1, keepdim=True)
    part2 = -torch.exp(log_sum_intensity) * delta_t.unsqueeze(-1)

    return -(part1 + part2)
```

**The Role of No-Event Tokens**: The homogeneous Poisson assumption—that intensity is constant between events—becomes problematic over long time gaps. In EHR data, a patient might have no recorded events for 20 years, but their risk profile at age 60 is very different from age 40. To address this, **no-event tokens** (token ID `1`) are synthetically inserted into sequences to break long intervals into shorter segments. This transforms the model from a globally homogeneous process into a **piecewise-constant intensity** process: within each segment, intensity is constant, but the model re-evaluates and outputs new intensities at each no-event token based on the updated age and context. Without no-event tokens, the model would be forced to predict with the same intensity over arbitrarily long horizons, severely limiting its expressiveness for time-varying risk.


### `homo_cluster_poisson`

Handles **event clusters** (multiple events at the same timestamp) by introducing an auxiliary "data entry" event:

- Events occur continuously but are only observed at discrete recording times
- `aux_rates`: intensity of the recording/observation process
- Computes: P(observed events occurred before recording)

Returns three components:
```python
return {
    "loss_nll": nll,           # standard TPP likelihood for recording times
    "loss_cluster": nll_cluster, # likelihood of event occurrences within cluster
    "mask": ~cooccur,          # exclude cluster-continuation positions
}
```

### `hawkes`

Parametric TPP with exponential decay kernel and a **binned age-dependent baseline intensity**:

$$\lambda_v(\tau) = \mu_v(\text{bin}(\tau)) + \alpha_v \cdot \exp(-\beta_v \cdot (\tau - t_i))$$

Uses `HawkesHead` to output `alpha` (excitation), `beta` (decay rate), and `mu` (binned baseline, `nn.Parameter` of shape `(n_bins, V)`).

```python
def nll_hawkes(
    mu: torch.Tensor,          # (n_bins, V) age-binned baseline intensity
    alpha: torch.Tensor,       # (B, L, V) excitation amplitude
    beta: torch.Tensor,        # (B, L, V) decay rate
    age: torch.Tensor,         # (B, L)
    idx: torch.Tensor,         # (B, L) for self-termination
    targets_age: torch.Tensor, # (B, L)
    targets: torch.Tensor,     # (B, L)
) -> torch.Tensor:
    # Part 1: log(μ_k(bin(t_{i+1})) + α_k · exp(-β_k · Δt))
    # Part 2: -∫ Σ_v [μ_v(τ) + α_v·exp(-β_v·(τ-t_i))] dτ
    #   μ integral: bin_mu_compensator (overlap × mu)
    #   α integral: (α/β)·(1 - exp(-β·Δt))
    ...
```

The binned baseline μ provides piecewise-constant age-dependent risk, so the model can maintain non-zero intensity during long event-free intervals.

**Sampling not implemented**: `sample_next` raises `NotImplementedError` for this loss.

### `neural_tpp`

A fully neural TPP where the intensity is an arbitrary function of the transformer hidden state and elapsed time:

$$\lambda_v(\tau) = \text{softplus}\!\bigl(\text{net}(h + \text{time\_encode}(\tau - t_i))\bigr)_v$$

Unlike the Hawkes head, which constrains intensity to exponential decay, `NeuralTPPHead` uses an MLP that can learn any intensity shape — rising, oscillating, multi-modal, etc.

#### NeuralTPPHead

```python
class NeuralTPPHead(nn.Module):
    def __init__(
        self,
        n_embd,
        vocab_size,
        time_encoder,
        n_integrate_grid=20,
        self_terminate=True,
        self_terminate_except=None,
        time_unit=1.0,
    ):
        self.time_encoder = time_encoder   # shared AgeEncoding module
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.GELU(),
            nn.Linear(n_embd, vocab_size),
        )
        self.n_integrate_grid = n_integrate_grid
        self.self_terminate = self_terminate
        self.register_buffer("terminate_except", torch.tensor(...))
        self.time_unit = time_unit

    def forward(self, h, delta_t):
        # h: (B, L, n_embd), delta_t: (B, L) or (B, L, G)
        time_emb = time_encoder(delta_t)   # sinusoidal encoding of elapsed time
        return self.net(h + time_emb)       # raw pre-softplus values

    def nll(self, h, targets, age, targets_age, idx):
        ...
```

The head accepts `delta_t` as either `(B, L)` (single time per position, used for Part 1) or `(B, L, G)` (a grid of times per position, used for Part 2). When given a grid, it broadcasts `h` across grid points and returns `(B, L, G, V)`.

`nll()` is implemented as a method on `NeuralTPPHead`, not a standalone function.

#### NLL decomposition

**Part 1 — log-intensity of the observed event**:

A single forward pass at Δt = t_{i+1} − t_i gives λ_v for all event types at the observation time. Gather the observed type k:

$$\text{Part 1} = \log \lambda_k(t_{i+1})$$

**Part 2 — compensator (survival term)**:

Because the intensity is an arbitrary neural network, the integral ∫ Σ_v λ_v(τ) dτ has no closed form. It is approximated via **trapezoidal numerical integration**:

1. Build a unit grid `[0, 1]` of G points (via `torch.linspace`)
2. Scale by each position's Δt: `grid_times = t_unit * delta_t` → `(B, L, G)`
3. Evaluate intensity at all grid points: `head(h, grid_times)` → `(B, L, G, V)`
4. Sum over event types and integrate: `torch.trapezoid(total_intensity, grid_times) / time_unit`

The grid spacing is **proportional to the inter-event interval**: the unit grid is shared across all positions, and scaling by each position's Δt means short intervals get fine physical spacing and long intervals get coarser spacing. The number of grid points G is set directly by `n_integrate_grid` (default 20) and is fixed across batches. The compensator is divided by `time_unit` to match the normalization used by other losses, while the time encoder still receives raw days.

#### Self-termination in the compensator

When `self_terminate=True`, the compensator zeros out intensity for event types already seen in the history (preventing the model from being penalised for not predicting impossible re-occurrences). The `(B, L, G, V)` intensity grid is reshaped to `(B, L*G, V)`, `idx` is tiled across grid points, and `self_terminate()` is applied, then reshaped back.

#### Comparison with Hawkes

| Aspect | Hawkes | Neural TPP |
|--------|--------|------------|
| Intensity shape | Exponential decay only | Arbitrary (learned) |
| Compensator | Closed-form: (α/β)(1 − e^{−βΔt}) | Numerical integration (trapezoidal) |
| Parameters per position | α, β (2 × V) + global μ | MLP weights (shared) |
| Compute cost | O(V) per position | O(G × V) per position (G grid evaluations) |
| Baseline intensity | Binned μ(age) parameter | Absorbed into the MLP |

**Sampling not implemented**: `sample_next` raises `NotImplementedError` for this loss.

---

## Helper Functions

### `untie_idx` and `untie`: Handling Tied Events

In EHR data, multiple events can share the same timestamp (e.g. several diagnoses recorded on the same visit). At position `i` in a tied cluster, `targets_age[i] == age[i]`, giving `delta_t = 0`. This causes two problems:
- **For intensity models**: Δt = 0 events have logits produced by an earlier position in the cluster, not a fresh prediction after the event.
- **For parametric models (Weibull)**: Δt = 0 cannot be evaluated (log(0) or PDF(0) = 0).

`untie_idx` computes the remapping — for tied positions it returns the index of the first position in the tie group:

```python
def untie_idx(age: torch.Tensor, targets_age: torch.Tensor) -> torch.Tensor:
    """
    Returns (B, L) index tensor that maps each position to itself, except
    tied positions (delta_t == 0) which map back to the first position of the tie.
    """
    dt = targets_age - age
    is_tie = dt == 0
    is_tie[age == -1e4] = False  # ignore padding
    corr_idx = torch.where(is_tie, 0, torch.arange(age.shape[1], device=age.device))
    corr_idx = torch.cummax(corr_idx, dim=1)[0]
    return corr_idx
```

`untie` applies this remapping to the full `outputs` dict and `age` tensor in one call:

```python
def untie(
    outputs: dict[str, torch.Tensor],
    age: torch.Tensor,
    targets_age: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """
    Remaps outputs and age so that all tied positions use the outputs/age of
    the first position in their tie group.
    Handles 2-D (B, L) and 3-D (B, L, V) output tensors; skips ≤1-D tensors.
    """
```

**Implicit assumptions**:

1. **Sequences are time-sorted** — `cummax` propagates the last non-tied index
   forward. If events are out of chronological order, the remapping silently points
   to the wrong position. The dataset and `generate()` both enforce sorted order.

2. **Padding sentinel is exactly `-1e4`** — `is_tie[age == -1e4] = False` is
   hardcoded. If the padding convention changes, padding positions that happen to
   have coincident `age` and `targets_age` would be misidentified as ties.

3. **Every cluster is preceded by at least one non-tied position** — when
   `is_tie=True`, `untie_idx` temporarily sets that position's index to 0 before
   the `cummax`. If the cluster starts at position 0 itself, those positions remain
   mapped to index 0 (themselves), which still has Δt=0. In practice, sequences
   always begin with padding positions (excluded by the `-1e4` guard), so a
   non-tied real-event position always precedes any cluster.

4. **All tensors in `outputs` are at most 3-D** — for `dim > 2`, `untie` uses
   `corr_idx.unsqueeze(-1)` as the index for `take_along_dim`. A 4-D tensor (e.g.
   `attn_mask`) would silently receive wrong indices. This is safe today because
   `attn_mask` is stored in `misc`, not `outputs`.

5. **`targets_age[i]` is the immediately-following event's timestamp** — tie
   detection uses `targets_age - age == 0`. If `targets_age` contains anything
   other than the next event's timestamp (e.g. an arbitrary evaluation horizon),
   the tie criterion is meaningless. This is guaranteed by the data loading
   convention (`x0, t0, x1, t1 = ds.get_batch(...)` where `t1` is the shifted
   `t0`).

**`mask_ties` config field** — controls whether untying is applied during training:

```python
@dataclass
class Delphi2MConfig:
    mask_ties: bool = True   # default ON
    ...
```

When `mask_ties=True`, `model.loss()` calls `untie(outputs, age, targets_age)` before computing the NLL. This means all events in a simultaneous cluster are evaluated against the model predictions *before* any of them were observed, keeping `delta_t > 0`.

**Comparability caveat**: Models trained with different `mask_ties` settings are **not directly comparable**. The correct setting for all non-cluster losses is `mask_ties=True`.

When computing likelihoods outside `model.loss()` (e.g. in eval scripts), mirror the training behaviour:
```python
if model.config.mask_ties:
    outputs, t0 = untie(outputs, t0, t1)
```

**Constraint**: `mask_ties=False` is required for `homo_cluster_poisson` (enforced in `__post_init__`), which has its own cluster handling via `multi_hot`.

### `multi_hot`: Cluster Encoding

Groups co-occurring events into clusters with multi-hot encoding:

```python
def multi_hot(
    targets: torch.Tensor,
    targets_age: torch.Tensor,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        hot_targets: (B, L, V) multi-hot encoding of events per cluster
        cooccur: (B, L) boolean mask, True for cluster-continuation positions
    """
    ...
```

### `self_terminate_single` and `self_terminate`: No-Repeat Masking

Two utilities in `delphi/model/utils.py` suppress already-seen tokens in logits:

```python
def self_terminate_single(
    idx: torch.Tensor,           # (B, L) full sequence
    logits: torch.Tensor,        # (B, V) logits at the last position
    terminate_except: torch.Tensor,  # token IDs exempt from suppression (e.g. no-event)
) -> torch.Tensor:
    """
    Sets logits to -inf for tokens already present in idx (except exempt ones).
    Used inside sample_next() for single-step generation.
    Returns: masked logits (B, V)
    """
```

```python
def self_terminate(
    idx: torch.Tensor,           # (B, L) full sequence
    logits: torch.Tensor,        # (B, L, V) logits at all positions
    terminate_except: torch.Tensor,
) -> torch.Tensor:
    """
    Causal variant: at position j, suppresses tokens that appeared in positions 0..j.
    Used after generate() to clean up logits for the full completed trajectory.
    Returns: masked logits (B, L, V)
    """
```

The distinction: `self_terminate_single` operates on the last-position logits `(B, V)` and is called inside `sample_next` during generation. `self_terminate` operates on all-position logits `(B, L, V)` and is called in post-generation analysis (e.g. `apps/sampling.py`) to obtain causally-masked logits for the full trajectory.

---

## Sampling Interface

Each loss must implement sampling for generation:

```python
@torch.no_grad()
def sample_next(
    self,
    outputs: dict[str, torch.Tensor],
    idx: torch.Tensor,              # (B, L) full current sequence (for self-termination)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        idx_next: (B, N) next token(s) — N can vary by loss type
        time_til_next: (B, N) time until next event(s)
    """
```

`sample_next` is responsible for **self-termination** — masking already-seen tokens before sampling. It reads `self.config.self_terminate_except` (list of tokens exempt from suppression, e.g. `[1]` for no-event). The utility `self_terminate_single(idx, logits, terminate_except)` from `delphi/model/utils.py` handles this masking at the last sequence position.

### Existing Samplers

```python
def sample_competing_exponentials(
    logits: torch.Tensor,
    clamp_min: float = 0.0,
    clamp_max: float = 365.25 * 80.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Inverse CDF sampling for competing exponential processes.
    Each event type races; earliest wins.

    Returns: (next_token, time_til_next), each (B, 1)
    """
    t_next = torch.clamp(
        -torch.exp(-logits) * torch.rand(logits.shape, device=logits.device).log(),
        min=clamp_min,
        max=clamp_max,
    ).min(1)
    return t_next[1][:, None], t_next[0][:, None]
```

### Losses without samplers

`hawkes` and `neural_tpp` do not have sampling implementations. `sample_next` raises `NotImplementedError` for these losses.

---

## Implementation Checklist

When implementing a new loss function:

### 1. Define the NLL Function

```python
def nll_your_loss(
    # parameters from outputs dict
    log_intensity: torch.Tensor,  # or other params
    targets: torch.Tensor,
    age: torch.Tensor,
    targets_age: torch.Tensor,
    # any additional args
) -> torch.Tensor:  # or dict for multiple components
    """
    Compute negative log-likelihood for your TPP model.

    Args:
        ...

    Returns:
        NLL tensor of shape (B, L)
    """
    delta_t = targets_age - age
    delta_t /= time_unit  # normalize if needed

    # Your NLL computation here
    ...

    return nll
```

### 2. Add Config Fields (if needed)

In `Delphi2MConfig`:
```python
@dataclass
class Delphi2MConfig:
    ...
    your_loss_param: float = 1.0  # add any loss-specific hyperparameters
```

Existing self-termination fields (already in config, no need to add):
```python
self_terminate: bool = True                          # enable no-repeat suppression in sample_next
self_terminate_except: None | list = field(default_factory=lambda: [1])  # tokens exempt (no-event by default)
```

### 3. Add Head(s) in `__init__` (if needed)

```python
def __init__(self, config: Delphi2MConfig):
    ...

    # If your loss is parametric (not intensity-based)
    parametric_losses = {"hawkes", "your_loss"}  # add to set

    # If your loss needs auxiliary outputs
    if "your_loss" in config.loss:
        self.your_head = nn.Linear(config.n_embd, output_dim)
```

### 4. Add Forward Pass Logic (if needed)

```python
def forward(self, idx, age, targets=None, targets_age=None):
    ...

    if hasattr(self, "your_head"):
        outputs["your_param"] = self.your_head(x)
```

### 5. Add Loss Branch

```python
def loss(self, outputs, targets, age, targets_age):
    ...
    elif self.config.loss == "your_loss":
        nll = nll_your_loss(
            your_param=outputs["your_param"],
            targets=targets,
            age=age,
            targets_age=targets_age,
        )
        return {"loss_nll": nll}
```

### 6. Add Sampling Method

```python
def sample_next(self, outputs: dict, idx: torch.Tensor):
    ...
    elif self.config.loss == "your_loss":
        logits = outputs["logits"][:, -1, :]
        logits = self_terminate_single(
            idx=idx,
            logits=logits,
            terminate_except=torch.tensor(self.config.self_terminate_except).to(idx.device),
        )
        idx_next, time_til_next = sample_your_loss(
            your_param=outputs["your_param"],
            ...
        )
    return idx_next, time_til_next
```

Remember: `sample_next` now receives the **full current sequence** (`idx`) and is responsible for suppressing already-seen tokens via `self_terminate_single`. `generate()` no longer applies any logit masking externally.

---

## Numerical Stability Conventions

### Use Log-Space

Prefer log-space computations to avoid underflow:

```python
# Bad: direct multiplication
prob = intensity * torch.exp(-intensity * dt)

# Good: log-space
log_prob = log_intensity - torch.exp(log_intensity) * dt
```

### Use `logsumexp`

```python
# Bad: sum then log
log_total = torch.log(torch.exp(log_a) + torch.exp(log_b))

# Good: numerically stable
log_total = torch.logsumexp(torch.stack([log_a, log_b]), dim=0)
```

### Epsilon Constants

Use inline epsilon for division/log stability:

```python
eps = 1e-8
log_val = torch.log(val + eps)
ratio = a / (b + eps)
```

### Clamping Delta-t

Only clamp `delta_t` when 0 is not in the distribution's support:

```python
# Exponential/Weibull: 0 not in support, need clamping
dt = torch.clamp(targets_age - age, min=config.t_min)

# If your distribution supports dt=0, no clamping needed
dt = targets_age - age
```

---

## Type Hints

All functions must include type hints:

```python
def nll_your_loss(
    log_intensity: torch.Tensor,
    targets: torch.Tensor,
    delta_t: torch.Tensor,
    time_unit: float = 365.25,
) -> torch.Tensor:
    ...
```

---

---

---

## File Locations

- **Model**: `delphi/model/transformer.py`
- **Config**: `Delphi2MConfig` in same file
- **Loss functions & utilities**: `delphi/model/utils.py`
- **TPP evaluation class**: `delphi/model/tpp.py`

---

## DelphiM4

### Overview

**DelphiM4** (`model_type = "delphi-m4"`, implemented in `delphi/model/multimodal.py`) is the multimodal extension of Delphi2M. It fuses biomarker modalities (e.g. blood biomarkers) with disease event tokens early in the transformer forward pass. The transformer backbone, cross-entropy loss, and exponential NLL loss are identical to Delphi2M.

Key differences from Delphi2M:
- Additional input tensors: `biomarker`, `mod_age`, `mod_idx`
- `DelphiEmbedding` replaces the standard embedding to project biomarker features
- `fuse_embed()` merges biomarker and token embeddings into a single sorted sequence
- Ablation modes allow isolating the contribution of biomarkers vs. disease tokens

---

### Additional Inputs (beyond Delphi2M)

| Tensor | Shape | Description |
|--------|-------|-------------|
| `biomarker` | `dict[Modality, Tensor]` | Per-modality raw biomarker features; each value is `(N_mod, input_size)` where `N_mod` is the total number of measurements of that modality across the batch |
| `mod_age` | `(B, n_biomarkers)` | Timestamps (days) of each biomarker slot; padding sentinel = `-1e4` |
| `mod_idx` | `(B, n_biomarkers)` | `Modality.value` integer for each biomarker slot; `0` = padding |

`n_biomarkers` is the fixed number of biomarker slots per sample (set by the dataset). Sparse per-modality data is scattered into these dense slots via the `mod_idx` mask.

---

### DelphiM4Config Fields

```python
@dataclass
class DelphiM4Config:
    block_size: None | int = 256        # max sequence length (None = unlimited)
    vocab_size: int = 1270
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 120
    dropout: float = 0.1
    token_dropout: float = 0.0          # dropout on token embeddings (scaled out at inference)
    t_min: float = 0.1                  # epsilon for time-to-next-event
    bias: bool = True
    mask_ties: bool = True              # handle simultaneous events (same as Delphi2M)
    attn_mask: str = "time"             # "time" (timestamp-causal) or "triangular"
    weight_tying: bool = True           # tie token embedding ↔ lm_head weights
    ignore_tokens: list = [0, 2, ..., 12]  # tokens excluded from loss targets
    biomarkers: dict[str, BiomarkerEmbedConfig] = {}  # name → embed config per modality
    modality_emb: bool = True           # add a per-modality learned embedding to biomarker tokens
    ablate_biomarker: None | str = None # ablation mode: None / "biomarker" / "token" / "both"
    ce_beta: float = 1.0                # weight on cross-entropy loss
    dt_beta: float = 1.0                # weight on exponential NLL loss
    fuse: str = "early"                 # fusion strategy; only "early" is implemented
```

`biomarkers` maps lowercase modality names (matching `Modality` enum keys) to `BiomarkerEmbedConfig` dicts:

```python
config.biomarkers = {
    "blood": {"input_size": 64, "projector": "mlp", "n_layers": 2, "n_hidden": 128},
}
```

---

### BiomarkerEmbedConfig / BiomarkerEmbedding

**`BiomarkerEmbedConfig`** is a `TypedDict`:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `input_size` | `int` | yes | Dimensionality of the raw biomarker feature vector |
| `projector` | `str` | yes | `"linear"` or `"mlp"` |
| `n_layers` | `int \| None` | no | Number of layers (required for `"mlp"`) |
| `n_hidden` | `int \| None` | no | Hidden dimension (required for `"mlp"`) |
| `bias` | `bool` | no | Whether to include bias terms (default: `False`) |

**`BiomarkerEmbedding`** projects raw biomarker features `(N_mod, input_size)` → `(N_mod, n_embd)`:

- `"linear"`: single `nn.Linear(input_size, n_embd)`
- `"mlp"`: stack of `nn.Linear` + `nn.ReLU`, final layer projects to `n_embd`

---

### DelphiEmbedding

`DelphiEmbedding` computes embeddings for both disease tokens and biomarkers.

```python
def forward(idx, age, mod_idx, mod_age, biomarker_x) -> (emb, biomarker_emb, raw):
```

**Disease token path** (same as Delphi2M):
1. `idx_emb = token_embedding(idx)` — lookup `(B, L, n_embd)`
2. `idx_emb = token_drop(idx_emb) * (1 - token_dropout)` — dropout with scale compensation
3. `age_emb = age_encoding(age.unsqueeze(-1))` — sinusoidal age encoding `(B, L, n_embd)`
4. `emb = idx_emb + age_emb`

**Biomarker path**:
1. `mod_age_emb = age_encoding(mod_age.unsqueeze(-1))` — age encoding for all biomarker slots `(B, n_biomarkers, n_embd)`
2. For each modality in `biomarker_x`:
   - `biomarker_embed[modality](biomarker_x[modality])` projects `(N_mod, input_size)` → `(N_mod, n_embd)`
   - Add age encoding at measurement timestamps: `biomarker_emb[modality] += mod_age_emb[mod_mask]`
   - If `modality_emb=True`: add a learned per-modality scalar embedding `mod_embedding(modality.value)`

**Returns**: `(emb, biomarker_emb, raw)`
- `emb`: `(B, L, n_embd)` — disease token embeddings
- `biomarker_emb`: `dict[Modality, Tensor]` — `(N_mod, n_embd)` per modality (sparse)
- `raw`: dict `{"idx": idx_emb, "age": age_emb, "mod_age": mod_age_emb, "biomarker": biomarker_emb}` — for analysis/debugging only; assigned to `_` in model forward

---

### fuse_embed()

```python
def fuse_embed(mod_idx, mod_age, mod_emb, emb, age) -> (fused_emb, fused_age, fused_mod_idx):
```

**Purpose**: Merge sparse per-modality biomarker embeddings and disease token embeddings into one time-sorted sequence.

**Steps**:
1. Scatter sparse `mod_emb` dict → dense `(B, n_biomarkers, n_embd)` tensor (`mod_emb_dense`) using `mod_idx` masks
2. Concatenate along the sequence dimension (biomarkers first, then disease tokens):
   - `fused_emb_unsorted = cat([mod_emb_dense, emb], dim=1)` → `(B, n_biomarkers + L, n_embd)`
   - `fused_age_unsorted = cat([mod_age, age], dim=1)` → `(B, n_biomarkers + L)`
   - `fused_idx_unsorted = cat([mod_idx, ones_like(age)], dim=1)` → disease tokens get `fused_mod_idx = 1`
3. Sort by timestamp: `sort_indices = argsort(fused_age_unsorted, stable=True, dim=1)`
   - `stable=True` ensures biomarker slots (placed first in the unsorted tensor) precede disease tokens with equal timestamps
4. Apply sort to all three tensors

**Returns**: `(fused_emb, fused_age, fused_mod_idx)` — three parallel views of the merged `(B, n_biomarkers+L, ...)` sequence passed to the transformer.

---

### ablate_biomarker

Ablation modes control which information context is available during the forward pass, enabling ablation studies. Loss is always computed on positions where `age > min_mod_age` (disease tokens after the first biomarker measurement), providing a consistent target set across all modes.

`min_mod_age` is the earliest non-padding biomarker timestamp per sample (`mod_age` with padding `-1e4` replaced by `+inf`, then `min(dim=1)`).

**Token zeroing** (applied before `fuse_embed`):

| Mode | Token zeroing |
|------|--------------|
| `None` | None |
| `"biomarker"` | `idx *= 0` — all disease token indices set to zero (padding) |
| `"token"` or `"both"` | `idx[age > min_mod_age] = 0` — disease tokens after first biomarker zeroed |

**Pad mask** (applied after `fuse_embed`, controls attention column availability):

The base pad mask is `fused_age != -1e4` (excludes padding slots). Ablation narrows it further:

| Mode | Additional pad mask constraint |
|------|-------------------------------|
| `None` | None (all non-padding positions visible) |
| `"biomarker"` | `pad *= (fused_mod_idx != 1)` — exclude disease token columns |
| `"token"` | `pad *= (fused_age <= min_mod_age) AND (fused_mod_idx == 1)` — only disease tokens before first biomarker |
| `"both"` | `pad *= (fused_age <= min_mod_age)` — biomarkers at first visit + disease tokens before first biomarker |

**Attention masking note**: `causal_attention_mask` applies `pad_mask` to **columns only** — a position with `pad=False` cannot be attended to by others, but it CAN still attend to valid positions (plus itself via forced diagonal). This is why positions after the first biomarker can still produce meaningful logits for loss computation even when they are excluded from the pad mask.

**Summary table**:

| Mode | Context available to model | Zeroed/masked |
|------|---------------------------|---------------|
| `None` | All tokens + all biomarkers | Nothing |
| `"biomarker"` | Only biomarkers | `idx *= 0`; disease token columns excluded from pad |
| `"token"` | Only disease tokens before first biomarker | Post-biomarker disease tokens zeroed; biomarker + post-biomarker disease columns excluded |
| `"both"` | Biomarkers at first visit + disease tokens before first biomarker | Post-biomarker disease tokens zeroed; anything after `min_mod_age` excluded from pad |

**UKB context**: designed primarily for blood biomarkers collected at a single recruitment visit, so `min_mod_age ≈ visit_age` and all biomarkers share approximately the same timestamp.

---

### Forward Pass Summary

```
idx, age, biomarker, mod_age, mod_idx
      │
      ▼ (ablate_biomarker: optionally zero idx)
      │
      ▼ DelphiEmbedding.forward(idx, age, mod_idx, mod_age, biomarker)
      │   → emb (B, L, H), biomarker_emb dict, raw (discarded)
      │
      ▼ fuse_embed(mod_idx, mod_age, biomarker_emb, emb, age)
      │   → fused_emb (B, n_bio+L, H), fused_age, fused_mod_idx
      │
      ▼ pad = fused_age != -1e4  [+ ablation narrowing]
      │
      ▼ causal_attention_mask(pad=pad, timestep=fused_age)
      │   → attn_mask (B, n_bio+L, n_bio+L)
      │
      ▼ Dropout → Transformer blocks → LayerNorm
      │   → x (B, n_bio+L, H)
      │
      ▼ lm_head(x)
      │   → logits (B, n_bio+L, vocab_size)
      │
      ▼ fuse_targets_mask(targets_age, mod_age)  [if targets provided]
      │   → is_target bool mask; extract logits[is_target] → (B, L, vocab_size)
      │
      ▼ loss(logits, targets, age, targets_age)
          is_valid_target = targets != 0 AND targets not in ignore_tokens
          [+ ablation: is_valid_target *= age > min_mod_age]
          loss_ce = mean(cross_entropy[is_valid_target]) * ce_beta
          loss_dt = mean(exponential_nll[is_valid_target]) * dt_beta
```

**Returns**: `({"logits": logits, "attn_mask": attn_mask}, loss_dict, att)`

---

### fuse_targets_mask()

```python
def fuse_targets_mask(targets_age: Tensor, mod_age: Tensor) -> Tensor:
```

**Purpose**: Identify the optimal **readout positions** in the fused `(B, n_biomarkers+L)` sequence for each target prediction — i.e., the positions that have integrated the maximum available context (biomarkers + disease tokens) up to each prediction time.

**Method**: Concatenates `[zeros(mod_age), ones(targets_age)]` and sorts by `argsort(stable=True)`. Returns a `(B, n_biomarkers+L)` binary tensor where `1` marks readout positions.

**Why `targets_age`, not `age`**: `fuse_embed` sorts by `[mod_age, age]` to build the fused sequence, but `fuse_targets_mask` intentionally sorts by `[mod_age, targets_age]`. The readout position for predicting `targets[i]` (which occurs at `targets_age[i]`) should be the last fused position with `fused_age ≤ targets_age[i]` — this ensures the logits come from a position that has attended to all biomarkers and disease tokens available before the prediction time. A biomarker measured between `age[i]` and `targets_age[i]` *should* inform the prediction of `targets[i]`, and sorting by `targets_age` guarantees its logits are read from a position that attended to that biomarker.

**Usage in forward**: `logits[is_target.bool()].view(*idx.shape, -1)` extracts the `(B, L, vocab_size)` logits at readout positions; `fused_age[is_target.bool()].view(*idx.shape)` extracts the corresponding timestamps for `delta_t` computation in the loss.

---

### File Locations (DelphiM4)

- **Model + Config**: `delphi/model/multimodal.py`
- **Modality enum**: `delphi/multimodal.py` (`Modality`, `module_name`)
- **Shared transformer components**: `delphi/model/transformer.py` (`AgeEncoding`, `Block`, `LayerNorm`, `causal_attention_mask`, `exponential_nll`)
- **Training script**: `apps/train-delphi-m4.py`

---

## Future Direction: ODE-TPP and Continuous-Time Multi-Task Prediction

### Motivation

The goal is a model that can decode the hidden state onto multiple output spaces at arbitrary times — e.g. predict both event intensities and biomarker values at any query time τ.

### ODE-TPP Sketch

Replace the `neural_tpp` time-additive hidden state with a Neural ODE:

```
z(t_i) = h_i                    (transformer hidden state as initial condition)
dz/dτ  = f_θ(z, τ)              (learned continuous dynamics)
λ_v(τ) = softplus(W_tpp · z(τ)) (TPP head)
bio(τ)  = W_bio · z(τ)          (biomarker regression head)
```

**Compensator**: use an **augmented ODE** — extend the state with a scalar `c` that accumulates the survival integral:

```
d/dτ [z]  =  [f_θ(z, τ)           ]    initial: [h_i, 0]
     [c]     [Σ_v softplus(W·z + b)]
```

Solving once from `t_i` to `t_{i+1}` yields both `z(t_{i+1})` (for log λ_k) and `c(t_{i+1})` (compensator directly) — no separate grid integration needed.

### Why not do this now

`neural_tpp` already supports multi-task decoding at arbitrary τ via `z(τ) = h + time_encode(τ − t_i)`. Additional projection heads can be attached to `z(τ)` for any output modality. The ODE would give richer inter-event dynamics but is significantly more expensive (multiple ODE solver evaluations per position, adjoint-method gradients, `torchdiffeq` dependency).

**The binding constraint**: meaningful biomarker interpolation requires **repeated longitudinal measurements** as training signal. UKB provides biomarkers at a single recruitment visit — no ground-truth values exist at intermediate times, so the interpolation trajectory is unconstrained regardless of architecture. Return to ODE-TPP when a dataset with longitudinal biomarker series is available.
