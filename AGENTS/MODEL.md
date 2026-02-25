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

# For parametric losses (hawkes, hawkes_weibull, weibull)
self.param_head = ParametricHead(n_embd, vocab_size)  # outputs distribution params

# For cluster losses (homo_cluster_poisson)
self.aux_head = nn.Linear(config.n_embd, 1)  # outputs auxiliary rate
```

All parametric losses (`hawkes`, `hawkes_weibull`, `weibull`) share the unified `self.param_head` attribute — the head class varies by loss type.

### ParametricHead

Used when the loss requires learned distribution parameters rather than direct intensities:

```python
class ParametricHead(nn.Module):
    def __init__(self, n_embd: int, vocab_size: int):
        super().__init__()
        self.proj_alpha = nn.Linear(n_embd, vocab_size)
        self.proj_beta = nn.Linear(n_embd, vocab_size)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        param_alpha = F.softplus(self.proj_alpha(x))
        param_beta = F.softplus(self.proj_beta(x)) + 0.1
        return {"alpha": param_alpha, "beta": param_beta}
```

**Note**: `F.softplus` ensures positivity. Add small constants (e.g., `+ 0.1`) to prevent numerical issues when parameters must be strictly positive.

### HawkesWeibullHead

Combines **global Weibull baseline parameters** with **context-dependent excitation parameters**. This is a deliberate architectural split:

- **Weibull params** (k, λ, A) are `nn.Parameter` — one value per event type, shared across all patients and positions. They represent population-level age-dependent disease incidence profiles that don't depend on individual patient history.
- **Excitation params** (α, β) are linear projections of the transformer hidden state — context-dependent per position. They capture how a patient's specific history modulates short-term event clustering.

```python
class HawkesWeibullHead(nn.Module):
    def __init__(self, n_embd: int, vocab_size: int):
        super().__init__()
        self.log_k = nn.Parameter(torch.zeros(vocab_size))    # Weibull shape
        self.log_lam = nn.Parameter(torch.zeros(vocab_size))  # Weibull scale
        self.log_A = nn.Parameter(torch.zeros(vocab_size))    # Weibull amplitude
        self.proj_alpha = nn.Linear(n_embd, vocab_size)
        self.proj_beta = nn.Linear(n_embd, vocab_size)

    def forward(self, x, age) -> dict:
        # age is passed for interface consistency but Weibull evaluation
        # happens in the NLL function (which needs both age and targets_age)
        ...
```

The head accepts `age` in its forward signature even though Weibull baseline evaluation happens in the NLL function. This is because the NLL needs both `age` (interval start) and `targets_age` (interval end) to compute the compensator integral, and `targets_age` is not available during the forward pass when generating.

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

Note: There is no explicit $\mu_v$ term because baseline intensity is absorbed into the context-dependent $\alpha_v$.


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

Parametric TPP with exponential decay kernel and **no explicit baseline intensity**:

$$\lambda_v(\tau) = \alpha_v \cdot \exp(-\beta_v \cdot (\tau - t_i))$$

Uses `ParametricHead` to output `alpha` (excitation) and `beta` (decay rate).

```python
def nll_hawkes(
    alpha: torch.Tensor,       # (B, L, V) excitation amplitude
    beta: torch.Tensor,        # (B, L, V) decay rate
    age: torch.Tensor,         # (B, L)
    targets_age: torch.Tensor, # (B, L)
    targets: torch.Tensor,     # (B, L)
    time_unit: float = 365.25,
) -> torch.Tensor:
    # Part 1: log λ_k(t_{i+1}) = log(α_k) - β_k · Δt
    # Part 2: -∫ Σ_v λ_v dτ = -Σ_v (α_v/β_v)·[1 - exp(-β_v·Δt)]
    ...
```

**Limitation — no-event tokens still needed**: Because the intensity is purely self-exciting with no baseline, the exponential kernel decays toward zero over long gaps between events. A patient with no events for 20 years would have near-zero intensity for all event types, even though their age-dependent risk may be increasing. The transformer can partially compensate by outputting large α values, but this conflates two distinct phenomena: age-dependent baseline risk and event-triggered excitation. The `hawkes_weibull` loss was developed to address this.

**Sampling not implemented**: `sample_next` raises `NotImplementedError` for this loss.

### `hawkes_weibull`

**Motivation**: The pure `hawkes` loss has no explicit baseline intensity — the exponential decay kernel drives intensity toward zero during long event-free intervals. This is a fundamental problem for EHR modeling: a 60-year-old with no recent events still has substantial disease risk, but the exponential kernel can't express this. The only workaround is injecting synthetic no-event tokens, which wastes sequence capacity and introduces an artificial dependency on the token insertion strategy.

The `hawkes_weibull` loss adds an explicit **age-dependent baseline intensity** using the Weibull distribution, so the model can maintain non-zero intensity over arbitrarily long gaps without no-event tokens.

#### Intensity Function

$$\lambda_v(t) = \underbrace{\mu_v(t)}_{\text{Weibull baseline}} + \underbrace{\alpha_v^{(i)} \cdot e^{-\beta_v^{(i)}(t - t_i)}}_{\text{self-exciting kernel}}$$

The two components serve distinct purposes:
- **Weibull baseline** $\mu_v(t)$: population-level age-dependent disease incidence. "How likely is this disease at this age, regardless of patient history?"
- **Excitation kernel** $\alpha_v^{(i)} e^{-\beta_v^{(i)} \Delta t}$: patient-specific short-term clustering. "Given this patient's recent events, how much more likely is a follow-up?"

#### Why Weibull Density (Not Hazard)

The Weibull **hazard function** $h(t) = (k/\lambda)(t/\lambda)^{k-1}$ is monotonic — always increasing or always decreasing. But disease incidence often peaks at specific ages (e.g., childhood infections peak early, cardiovascular disease peaks late, some cancers peak mid-life).

The Weibull **density function** (unnormalized) is non-monotonic for $k > 1$:

$$\mu_v(t) = A_v \cdot \frac{k_v}{\lambda_v}\left(\frac{t}{\lambda_v}\right)^{k_v-1} \exp\left(-\left(\frac{t}{\lambda_v}\right)^{k_v}\right)$$

It peaks at $t = \lambda_v \cdot ((k_v - 1)/k_v)^{1/k_v}$. By varying $k_v$ and $\lambda_v$ per event type, the model can place the peak of baseline risk at different ages:
- **Small $\lambda$, moderate $k$** → peaks early in life
- **Large $\lambda$, large $k$** → peaks late in life
- **Medium $\lambda$, large $k$** → peaks mid-life

The additional amplitude parameter $A_v$ controls the overall scale of spontaneous incidence.

#### Why Global Weibull Parameters

The Weibull parameters $(k_v, \lambda_v, A_v)$ are **global `nn.Parameter`s** (per event type, shared across all patients/positions), not transformer outputs. This is intentional:

1. **Baseline risk is a population property**: The age profile of type 2 diabetes incidence is roughly the same across the population — it's the spontaneous rate given only age. Patient-specific modulation belongs in the excitation kernel.
2. **Identifiability**: If both baseline and excitation were context-dependent, the model could collapse one into the other. Keeping the baseline global forces the model to learn a clean decomposition.
3. **Regularization**: With ~1270 event types × 3 params = ~3810 global parameters, the Weibull baseline is a lightweight addition. Making them context-dependent would add 3 × (n_embd × vocab_size) ≈ 450K parameters.

#### NLL Derivation

**Part 1 — event log-likelihood** (log of combined intensity at target time):
$$\log[\mu_k(t_{i+1}) + \alpha_k \cdot e^{-\beta_k \Delta t}]$$

Unlike pure Hawkes where this decomposes into $\log \alpha_k - \beta_k \Delta t$, the sum inside the log means we can't simplify further. This is computed directly with an epsilon for numerical safety.

**Part 2 — Weibull compensator** (closed-form integral of the Weibull density):
$$\int_{t_i}^{t_{i+1}} \mu_v(\tau) \, d\tau = A_v \left[e^{-(t_i/\lambda_v)^{k_v}} - e^{-(t_{i+1}/\lambda_v)^{k_v}}\right]$$

This is simply $A_v \cdot [S(t_i) - S(t_{i+1})]$ where $S$ is the Weibull survival function. The closed form is a key advantage of choosing Weibull — no numerical quadrature needed.

**Part 3 — excitation compensator** (identical to pure Hawkes):
$$\sum_v \frac{\alpha_v}{\beta_v}\left(1 - e^{-\beta_v \Delta t}\right)$$

**Full NLL:**
$$-\ell = -\log[\mu_k(t_{i+1}) + \alpha_k e^{-\beta_k \Delta t}] + \sum_v \left[A_v \left(e^{-(t_i/\lambda_v)^{k_v}} - e^{-(t_{i+1}/\lambda_v)^{k_v}}\right) + \frac{\alpha_v}{\beta_v}(1 - e^{-\beta_v \Delta t})\right]$$

#### Numerical Stability

- $(t/\lambda)^k$ is computed in log-space: $\exp(k \cdot \log(t/\lambda))$ to avoid overflow for large $k$
- The Weibull baseline is computed in log-space then exponentiated, to handle the product of many terms
- All times are clamped to $\epsilon$ before taking logs (ages near zero)
- Weibull parameters use `F.softplus() + 0.1` to keep strictly positive and bounded away from zero

#### Usage

```python
config = Delphi2MConfig(
    loss="hawkes_weibull",
    time_unit=365.25,  # normalize to years (recommended for Weibull stability)
)
model = Delphi2M(config)
```

**Sampling not yet implemented**: `sample_next` raises `NotImplementedError` for this loss. When implementing, note that sampling from the sum of a Weibull baseline and exponential kernel requires either thinning (rejection sampling) or numerical inversion of the compensator — there is no simple inverse CDF.

### `weibull`

The kernel is the **pure Weibull PDF** parameterised by shape $k$ and scale $\lambda$:

$$\lambda_v(\Delta t) = \frac{k_v}{\lambda_v}\left(\frac{\Delta t}{\lambda_v}\right)^{k_v-1} \exp\left(-\left(\frac{\Delta t}{\lambda_v}\right)^{k_v}\right)$$

Where $\Delta t = t - t_i$ is the time since the previous event (not absolute age).

Unlike `hawkes_weibull` where the Weibull operates on absolute age, here it operates on **inter-event time**. This means:
- **$k < 1$**: intensity peaks immediately after the event and decays (heavy-tailed, slower than exponential)
- **$k = 1$**: recovers the exponential kernel (equivalent to Hawkes with $\beta = 1/\lambda$)
- **$k > 1$**: intensity peaks *after a delay* then decays — captures mid/long-range triggering that exponential kernels cannot express

Since the Weibull PDF integrates to 1 over $[0, \infty)$, the compensator has a simple interpretation: the expected number of triggered events of each type is at most 1.

#### Head

Both parameters are **context-dependent** (linear projections of the transformer hidden state):

```python
class WeibullHead(nn.Module):
    def __init__(self, n_embd: int, vocab_size: int):
        super().__init__()
        self.proj_k = nn.Linear(n_embd, vocab_size)
        self.proj_lam = nn.Linear(n_embd, vocab_size)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        k = F.softplus(self.proj_k(x)) + 0.1      # (B, L, V)
        lam = F.softplus(self.proj_lam(x)) + 0.1  # (B, L, V)
        return {"weibull_k": k, "weibull_lam": lam}
```

#### Compensator (Closed-Form)

$$\int_0^{\Delta t} \lambda_v(\tau) \, d\tau = 1 - \exp\left(-\left(\frac{\Delta t}{\lambda_v}\right)^{k_v}\right)$$

#### NLL Derivation

**Part 1 — event log-likelihood** (computed in log-space for stability):
$$\log \lambda_k(\Delta t) = \log k_k - \log \lambda_k + (k_k - 1) \log(\Delta t / \lambda_k) - (\Delta t / \lambda_k)^{k_k}$$

**Part 2 — compensator** (survival penalty):
$$\sum_v \left[1 - \exp\left(-\left(\frac{\Delta t}{\lambda_v}\right)^{k_v}\right)\right]$$

**Full NLL:**
$$-\ell = -\log k_k + \log \lambda_k - (k_k - 1)\log(\Delta t/\lambda_k) + (\Delta t/\lambda_k)^{k_k} + \sum_v\left[1 - \exp\left(-(\Delta t/\lambda_v)^{k_v}\right)\right]$$

#### Usage

```python
config = Delphi2MConfig(
    loss="weibull",
    time_unit=365.25,  # normalize to years
)
model = Delphi2M(config)
```

**Sampling not yet implemented**: `sample_next` raises `NotImplementedError`.

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

**Comparability caveat**: Models trained with different `mask_ties` settings are **not directly comparable**. `mask_ties=False` was used by mistake when training early `weibull`/`hawkes` models — those models saw Δt=0 positions during training, which for Weibull yields PDF(0)=0 → log(0)=−∞ and corrupts the NLL. It is not a meaningful design choice; the correct setting for all non-cluster losses is `mask_ties=True`.

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

### `find_nearest_pred`: Nearest-Previous-Timestep Lookup

Used to align model outputs (defined at event positions) with arbitrary evaluation timesteps:

```python
def find_nearest_pred(
    timesteps: torch.Tensor,              # (B, L) event timestamps in days
    target_timesteps: float | torch.Tensor,  # scalar, (B,), or (B, T)
) -> torch.Tensor:                         # (B, T) indices into seq dim
    """
    For each target timestep, returns the index of the latest event that
    occurred strictly before it (i.e., the nearest previous event).

    Supports scalar, 1-D (B,), and 2-D (B, T) target_timesteps via broadcasting.
    Future/current positions (time_since <= 0) are masked to infinity before argmin.
    """
```

**Usage**: pass the returned indices to `torch.gather` or `torch.take_along_dim` to select the model's predictions at the appropriate position for each evaluation time.

### Weibull Utility Functions

Low-level building blocks used by the `weibull` / `hawkes_weibull` losses and the `TPP` evaluation class:

```python
def weibull_cdf(lam, k, t):
    """CDF of Weibull(lam, k) at time t: 1 - exp(-(t/lam)^k)"""

def weibull_integral(lam, k, start, end):
    """Definite integral of Weibull PDF from start to end (= CDF(end) - CDF(start))"""

def weibull_pdf(lam, k, dt, sum=True):
    """
    Weibull PDF: (k/lam)·(dt/lam)^(k-1)·exp(-(dt/lam)^k)
    If sum=True, sums over the last (vocab) dimension and returns (B, L).
    If sum=False, returns per-event-type values (B, L, V).
    """

def weibull_mode(lam, k):
    """
    Mode of Weibull(lam, k): lam·((k-1)/k)^(1/k) for k > 1, else 0.
    Returns shape matching lam/k.
    """

def weibull_max(lam, k):
    """Peak PDF value: weibull_pdf evaluated at the mode. Used as lam_max for thinning."""
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

### `sample_tpp` and `thinning_sample`: Thinning-Based Sampling

```python
def sample_tpp(
    outputs: dict[str, torch.Tensor],
    kernel: str,  # e.g. "weibull"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Samples next inter-event time and event type from a TPP via Lewis-Shedler thinning.

    Currently supports kernel="weibull":
      - Reads weibull_lam and weibull_k from outputs[:, -1, :]
      - Computes lam_max = weibull_max(lam, k) (PDF at the mode, a valid upper bound)
      - Calls thinning_sample to get dt
      - Selects event type via torch.multinomial weighted by per-type PDF at dt

    Returns: (next_idx, dt) each (B, 1)
    """
```

```python
def thinning_sample(
    lam_max: torch.Tensor,   # (B,) upper bound on total intensity
    lam_func,                # callable: (n_active,) -> (n_active,) total intensity
    max_steps: int = 1000,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Lewis-Shedler thinning algorithm for non-homogeneous TPPs.

    Proposes candidate inter-event times from Exp(lam_max) and accepts each
    with probability lam_func(t) / lam_max. Uses the same active-batch management
    as generate(): accepted sequences are removed from the active set, so lam_func
    is called on a shrinking sub-batch at each iteration.

    Raises RuntimeError if any sequence fails to accept within max_steps steps.

    Returns:
        dt:      (B,) sampled inter-event times
        metrics: {"n_steps": (B,), "accept_rate": (B,)}
    """
```

**Requirement for `lam_func`**: must accept a variable-length 1-D tensor of elapsed times and return total intensities of the same shape. It will be called with the currently active sub-batch only.

### Losses without samplers

`hawkes` and `hawkes_weibull` do not have sampling implementations. `sample_next` raises `NotImplementedError` for these losses.

`weibull` now has sampling implemented via `sample_tpp` / `thinning_sample` — `sample_next` calls `sample_tpp(outputs, kernel="weibull")` for this loss.

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
    parametric_losses = {"hawkes", "hawkes_weibull", "weibull", "your_loss"}  # add to set

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

## TPP Evaluation Class (`delphi/model/tpp.py`)

`TPP` is a thin evaluation wrapper that encapsulates intensity and survival computations for post-hoc analysis (e.g. AUC, calibration, likelihood decomposition). It is **not** used during training — its role is to provide a unified interface over the different loss kernels when computing pointwise or integrated quantities over arbitrary evaluation timesteps.

```python
class TPP:
    def __init__(self, loss: str, time_unit: float):
        self.loss = loss          # "default", "homo_poisson", or "weibull"
        self.time_unit = time_unit
```

### Methods

#### `intensity(outputs, eval_timesteps, timesteps)`

Returns the instantaneous intensity for each event type at the requested evaluation times.

```python
def intensity(
    self,
    outputs: dict[str, torch.Tensor],   # forward pass outputs (logits / weibull params)
    eval_timesteps: torch.Tensor,        # (B, L) times to evaluate intensity at (days)
    timesteps: torch.Tensor,             # (B, L) event timestamps (days, = age in model)
) -> torch.Tensor:                       # (B, L, V) intensity per event type
```

- `"default"` / `"homo_poisson"`: returns `exp(logits)` (constant between events)
- `"weibull"`: returns `weibull_pdf(lam, k, dt=eval_timesteps - timesteps, sum=False)`

Times are normalised by `self.time_unit` before computation.

#### `survival_prob(outputs, timesteps, start_age, end_age)`

Returns the survival probability (probability of no event of any type) over an interval:

```python
def survival_prob(
    self,
    outputs: dict[str, torch.Tensor],
    timesteps: torch.Tensor,             # (B, L) last event time (days)
    start_age: float | torch.Tensor,     # interval start (days)
    end_age: float | torch.Tensor,       # interval end (days)
) -> torch.Tensor:                       # (B, L, V) per-type survival probabilities
```

Returns `exp(-∫ λ_v(τ) dτ)` from `start_age` to `end_age`:
- `"default"` / `"homo_poisson"`: `exp(-λ_v · (end - start))`
- `"weibull"`: uses `weibull_integral(lam, k, start=start-timesteps, end=end-timesteps)`

#### `ll_time(outputs, eval_timesteps, timesteps)`

Joint time log-likelihood at `eval_timesteps`:

```python
ll_time = intensity * survival_prob(start_age=timesteps, end_age=eval_timesteps)
```

This is the time density $f(t) = \lambda(t) \cdot S(t_{\text{prev}}, t)$ — the probability that the *next* event fires at exactly `eval_timesteps`.

#### `ll_mark_conditional(outputs, timesteps, eval_timesteps, marks)`

Log conditional probability of the mark (event type) given the event time:

```python
log p(mark | time) = log(λ_mark(t) / Σ_v λ_v(t))
```

`marks` is an integer tensor of shape `(B, L, 1)` indexing into the vocab dimension.

### Design Notes

- All methods accept the full sequence tensors from a forward pass; callers are responsible for selecting the relevant positions (e.g. via `find_nearest_pred`).
- `tpp.py` imports `find_nearest_pred`, `weibull_pdf`, and `weibull_integral` from `delphi/model/utils.py`. `find_nearest_pred` is available but currently commented out in the implementation — it would be used to automatically select the correct model output position for each evaluation timestep.

---

## File Locations

- **Model**: `delphi/model/transformer.py`
- **Config**: `Delphi2MConfig` in same file
- **Loss functions & utilities**: `delphi/model/utils.py`
- **TPP evaluation class**: `delphi/model/tpp.py`
